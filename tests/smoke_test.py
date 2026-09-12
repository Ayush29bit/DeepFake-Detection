"""Smoke tests: prove every stage of the pipeline executes and has the right shape.

These are executability checks, not research results. They run on a handful of
tiny synthetic clips (see tests/make_synthetic_videos.py) and on random tensors,
so no accuracy number produced here means anything about deepfake detection.

    python -m tests.smoke_test
    python -m tests.smoke_test --keep-data     # keep the generated clips
"""

from __future__ import annotations

import argparse
import os
import shutil
import tempfile
import time
import traceback
from typing import Callable, List, Tuple

import numpy as np
import torch
import torch.nn as nn

from evaluation.metrics import compute_metrics, format_metrics
from inference.predict import predict_video
from models.baseline import ResNetBaseline
from models.frequency import FrequencyBranch, FrequencyCNN, fft_log_magnitude
from models.hybrid import HybridDeepfakeDetector, build_model
from preprocessing.dataset import VideoFaceDataset, build_dataloader
from preprocessing.face import FaceExtractor
from preprocessing.splits import (assert_no_leakage, discover_videos, load_manifest,
                                  split_videos, write_manifest,
                                  split_summary)
from preprocessing.video import VideoReader, load_sampled_frames, sample_indices
from tests.make_synthetic_videos import build_dataset
from training.config import Config
from training.train import evaluate, run_epoch

# Small, fast settings: the point is executability, not convergence.
SMOKE = dict(num_frames=8, image_size=224, batch_size=2, d_model=256,
             num_heads=8, num_transformer_layers=2, freq_dim=128)

results: List[Tuple[str, bool, str]] = []


def check(name: str, fn: Callable[[], str]) -> bool:
    started = time.time()
    try:
        detail = fn() or ""
        results.append((name, True, detail))
        print(f"  PASS  {name}  ({time.time() - started:.1f}s)  {detail}")
        return True
    except Exception as exc:
        results.append((name, False, f"{type(exc).__name__}: {exc}"))
        print(f"  FAIL  {name}  ({time.time() - started:.1f}s)")
        traceback.print_exc()
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="",
                        help="where to write synthetic clips (default: a temp dir)")
    parser.add_argument("--keep-data", action="store_true")
    parser.add_argument("--pretrained", action="store_true",
                        help="download ImageNet weights (slower; off by default)")
    args = parser.parse_args()

    tmp_root = args.data_dir or tempfile.mkdtemp(prefix="deepfake_smoke_")
    data_dir = os.path.join(tmp_root, "videos")
    cache_dir = os.path.join(tmp_root, "cache")

    config = Config(model="hybrid", pretrained=args.pretrained, num_workers=0,
                    cache_dir=cache_dir, device="cpu", augment=True,
                    train_jitter=True, **SMOKE)
    device = torch.device("cpu")

    print("Deepfake detection pipeline - smoke tests")
    print(f"  scratch dir: {tmp_root}")
    print(f"  pretrained backbone weights: {args.pretrained}\n")

    state: dict = {}

    # ------------------------------------------------------------- 1. video
    def t01():
        real_dir, fake_dir = build_dataset(data_dir, per_class=4, frames=30)
        state["real_dir"], state["fake_dir"] = real_dir, fake_dir
        with VideoReader(os.path.join(real_dir, "real_00.mp4")) as reader:
            state["frame_count"] = reader.frame_count
            assert reader.frame_count > 0
        return f"opened a clip with {state['frame_count']} frames"

    check("01 video can be opened", t01)

    # ---------------------------------------------------------- 2. sampling
    def t02():
        for total in (5, 8, 30, 80, 301):
            idx = sample_indices(total, 8)
            assert len(idx) == 8, idx
            assert all(0 <= i < total for i in idx), (total, idx)
            assert idx == sorted(idx), idx
        det_a = sample_indices(80, 8)
        det_b = sample_indices(80, 8)
        assert det_a == det_b, "sampling must be deterministic without jitter"
        jit = sample_indices(80, 8, jitter=True, rng=np.random.default_rng(0))
        assert len(jit) == 8
        frames = load_sampled_frames(os.path.join(state["real_dir"], "real_01.mp4"), 8)
        assert len(frames) == 8 and frames[0].ndim == 3
        state["frames"] = frames
        return (f"80-frame video -> indices {det_a}; "
                f"loaded {len(frames)} RGB frames of shape {frames[0].shape}")

    check("02 frames can be sampled (any video length)", t02)

    # ------------------------------------------------------- 3. face crops
    def t03():
        extractor = FaceExtractor("haar", image_size=config.image_size)
        crops, detected = extractor.extract_sequence(state["frames"])
        assert len(crops) == 8, len(crops)
        for crop in crops:
            assert crop.shape == (224, 224, 3), crop.shape
            assert crop.dtype == np.uint8
        blank = [np.zeros((240, 320, 3), np.uint8)] * 8   # forces the fallback
        fallback_crops, fallback_detected = extractor.extract_sequence(blank)
        assert fallback_detected == 0 and len(fallback_crops) == 8
        return (f"{detected}/8 frames had a detected face; the rest used the "
                f"documented fallback; fallback-only sequence still yielded 8 crops")

    check("03 face detection + cropping (with fallback)", t03)

    # ---------------------------------------------------------- 4. dataset
    def t04():
        items = discover_videos(state["real_dir"], state["fake_dir"])
        assert len(items) == 8, len(items)
        state["items"] = items
        dataset = VideoFaceDataset(items, config, train=True)
        frames, label = dataset[0]
        assert frames.shape == (8, 3, 224, 224), frames.shape
        assert frames.dtype == torch.float32
        assert label.item() in (0.0, 1.0)
        state["dataset"] = dataset
        return f"item shape {tuple(frames.shape)}, label {label.item()}"

    check("04 Dataset returns (NUM_FRAMES, 3, 224, 224) + label", t04)

    # ------------------------------------------------------- 5. dataloader
    def t05():
        eval_dataset = VideoFaceDataset(state["items"], config, train=False)
        loader = build_dataloader(eval_dataset, config, shuffle=False)
        frames, labels = next(iter(loader))
        assert frames.shape == (config.batch_size, 8, 3, 224, 224), frames.shape
        assert labels.shape == (config.batch_size,), labels.shape
        state["batch"] = (frames, labels)
        state["loader"] = loader

        # The crop cache must actually be written, and a cached reload must
        # return the identical crops (a silently failing cache would make every
        # epoch redo face detection, or worse, substitute blank frames).
        files = os.listdir(cache_dir)
        cached = [f for f in files if f.endswith(".npy") and ".tmp" not in f]
        leftover = [f for f in files if ".tmp" in f]
        assert cached, "no face-crop cache was written"
        assert not leftover, f"temporary cache files left behind: {leftover}"
        first, _ = eval_dataset.load_crops(state["items"][0])
        second, flag = eval_dataset.load_crops(state["items"][0])
        assert flag == -1, "second load did not come from the cache"
        assert np.array_equal(first, second), "cached crops differ from fresh crops"
        assert second.any(), "cached crops are entirely blank"
        return (f"batch {tuple(frames.shape)}, labels {tuple(labels.shape)}; "
                f"{len(cached)} face-crop caches written and verified")

    check("05 DataLoader yields a batch", t05)

    # ------------------------------------------------------ 6. video split
    def t06():
        splits = split_videos(state["items"], 0.70, 0.15, 0.15, seed=42)
        assert_no_leakage(splits)
        ids = [i.video_id for group in splits.values() for i in group]
        assert len(ids) == len(set(ids)) == len(state["items"])
        again = split_videos(state["items"], 0.70, 0.15, 0.15, seed=42)
        assert [i.video_id for i in again["train"]] == \
               [i.video_id for i in splits["train"]], "split must be reproducible"
        # Manifest round-trip: a written split must reload to the same videos.
        manifest = os.path.join(tmp_root, "split_train.csv")
        write_manifest(splits["train"], manifest)
        reloaded = load_manifest(manifest)
        assert [(i.video_id, i.label) for i in reloaded] == \
               [(i.video_id, i.label) for i in splits["train"]], "manifest mismatch"

        state["splits"] = splits
        return ("no video appears in two splits; same seed reproduces the split; "
                "manifest round-trips\n" + split_summary(splits))

    check("06 video-level split, no leakage, reproducible", t06)

    # --------------------------------------------------------- 7. baseline
    def t07():
        model = ResNetBaseline(backbone="resnet18", pretrained=config.pretrained)
        frames, _ = state["batch"]
        with torch.no_grad():
            logits = model(frames)
            per_frame = model.frame_logits(frames)
        assert logits.shape == (config.batch_size,), logits.shape
        assert per_frame.shape == (config.batch_size, 8), per_frame.shape
        dummy = torch.randn(1, 8, 3, 224, 224)
        with torch.no_grad():
            assert model(dummy).shape == (1,)
        state["baseline"] = model
        return f"per-frame logits {tuple(per_frame.shape)} -> video logit {tuple(logits.shape)}"

    check("07 ResNet18 baseline accepts a batch", t07)

    # --------------------------------------------------------------- 8. FFT
    def t08():
        frames, _ = state["batch"]
        flat = frames.flatten(0, 1)
        spectrum = fft_log_magnitude(flat)
        assert spectrum.shape == (flat.shape[0], 1, 224, 224), spectrum.shape
        assert torch.isfinite(spectrum).all()
        assert float(spectrum.min()) >= 0.0 and float(spectrum.max()) <= 1.0
        state["spectrum"] = spectrum
        return (f"{tuple(flat.shape)} -> {tuple(spectrum.shape)}, "
                f"range [{float(spectrum.min()):.3f}, {float(spectrum.max()):.3f}]")

    check("08 FFT log-magnitude representation", t08)

    # ---------------------------------------------------- 9. frequency CNN
    def t09():
        cnn = FrequencyCNN(out_dim=config.freq_dim)
        embed = cnn(state["spectrum"])
        assert embed.shape == (state["spectrum"].shape[0], config.freq_dim), embed.shape
        branch = FrequencyBranch(out_dim=config.freq_dim)
        frames, _ = state["batch"]
        branch_embed = branch(frames.flatten(0, 1))
        assert branch_embed.shape == (frames.shape[0] * 8, config.freq_dim)
        params = sum(p.numel() for p in cnn.parameters())
        return (f"frequency embedding {tuple(branch_embed.shape)}; "
                f"CNN has {params:,} parameters (lightweight)")

    check("09 frequency CNN produces embeddings", t09)

    # ----------------------------------------------- 10. hybrid + fusion
    def t10():
        model = build_model(config)
        assert isinstance(model, HybridDeepfakeDetector)
        frames, _ = state["batch"]
        with torch.no_grad():
            fused = model.frame_embeddings(frames)
            seq = model.sequence_features(frames)
            logits = model(frames)
        assert fused.shape == (config.batch_size, 8, config.d_model), fused.shape
        assert seq.shape == (config.batch_size, 8, config.d_model), seq.shape
        assert logits.shape == (config.batch_size,), logits.shape
        assert not torch.equal(fused, seq), "the Transformer must change the sequence"
        state["hybrid"] = model
        return (f"fusion {tuple(fused.shape)} -> transformer {tuple(seq.shape)} "
                f"-> mean-pool -> logit {tuple(logits.shape)}")

    check("10 hybrid model: fusion + Transformer + head", t10)

    # ------------------------------------------------------- 11. ablations
    def t11():
        frames, _ = state["batch"]
        shapes = {}
        for name, kwargs in (
                ("spatial only", dict(use_frequency=False, use_temporal=False)),
                ("spatial+freq", dict(use_frequency=True, use_temporal=False)),
                ("full hybrid", dict(use_frequency=True, use_temporal=True))):
            model = HybridDeepfakeDetector(pretrained=False, freq_dim=config.freq_dim,
                                           d_model=config.d_model, **kwargs)
            with torch.no_grad():
                shapes[name] = tuple(model(frames).shape)
            assert shapes[name] == (config.batch_size,)
        return "; ".join(f"{k} -> {v}" for k, v in shapes.items())

    check("11 ablation configurations run", t11)

    # ------------------------------------------------------------ 12. loss
    def t12():
        criterion = nn.BCEWithLogitsLoss()
        frames, labels = state["batch"]
        logits = state["hybrid"](frames)
        loss = criterion(logits, labels)
        assert loss.ndim == 0 and torch.isfinite(loss)
        state["loss_value"] = float(loss)
        return f"BCEWithLogitsLoss = {float(loss):.4f}"

    check("12 loss can be computed", t12)

    # --------------------------------------------------- 13. backward pass
    def t13():
        model = state["hybrid"]
        model.zero_grad()
        frames, labels = state["batch"]
        loss = nn.BCEWithLogitsLoss()(model(frames), labels)
        loss.backward()
        with_grad = [n for n, p in model.named_parameters()
                     if p.grad is not None and torch.any(p.grad != 0)]
        assert with_grad, "no parameter received a gradient"
        for key in ("fusion.weight", "head.weight"):
            grad = dict(model.named_parameters())[key].grad
            assert grad is not None and torch.isfinite(grad).all()
        model.zero_grad()
        return f"{len(with_grad)} tensors received non-zero gradients"

    check("13 backpropagation works", t13)

    # ------------------------------------------------- 14. one train step
    def t14():
        model = state["hybrid"]
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        before = model.head.weight.detach().clone()
        loss, y_true, y_prob = run_epoch(model, state["loader"],
                                         nn.BCEWithLogitsLoss(), device, optimizer)
        after = model.head.weight.detach()
        assert not torch.equal(before, after), "weights did not update"
        assert len(y_true) == len(state["items"]) == len(y_prob)
        assert all(0.0 <= p <= 1.0 for p in y_prob)
        return f"one training epoch over {len(y_true)} videos, mean loss {loss:.4f}"

    check("14 one training step / epoch completes", t14)

    # ------------------------------------------------- 15. validation step
    def t15():
        metrics = evaluate(state["hybrid"], state["loader"], nn.BCEWithLogitsLoss(),
                           device, threshold=0.5)
        for key in ("loss", "accuracy", "precision", "recall", "f1",
                    "confusion_matrix"):
            assert key in metrics, key
        state["metrics"] = metrics
        return f"validation pass produced {sorted(k for k in metrics)}"

    check("15 one validation step completes", t15)

    # ---------------------------------------------------------- 16. metrics
    def t16():
        y_true = [0, 0, 1, 1, 0, 1]
        y_prob = [0.1, 0.4, 0.9, 0.6, 0.7, 0.2]
        m = compute_metrics(y_true, y_prob, threshold=0.5)
        assert abs(m["accuracy"] - 4 / 6) < 1e-9, m["accuracy"]
        assert abs(m["precision"] - 2 / 3) < 1e-9, m["precision"]
        assert abs(m["recall"] - 2 / 3) < 1e-9, m["recall"]
        assert abs(m["f1"] - 2 / 3) < 1e-9, m["f1"]
        assert m["roc_auc"] is not None
        cm = m["confusion_matrix"]
        assert (cm["tn"], cm["fp"], cm["fn"], cm["tp"]) == (2, 1, 1, 2), cm
        single = compute_metrics([1, 1], [0.9, 0.8])
        assert single["roc_auc"] is None, "ROC-AUC must be None for a single class"
        assert format_metrics(m)
        return "metrics match hand-computed values on a known example"

    check("16 metrics are correct on a known example", t16)

    # -------------------------------------------------------- 17. inference
    def t17():
        model = state["hybrid"]
        video = os.path.join(state["fake_dir"], "fake_00.mp4")
        prediction = predict_video(video, model, config, device=device,
                                   return_crops=True)
        assert prediction.label in ("REAL", "FAKE")
        assert 0.0 <= prediction.fake_probability <= 1.0
        assert 0.5 <= prediction.confidence <= 1.0
        assert prediction.face_crops.shape == (8, 224, 224, 3)
        expected = ("FAKE" if prediction.fake_probability >= 0.5 else "REAL")
        assert prediction.label == expected
        assert prediction.summary()
        state["prediction"] = prediction
        return (f"{prediction.label} at {prediction.confidence * 100:.1f}% confidence "
                f"(untrained model - the value is meaningless, the path works)")

    check("17 end-to-end inference on a video file", t17)

    # ------------------------------------------------- 18. checkpoint cycle
    def t18():
        path = os.path.join(tmp_root, "smoke_ckpt.pt")
        torch.save({"model_state": state["hybrid"].state_dict(),
                    "config": config.to_dict(), "epoch": 1}, path)
        loaded = torch.load(path, map_location="cpu", weights_only=False)
        restored = build_model(Config(**loaded["config"]))
        restored.load_state_dict(loaded["model_state"])
        restored.eval()
        state["hybrid"].eval()
        frames, _ = state["batch"]
        with torch.no_grad():
            a, b = state["hybrid"](frames), restored(frames)
        assert torch.allclose(a, b, atol=1e-6), "restored model disagrees"
        return "checkpoint saved, reloaded, and reproduces identical logits"

    check("18 checkpoint save/load round-trip", t18)

    # ------------------------------------------------------- 19. Streamlit UI
    def t19():
        from streamlit.testing.v1 import AppTest

        # AppTest resolves relative paths against this file, so pass an absolute
        # one rooted at the repository.
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        app_path = os.path.join(repo_root, "app", "app.py")
        app = AppTest.from_file(app_path, default_timeout=180).run()
        assert app.exception == [], f"the app raised: {app.exception}"
        assert [t.value for t in app.title] == ["Deepfake Detection"], app.title
        if app.error:                      # no checkpoint trained yet
            assert "checkpoint" in app.error[0].value.lower()
            return "app runs and reports that no checkpoint is available yet"
        assert app.selectbox, "no checkpoint selector rendered"
        return (f"app runs; {len(app.selectbox[0].options)} checkpoint(s) "
                f"offered, uploader rendered")

    check("19 Streamlit app script runs", t19)

    # ------------------------------------------------------------- summary
    passed = sum(1 for _n, ok, _d in results if ok)
    print("\n" + "=" * 68)
    print(f"SMOKE TESTS: {passed}/{len(results)} passed")
    for name, ok, detail in results:
        if not ok:
            print(f"  FAILED: {name} -> {detail}")
    print("=" * 68)
    print("These tests prove the pipeline executes end to end. They use tiny "
          "synthetic clips\nand random weights, so they say nothing about "
          "detection accuracy.")

    if not args.keep_data and not args.data_dir:
        shutil.rmtree(tmp_root, ignore_errors=True)
    else:
        print(f"\nscratch data kept at {tmp_root}")

    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
