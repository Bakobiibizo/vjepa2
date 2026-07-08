"""L0 measurement: ACTUAL V-JEPA 2 predictor error vs distance-from-centroid.

Tightens the previous L1 (which used identity + linear proxies) to use the real
JEPA predictor from vitg.pt. Tests the same hypothesis: prediction error should
increase monotonically with distance from centroid.

Protocol:
- 1024 tokens per clip (4 temporal x 16x16 spatial).
- Fixed mask: context = first 3 temporal blocks (768 tokens), target = last
  temporal block (256 tokens). Same mask on every clip -> comparable error.
- predictor(context_tokens, masks_enc, masks_pred) -> predicted target latents.
- target_encoder(full clip) -> actual target latents.
- error per clip = mean |predicted - actual| over the 256 target tokens.
- Bin by context clip's distance to its cluster centroid.

This is the actual JEPA prediction objective, not a proxy.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import src.hub.backbones as _backbones
_backbones.VJEPA_BASE_URL = "https://dl.fbaipublicfiles.com/vjepa2"

LATENTS = REPO / "results" / "02_frozen_encoder" / "clip_pooled_latents.npy"
META = REPO / "results" / "02_frozen_encoder" / "clip_meta.json"
DATA_ROOT = REPO / "data" / "droid_100"
OUT = REPO / "results" / "02_frozen_encoder"
SEED = 0
K = 32
CROP = 256
PATCH = 16
TUBELET = 2
NUM_FRAMES = 8
GRID_H = CROP // PATCH  # 16
GRID_W = CROP // PATCH  # 16
TEMPORAL = NUM_FRAMES // TUBELET  # 4
TOKENS_PER_TEMPORAL = GRID_H * GRID_W  # 256
TOTAL_TOKENS = TEMPORAL * TOKENS_PER_TEMPORAL  # 1024
MAX_CLIPS = 600  # predictor forward is expensive; bounded sample


def load_encoder_predictor(device):
    from src.hub.backbones import _make_vjepa2_model
    encoder, predictor = _make_vjepa2_model(
        model_name="vit_giant", checkpoint_key="target_encoder",
        img_size=CROP, patch_size=PATCH, tubelet_size=TUBELET, num_frames=NUM_FRAMES,
        pretrained=True,
    )
    encoder = encoder.to(device).eval()
    predictor = predictor.to(device).eval()
    for p in list(encoder.parameters()) + list(predictor.parameters()):
        p.requires_grad = False
    return encoder, predictor


def make_transform():
    from evals.video_classification_frozen.utils import make_transforms
    return make_transforms(crop_size=CROP, training=False)


def load_centroids():
    pooled = np.load(LATENTS)
    meta = json.load(open(META))
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=K, n_init=10, random_state=SEED).fit(pooled)
    labels = km.labels_
    centers = km.cluster_centers_
    dist_to_centroid = np.linalg.norm(pooled - centers[labels], axis=1)
    return pooled, meta, labels, dist_to_centroid


def get_offsets():
    import pandas as pd
    df = pd.read_parquet(DATA_ROOT / "data" / "chunk-000" / "file-000.parquet")
    ep_lens = df.groupby("episode_index").size().to_dict()
    offsets = {}
    cur = 0
    for ep in sorted(ep_lens.keys()):
        offsets[int(ep)] = cur
        cur += int(ep_lens[ep])
    return offsets


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    print("loading models...")
    encoder, predictor = load_encoder_predictor(device)
    transform = make_transform()
    print("loading centroids...")
    pooled, meta, labels, dist_to_centroid = load_centroids()
    offsets = get_offsets()
    mp4 = DATA_ROOT / "videos" / "observation.images.exterior_image_1_left" / "chunk-000" / "file-000.mp4"

    # fixed mask: context = first 3 temporal blocks (indices 0..767), target = last block (768..1023)
    # token index = t * 256 + spatial, so temporal blocks are contiguous: [0,256),[256,512),[512,768),[768,1024)
    # Mask tensors must be [B, K], even for B=1. A 1-D mask is interpreted
    # as B=K by apply_masks(), which corrupts RoPE positions and produces a
    # tensor-size mismatch in rotate_queries_or_keys.
    masks_enc = [torch.arange(0, 768, device=device).long().unsqueeze(0)]      # context indices
    masks_pred = [torch.arange(768, 1024, device=device).long().unsqueeze(0)]  # target indices

    # sample clips
    rng = np.random.default_rng(SEED)
    clip_indices = rng.choice(len(meta), size=min(MAX_CLIPS, len(meta)), replace=False)

    from decord import VideoReader, cpu

    results = []  # (clip_idx, dist_to_centroid, pred_error)
    done = 0
    with torch.no_grad():
        for ci in clip_indices:
            m = meta[ci]
            ep = m["episode"]
            start = m["clip_start_frame"]
            global_frames = [offsets[ep] + start + i for i in range(NUM_FRAMES)]
            try:
                vr = VideoReader(str(mp4), num_threads=1, ctx=cpu(0))
                vlen = len(vr)
                indices = [min(i, vlen - 1) for i in global_frames]
                frames = vr.get_batch(indices).asnumpy()
                clip = transform([frames[i] for i in range(len(indices))])[0].unsqueeze(0).to(device)
            except Exception as e:
                print(f"skip clip {ci}: {e}")
                continue

            # context: encode with masks_enc (returns only context tokens)
            z_ctx = encoder(clip, masks_enc)  # list[[1, 768, 1408]]
            # target: encode full clip, then we'll index the target tokens
            z_full = encoder(clip)  # list[[1, 1024, 1408]]
            z_full = z_full[0] if isinstance(z_full, list) else z_full
            z_target = z_full[0, 768:1024, :]  # [256, 1408]

            # predictor returns predicted target latents
            z_pred = predictor(z_ctx, masks_enc, masks_pred)  # list[[1, 256, 1408]]
            z_pred = z_pred[0] if isinstance(z_pred, list) else z_pred
            z_pred = z_pred[0]  # [256, 1408]

            # layer-norm target like training does
            z_target_ln = torch.nn.functional.layer_norm(z_target, (z_target.size(-1),))

            error = (z_pred - z_target_ln.unsqueeze(0)).abs().mean().item()
            results.append((int(ci), float(dist_to_centroid[ci]), error))

            done += 1
            if done % 50 == 0:
                print(f"  {done}/{len(clip_indices)}", flush=True)
            del clip, z_ctx, z_full, z_target, z_pred
            torch.cuda.empty_cache()

    print(f"\ncomputed {len(results)} prediction errors")
    errors = np.array([r[2] for r in results])
    dists = np.array([r[1] for r in results])

    # bin by dist-to-centroid
    print(f"\n=== ACTUAL V-JEPA predictor error by distance-from-centroid ===")
    print(f"{'dist bin':>20} {'n':>5} {'mean_err':>10} {'median':>8}")
    n_bins = 8
    bins = np.linspace(dists.min(), dists.max(), n_bins + 1)
    binned = []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (dists >= lo) & (dists < hi if i < n_bins - 1 else dists <= hi)
        if mask.sum() < 3:
            continue
        e = errors[mask]
        row = {"dist_lo": float(lo), "dist_hi": float(hi), "n": int(mask.sum()),
               "mean_err": float(e.mean()), "median_err": float(np.median(e))}
        binned.append(row)
        print(f"  [{lo:5.1f}, {hi:5.1f})   {row['n']:>5} {row['mean_err']:>10.4f} {row['median_err']:>8.4f}")

    r = float(np.corrcoef(dists, errors)[0, 1])
    print(f"  Pearson r(dist, predictor_error) = {r:.3f}")
    print(f"  overall: error range {errors.min():.4f} - {errors.max():.4f}, mean {errors.mean():.4f}")

    report = {
        "k": K, "model": "V-JEPA 2 vitg predictor (masked latent prediction)",
        "mask": "context=first 3 temporal blocks (768 tokens), target=last temporal block (256 tokens)",
        "n_clips_measured": len(results),
        "binned": binned,
        "pearson_r": r,
        "error_min": float(errors.min()), "error_max": float(errors.max()), "error_mean": float(errors.mean()),
    }
    out = OUT / "predictor_error_vs_centroid.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
