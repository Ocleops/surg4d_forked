#!/usr/bin/env python3
import argparse
import os
import sys
from typing import List

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sanity-check CLIP per-segment features and compute text similarities"
    )
    parser.add_argument(
        "--features",
        type=str,
        required=True,
        help="Path to *_f.npy (features) file",
    )
    parser.add_argument(
        "--segmap",
        type=str,
        default=None,
        help="Optional path to *_s.npy (seg map indices). If provided, will validate indices vs features",
    )
    parser.add_argument(
        "--layer_index",
        type=int,
        default=0,
        help="Layer index for segmap (0 for default if segmap is (L,H,W))",
    )
    parser.add_argument(
        "--prompts",
        type=str,
        default=(
            "an abdominal wall, a liver, gastrointestinal tract, fat, "
            "a grasper, connective tissue, blood, cystic duct, l-hook electrocautery, "
            "a gallbladder, hepatic vein, liver ligament"
        ),
        help="Comma-separated list of prompts",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="ViT-B-16",
        help="open_clip model type",
    )
    parser.add_argument(
        "--pretrained",
        type=str,
        default="laion2b_s34b_b88k",
        help="open_clip pretrained tag",
    )
    parser.add_argument(
        "--topk", type=int, default=3, help="Top-k prompts per segment to print"
    )
    return parser.parse_args()


def load_features(features_path: str) -> np.ndarray:
    if not os.path.isfile(features_path):
        raise FileNotFoundError(features_path)
    feats = np.load(features_path)
    if feats.ndim != 2:
        raise ValueError(f"Expected 2D features (N,512); got shape {feats.shape}")
    return feats.astype(np.float32)


def load_segmap(segmap_path: str, layer_index: int) -> np.ndarray:
    if not os.path.isfile(segmap_path):
        raise FileNotFoundError(segmap_path)
    seg = np.load(segmap_path)
    if seg.ndim == 3:
        if not (0 <= layer_index < seg.shape[0]):
            raise IndexError(
                f"layer_index {layer_index} out of range for seg shape {seg.shape}"
            )
        seg = seg[layer_index]
    elif seg.ndim == 2:
        pass
    else:
        raise ValueError(f"Unsupported seg ndim: {seg.ndim}")
    return seg


def check_norms(features: np.ndarray) -> None:
    norms = np.linalg.norm(features, axis=1)
    print(
        "Feature L2 norms:",
        f"min={norms.min():.4f}",
        f"median={np.median(norms):.4f}",
        f"mean={norms.mean():.4f}",
        f"max={norms.max():.4f}",
    )


def validate_indices(features: np.ndarray, segmap: np.ndarray) -> None:
    ids = np.unique(segmap)
    valid_ids = ids[ids >= 0]
    print(
        f"Segmap unique ids (incl -1)={len(ids)}; min={ids.min()} max={ids.max()} valid_count={len(valid_ids)}"
    )
    expected = set(range(features.shape[0]))
    present = set(int(v) for v in valid_ids.tolist())
    missing = sorted(list(expected - present))
    if missing:
        print(
            "WARNING: missing ids not present in segmap:",
            missing[:20],
            ("..." if len(missing) > 20 else ""),
        )
    # area per id
    areas = []
    for i in range(features.shape[0]):
        areas.append(int((segmap == i).sum()))
    areas = np.array(areas)
    if areas.size:
        print(
            "Areas per id (pixels):",
            f"min={areas.min()}",
            f"median={int(np.median(areas))}",
            f"mean={int(areas.mean())}",
            f"max={areas.max()}",
        )


def compute_text_similarities(
    features: np.ndarray,
    prompts: List[str],
    model_name: str,
    pretrained: str,
    topk: int,
) -> None:
    try:
        import torch
        import surgvlp
    except Exception as e:
        print("ERROR: requires SurgVLP and torch installed", file=sys.stderr)
        raise

    from mmengine.config import Config

    device = "cuda" if torch.cuda.is_available() else "cpu"
    configs = Config.fromfile('../SurgVLP/tests/config_surgvlp.py')['config']
    # Change the config file to load different models: config_surgvlp.py / config_hecvl.py / config_peskavlp.py

    model, preprocess = surgvlp.load(configs.model_config, device=device)

    model.eval()
    tokenizer = surgvlp.tokenize
    model = model.to(device)
    # model, _, _ = open_clip.create_model_and_transforms(
    #     model_name, pretrained=pretrained, precision="fp16"
    # )
    # model = model.eval().to(device)
    # tokenizer = open_clip.get_tokenizer(model_name)

    with torch.no_grad():
        tok_phrases = [tokenizer(phrase, device=device) for phrase in prompts]
        text = torch.cat(
            [model(None, tok_phrase, mode='text')['text_emb'] for tok_phrase in tok_phrases]
        ).to("cuda")
        # text = torch.cat([tokenizer(p) for p in prompts]).to(device)
        # text = model.encode_text(tok)
        text = text / text.norm(dim=-1, keepdim=True)
        T = text.float().cpu().numpy()  # (P,512)

    # Features are expected L2-normalized already
    sims = features @ T.T  # (N,P)
    for i in range(sims.shape[0]):
        top_idx = sims[i].argsort()[-topk:][::-1]
        result = [(prompts[j], float(sims[i, j])) for j in top_idx]
        print(f"segment {i}: {result}")


if __name__ == "__main__":
    args = parse_args()
    feats = load_features(args.features)
    print("Loaded features:", feats.shape, feats.dtype)
    check_norms(feats)

    if args.segmap:
        seg = load_segmap(args.segmap, args.layer_index)
        print("Loaded segmap:", seg.shape, seg.dtype)
        validate_indices(feats, seg)

    prompts = [p.strip() for p in args.prompts.split(",") if p.strip()]
    print("Prompts (", len(prompts), "):", prompts)
    compute_text_similarities(feats, prompts, args.model, args.pretrained, args.topk)
