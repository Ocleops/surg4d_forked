#!/usr/bin/env python3
import argparse
import os
import re
from pathlib import Path
from typing import List, Tuple

import numpy as np
from sklearn.decomposition import IncrementalPCA


def list_feature_pairs(src: Path) -> List[Tuple[str, Path, Path]]:
    """Return sorted list of (stem, f_path, s_path) for pairs present in src.
    stem is the 6-digit basename without suffix.
    """
    f_files = sorted([p for p in src.glob("*_f.npy")])
    pairs = []
    for f in f_files:
        stem = f.name[:-6]  # remove '_f.npy'
        s = src / f"{stem}_s.npy"
        if not s.exists():
            # skip unmatched
            continue
        pairs.append((stem, f, s))
    # sort by numeric stem if possible
    def stem_key(item):
        stem = item[0]
        m = re.match(r"^(\d+)$", stem)
        return int(m.group(1)) if m else stem
    pairs.sort(key=stem_key)
    return pairs


def fit_incremental_pca(f_paths: List[Path], n_components: int, batch_rows: int = 20000) -> IncrementalPCA:
    ipca = IncrementalPCA(n_components=n_components)
    # first pass: partial fit
    for fpath in f_paths:
        feats = np.load(fpath)
        # ensure 2D
        feats = feats.astype(np.float32, copy=False)
        if feats.ndim != 2:
            raise ValueError(f"Expected 2D features in {fpath}, got shape {feats.shape}")
        # chunk rows to limit memory
        n = feats.shape[0]
        for start in range(0, n, batch_rows):
            ipca.partial_fit(feats[start:start + batch_rows])
    return ipca


def transform_and_write(pairs: List[Tuple[str, Path, Path]], out_dir: Path, ipca: IncrementalPCA) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for stem, f_path, s_path in pairs:
        feats = np.load(f_path).astype(np.float32, copy=False)
        reduced = ipca.transform(feats).astype(np.float32)
        np.save(out_dir / f"{stem}_f.npy", reduced)
        # copy s as-is
        s = np.load(s_path)
        np.save(out_dir / f"{stem}_s.npy", s)


def main():
    ap = argparse.ArgumentParser(description="Extract PCA latent features (default 3 dims) from language features")
    ap.add_argument("--src", type=str, required=True, help="Source language feature dir containing *_f.npy and *_s.npy")
    ap.add_argument("--dst", type=str, required=True, help="Destination dir to write 3D features")
    ap.add_argument("--n_components", type=int, default=3, help="Number of PCA components")
    ap.add_argument("--batch_rows", type=int, default=20000, help="Rows per partial_fit batch (IncrementalPCA)")
    args = ap.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    pairs = list_feature_pairs(src)
    if not pairs:
        raise SystemExit(f"No feature pairs found in {src}")

    print(f"Found {len(pairs)} frame pairs in {src}")
    f_paths = [f for _, f, _ in pairs]
    print(f"Fitting IncrementalPCA with n_components={args.n_components} ...")
    ipca = fit_incremental_pca(f_paths, n_components=args.n_components, batch_rows=args.batch_rows)
    print("Transforming and writing reduced features ...")
    transform_and_write(pairs, dst, ipca)
    print(f"Done. Wrote reduced features to {dst}")


if __name__ == "__main__":
    main() 