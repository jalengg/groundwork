#!/usr/bin/env python3
"""Repackage data/flux_cnet/{cond,target,meta}/ into HF imagefolder layout.

The diffusers SDXL ControlNet training script expects either a HuggingFace
dataset on the Hub or a local directory loadable via the imagefolder builder.
The imagefolder format requires a single image folder + metadata.jsonl per
split, with conditioning image referenced as a sibling path.

We symlink rather than copy to avoid duplicating ~50 GB of PNGs.

Usage:
    python tools/build_hf_manifest.py \\
        --src /u/jalenj4/groundwork/data/flux_cnet \\
        --dst /u/jalenj4/groundwork/data/flux_cnet_hf
"""
import argparse
import json
import pathlib
import shutil


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True, help="Source dir with {cond,target,meta} subdirs.")
    p.add_argument("--dst", required=True, help="Output dir for HF imagefolder layout.")
    p.add_argument("--val-dir", default=None,
                   help="OUTSIDE-dataset dir to cherry-pick val cond tiles into. "
                   "Leave unset to skip; val tiles inside the dataset dir get "
                   "auto-detected as a 'validation' split and break load_dataset.")
    p.add_argument("--n-val", type=int, default=2)
    args = p.parse_args()

    # Resolve to absolute so symlinks remain valid no matter where they're read from
    src = pathlib.Path(args.src).resolve()
    dst = pathlib.Path(args.dst).resolve()
    dst.mkdir(parents=True, exist_ok=True)

    # Symlink cond/ and target/ — saves ~50 GB vs copy
    for sub in ("cond", "target"):
        link = dst / sub
        if link.is_symlink() or link.exists():
            print(f"  skip existing: {link}")
            continue
        link.symlink_to(src / sub, target_is_directory=True)
        print(f"  symlinked {link} -> {src / sub}")

    # Build metadata.jsonl
    target_files = sorted((src / "target").glob("*.png"))
    print(f"Found {len(target_files)} target tiles")
    with open(dst / "metadata.jsonl", "w") as f:
        for t in target_files:
            cap_path = src / "meta" / f"{t.stem}.txt"
            cap = cap_path.read_text().strip() if cap_path.exists() else ""
            f.write(json.dumps({
                "file_name": f"target/{t.name}",
                "text": cap,
                "conditioning_image": f"cond/{t.name}",
            }) + "\n")
    print(f"Wrote {dst / 'metadata.jsonl'}")

    # Optional: cherry-pick validation cond tiles into a separate dir
    # OUTSIDE the dataset folder (val/ inside dst would be auto-detected as
    # a 'validation' split by HF's imagefolder builder).
    if args.val_dir:
        val_dir = pathlib.Path(args.val_dir).resolve()
        val_dir.mkdir(parents=True, exist_ok=True)
        for i, t in enumerate(target_files[:args.n_val]):
            val_cond_dst = val_dir / f"val_cond_{i}.png"
            if not val_cond_dst.exists():
                shutil.copy(src / "cond" / t.name, val_cond_dst)
                print(f"  copied {val_cond_dst}")


if __name__ == "__main__":
    main()
