#!/usr/bin/env python3
"""Average several checkpoints' model weights (poor-man's EMA)."""
import argparse
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    ckpts = [torch.load(p, map_location="cpu") for p in args.inputs]
    state_dicts = [c["model"] for c in ckpts]

    avg = {}
    for k in state_dicts[0]:
        stacked = torch.stack([sd[k].float() for sd in state_dicts])
        avg[k] = stacked.mean(0).to(state_dicts[0][k].dtype)

    out = {"model": avg, "epoch": ckpts[-1]["epoch"], "optimizer": ckpts[-1].get("optimizer", {})}
    torch.save(out, args.output)
    print(f"Averaged {len(args.inputs)} checkpoints → {args.output}")


if __name__ == "__main__":
    main()
