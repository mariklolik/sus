#!/usr/bin/env python3
"""Merge per-method eval result files into one combined file."""

import json
import glob
import os

merged = {}
for f in sorted(glob.glob("outputs/eval_result_*.json")):
    with open(f) as fh:
        merged.update(json.load(fh))

out = "outputs/eval_results_final.json"
with open(out, "w") as fh:
    json.dump(merged, fh, indent=2)

print(f"Merged {len(merged)} methods into {out}")
print(f"{'Method':<20} {'Pass@1':>8} {'Pass@5':>8} {'Pass@8':>8}")
print("-" * 50)
for m, r in sorted(merged.items()):
    if isinstance(r, dict) and "pass@1" in r:
        print(f"{m:<20} {r['pass@1']:>8.4f} {r['pass@5']:>8.4f} {r['pass@8']:>8.4f}")
