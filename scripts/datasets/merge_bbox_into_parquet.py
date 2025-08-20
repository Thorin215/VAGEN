#!/usr/bin/env python3
"""
Merge bounding boxes from a CSV into a parquet dataset by matching image paths.

CSV required columns: image_path, gt_path, x1, y1, x2, y2, width, height
Parquet expectation: rows contain extra_info.env_config.image_path (created by create_dataset.py)

Effect: add/update extra_info.gt_bbox = [x1, y1, x2, y2] (ints). Optionally add extra_info.gt_path.

Usage example:
  python scripts/datasets/merge_bbox_into_parquet.py \
    --parquet-in ./train.parquet \
    --csv /opt/liblibai-models/user-workspace2/users/wc/m4detection/output_bboxes.csv \
    --parquet-out ./train.with_bbox.parquet
"""

import argparse
import csv
import os
from typing import Dict, List, Tuple, Optional

from datasets import load_dataset, Dataset


def _read_csv(csv_path: str,
              key_col: str = "image_path",
              cols: Tuple[str, str, str, str, str, str, str] = ("gt_path", "x1", "y1", "x2", "y2", "width", "height")) -> Tuple[Dict[str, dict], Dict[str, List[dict]]]:
    """Read CSV and build exact-path and basename maps.

    Returns:
      exact_map: image_path -> record
      base_map: basename(image_path) -> list of records (to detect ambiguities)
    """
    required = {key_col, *cols}
    exact_map: Dict[str, dict] = {}
    base_map: Dict[str, List[dict]] = {}
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"CSV missing required columns: {sorted(missing)}")
        for row in reader:
            path = row[key_col]
            # Normalize numeric fields and ensure ints
            try:
                x1, y1, x2, y2 = int(float(row["x1"])), int(float(row["y1"])), int(float(row["x2"])), int(float(row["y2"]))
            except Exception as e:
                raise ValueError(f"Invalid bbox in row: {row}") from e
            rec = {
                "image_path": path,
                "gt_path": row.get("gt_path", ""),
                "bbox": [x1, y1, x2, y2],
                "width": int(float(row.get("width", 0) or 0)),
                "height": int(float(row.get("height", 0) or 0)),
            }
            exact_map[path] = rec
            base = os.path.basename(path)
            base_map.setdefault(base, []).append(rec)
    return exact_map, base_map


def _match_record(image_path: str, exact_map: Dict[str, dict], base_map: Dict[str, List[dict]], allow_basename: bool = True) -> Optional[dict]:
    if image_path in exact_map:
        return exact_map[image_path]
    if not allow_basename:
        return None
    base = os.path.basename(image_path)
    cands = base_map.get(base, [])
    if len(cands) == 1:
        return cands[0]
    # ambiguous or missing
    return None


def _inject_bbox(example: dict, match: dict, store_gt_path: bool) -> dict:
    extra = example.get("extra_info") or {}
    # write gt_bbox
    extra["gt_bbox"] = list(map(int, match["bbox"]))
    if store_gt_path:
        extra["gt_path"] = match.get("gt_path", "")
    example["extra_info"] = extra
    return example


def main():
    ap = argparse.ArgumentParser(description="Merge bbox from CSV into parquet dataset")
    ap.add_argument("--parquet-in", required=True, help="Input parquet path")
    ap.add_argument("--csv", required=True, help="CSV with bbox columns")
    ap.add_argument("--parquet-out", required=True, help="Output parquet path")
    ap.add_argument("--csv-key", default="image_path", help="CSV column used for matching (default: image_path)")
    ap.add_argument("--allow-basename", action="store_true", help="Allow basename fallback matching when exact path not found")
    ap.add_argument("--store-gt-path", action="store_true", help="Also store extra_info.gt_path from CSV")
    args = ap.parse_args()

    exact_map, base_map = _read_csv(args.csv, key_col=args.csv_key)

    # Load parquet as a dataset
    ds = load_dataset('parquet', data_files={"data": args.parquet_in}, split='data')

    # Count stats
    matched = 0
    missing = 0

    def proc(example):
        nonlocal matched, missing
        # Expected location for image path: extra_info.env_config.image_path
        image_path = None
        try:
            extra = example.get("extra_info") or {}
            env_cfg = extra.get("env_config") or {}
            image_path = env_cfg.get("image_path")
        except Exception:
            image_path = None

        if not image_path:
            missing += 1
            return example

        match = _match_record(image_path, exact_map, base_map, allow_basename=args.allow_basename)
        if match is None:
            missing += 1
            return example

        # Validate bbox with width/height if provided
        x1, y1, x2, y2 = match["bbox"]
        W, H = match.get("width") or 0, match.get("height") or 0
        if W and H:
            # Clamp to image size
            x1 = max(0, min(x1, W - 1))
            x2 = max(0, min(x2, W - 1))
            y1 = max(0, min(y1, H - 1))
            y2 = max(0, min(y2, H - 1))
            # Ensure ordering
            if x2 < x1:
                x1, x2 = x2, x1
            if y2 < y1:
                y1, y2 = y2, y1
            match = {**match, "bbox": [int(x1), int(y1), int(x2), int(y2)]}

        matched += 1
        return _inject_bbox(example, match, store_gt_path=args.store_gt_path)

    ds2 = ds.map(proc)
    # Save to parquet
    os.makedirs(os.path.dirname(args.parquet_out) or ".", exist_ok=True)
    ds2.to_parquet(args.parquet_out)

    total = len(ds)
    print(f"Done. total={total}, matched={matched}, missing_or_unmatched={missing}")


if __name__ == "__main__":
    main()
