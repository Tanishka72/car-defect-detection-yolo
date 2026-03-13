from __future__ import annotations

import csv
import json
import logging
from pathlib import Path

from core.image_detector import ImageResult

logger = logging.getLogger(__name__)


def write_csv_report(results: list[ImageResult], output_path: str | Path) -> Path:
    """Write a CSV summary report of all detection results.

    Columns: image_name, defect_type, confidence, bbox
    Images with no detections get a single row with "No defect" entries.
    Skipped (corrupt) images are noted with their error.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_name", "defect_type", "confidence", "bbox"])

        for result in results:
            name = result.image_path.name

            if result.error:
                writer.writerow([name, f"SKIPPED: {result.error}", "", ""])
            elif not result.has_detections:
                writer.writerow([name, "No defect", "", ""])
            else:
                for det in result.detections:
                    bbox_str = f"[{', '.join(str(int(v)) for v in det.bbox)}]"
                    writer.writerow([name, det.class_name, f"{det.confidence:.4f}", bbox_str])

    logger.info("CSV report saved: %s", output_path)
    return output_path


def write_json_report(results: list[ImageResult], output_path: str | Path) -> Path:
    """Write a JSON summary report of all detection results."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    report = []
    for result in results:
        entry = {"image_name": result.image_path.name}

        if result.error:
            entry["status"] = "skipped"
            entry["error"] = result.error
        else:
            entry["status"] = "processed"
            entry["defect_count"] = len(result.detections)
            entry["detections"] = [
                {
                    "defect_type": d.class_name,
                    "confidence": d.confidence,
                    "bbox": [int(v) for v in d.bbox],
                }
                for d in result.detections
            ]

        report.append(entry)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    logger.info("JSON report saved: %s", output_path)
    return output_path


def print_summary(results: list[ImageResult]) -> dict:
    """Print and return a summary of the batch processing run."""
    total = len(results)
    skipped = sum(1 for r in results if r.error)
    with_defects = sum(1 for r in results if r.has_detections)
    clean = total - skipped - with_defects
    total_detections = sum(len(r.detections) for r in results)

    # Count by defect type
    defect_counts: dict[str, int] = {}
    for r in results:
        for d in r.detections:
            defect_counts[d.class_name] = defect_counts.get(d.class_name, 0) + 1

    summary = {
        "total_images": total,
        "processed": total - skipped,
        "skipped": skipped,
        "with_defects": with_defects,
        "clean": clean,
        "total_detections": total_detections,
        "defect_breakdown": defect_counts,
    }

    print("\n" + "=" * 55)
    print("  BATCH PROCESSING SUMMARY")
    print("=" * 55)
    print(f"  Total images      : {total}")
    print(f"  Processed          : {total - skipped}")
    print(f"  Skipped (corrupt)  : {skipped}")
    print(f"  With defects       : {with_defects}")
    print(f"  Clean (no defects) : {clean}")
    print(f"  Total detections   : {total_detections}")

    if defect_counts:
        print("\n  Defect Breakdown:")
        for defect, count in sorted(defect_counts.items(), key=lambda x: -x[1]):
            print(f"    {defect:<20s} : {count}")

    print("=" * 55 + "\n")

    return summary
