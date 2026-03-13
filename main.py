"""
Vehicle Damage Detector — CLI Batch Processor

Usage:
    python main.py <zip_file> [--model models/best.pt] [--conf 0.25] [--output output]

Example:
    python main.py vehicle_images.zip
    python main.py vehicle_images.zip --conf 0.4 --output results
"""

import argparse
import logging
import shutil
import tempfile
import time
from pathlib import Path

from core.zip_handler import extract_zip
from core.image_detector import ImageDetector
from core.result_writer import write_csv_report, write_json_report, print_summary

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Batch Vehicle Damage Detection")
    parser.add_argument("zip_file", help="Path to ZIP file containing images")
    parser.add_argument("--model", default="models/best.pt", help="Path to trained YOLOv8 model")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold (0-1)")
    parser.add_argument("--output", default="output", help="Output directory for results")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_images_dir = output_dir / "images"
    output_images_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Extract ZIP
    print(f"\n[1/5] Extracting: {args.zip_file}")
    temp_dir = tempfile.mkdtemp(prefix="img_analyzer_")
    try:
        image_files = extract_zip(args.zip_file, temp_dir)
        print(f"   Found {len(image_files)} image(s)\n")

        if not image_files:
            print("ERROR: No supported images found in the ZIP file.")
            return

        # Step 2: Load model
        print(f"[2/5] Loading model: {args.model}")
        detector = ImageDetector(model_path=args.model, conf=args.conf)

        # Step 3: Process images
        print(f"[3/5] Processing {len(image_files)} image(s)...\n")
        results = []
        start = time.time()

        for i, img_path in enumerate(image_files, 1):
            print(f"  [{i}/{len(image_files)}] {img_path.name}", end=" ")
            result = detector.process_and_save(img_path, output_images_dir)
            results.append(result)

            if result.error:
                print(f"  SKIPPED - {result.error}")
            elif result.has_detections:
                defects = ", ".join(d.class_name for d in result.detections)
                print(f"  FOUND {len(result.detections)} defect(s): {defects}")
            else:
                print("  CLEAN")

        elapsed = time.time() - start

        # Step 4: Generate reports
        csv_path = write_csv_report(results, output_dir / "report.csv")
        json_path = write_json_report(results, output_dir / "report.json")

        # Step 5: Print summary
        summary = print_summary(results)
        print(f"  Total time:       {elapsed:.1f}s ({elapsed/len(image_files):.2f}s per image)")
        print(f"  Annotated images: {output_images_dir}")
        print(f"  CSV report:       {csv_path}")
        print(f"  JSON report:      {json_path}\n")

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
