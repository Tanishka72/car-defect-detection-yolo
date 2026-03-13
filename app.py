import io
import shutil
import tempfile
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from core.image_detector import ImageDetector
from core.zip_handler import extract_zip
from core.result_writer import write_csv_report, write_json_report, print_summary

st.set_page_config(page_title="Vehicle Damage Detector", layout="wide")

st.title("Vehicle Damage Detector")
st.write(
    "Detect **dents, scratches, cracks, glass shatter, lamp broken, tire flat** "
    "in vehicle images using your trained YOLOv8 model."
)


@st.cache_resource
def load_detector(model_path: str, conf: float):
    return ImageDetector(model_path=model_path, conf=conf)


# --- Sidebar ---
with st.sidebar:
    st.header("Settings")

    model_path = st.text_input("Model path", value="models/best.pt")

    confidence = st.slider(
        "Confidence threshold",
        min_value=0.1,
        max_value=1.0,
        value=0.25,
        step=0.05,
    )

    st.markdown("---")
    st.markdown(
        "**Detectable damage types:**\n"
        "- Scratch\n"
        "- Dent\n"
        "- Crack\n"
        "- Glass Shatter\n"
        "- Lamp Broken\n"
        "- Tire Flat\n"
        "- Normal (no damage)"
    )


# Load detector
try:
    detector = load_detector(model_path, confidence)
except FileNotFoundError:
    st.error(f"Model not found at `{model_path}`. Place your `best.pt` in the `models/` folder.")
    st.stop()


tab1, tab2 = st.tabs(["Single Image", "Batch (ZIP Upload)"])

# ============================
# TAB 1: Single Image Analysis
# ============================
with tab1:
    uploaded_file = st.file_uploader(
        "Upload a vehicle image", type=["jpg", "jpeg", "png", "bmp", "webp"], key="single"
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        img_np = np.array(image)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        # Save to temp file for detector
        tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        tmp_path = tmp.name
        tmp.close()  # Close file handle before using it
        cv2.imwrite(tmp_path, img_bgr)

        with st.spinner("Detecting vehicle damage..."):
            result = detector.detect(Path(tmp_path))

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original")
            st.image(image, use_container_width=True)

        with col2:
            st.subheader("Detections")
            if result.has_detections:
                annotated_bgr = detector.draw_boxes(tmp_path, result.detections)
                annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
                st.image(annotated_rgb, use_container_width=True)
            else:
                st.image(image, use_container_width=True)
                st.info("No damage detected in this image.")

        try:
            Path(tmp_path).unlink(missing_ok=True)
        except:
            pass  # Ignore cleanup errors on Windows

        # Detection details
        st.markdown("---")
        if result.has_detections:
            st.subheader(f"{len(result.detections)} Damage(s) Detected")

            # Count by type
            counts = {}
            for d in result.detections:
                counts[d.class_name] = counts.get(d.class_name, 0) + 1

            cols = st.columns(min(len(counts), 6))
            for i, (name, count) in enumerate(sorted(counts.items(), key=lambda x: -x[1])):
                with cols[i % len(cols)]:
                    st.metric(label=name.capitalize(), value=count)

            st.markdown("#### Details")
            for i, d in enumerate(result.detections, 1):
                st.write(f"**{i}.** {d.class_name.capitalize()} — confidence: {d.confidence:.1%}")
        else:
            st.success("No vehicle damage detected.")


# ============================
# TAB 2: Batch ZIP Processing
# ============================
with tab2:
    st.write("Upload a **ZIP file** containing vehicle images for batch processing.")

    uploaded_zip = st.file_uploader(
        "Upload ZIP file", type=["zip"], key="batch"
    )

    if uploaded_zip is not None:
        # Save uploaded ZIP to a temp file
        tmp_zip = tempfile.NamedTemporaryFile(suffix=".zip", delete=False)
        tmp_zip.write(uploaded_zip.read())
        tmp_zip.close()

        tmp_extract = tempfile.mkdtemp(prefix="batch_extract_")
        tmp_output = tempfile.mkdtemp(prefix="batch_output_")

        try:
            with st.spinner("Extracting ZIP file..."):
                image_files = extract_zip(tmp_zip.name, tmp_extract)

            st.success(f"Found **{len(image_files)}** image(s) in the ZIP file.")

            if image_files:
                progress = st.progress(0, text="Processing images...")
                results = []

                for i, img_path in enumerate(image_files):
                    progress.progress(
                        (i + 1) / len(image_files),
                        text=f"Processing {i+1}/{len(image_files)}: {img_path.name}",
                    )
                    result = detector.process_and_save(img_path, tmp_output)
                    results.append(result)

                progress.progress(1.0, text="Done!")

                # --- Summary ---
                st.markdown("---")
                total = len(results)
                skipped = sum(1 for r in results if r.error)
                with_defects = sum(1 for r in results if r.has_detections)
                clean = total - skipped - with_defects
                total_dets = sum(len(r.detections) for r in results)

                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric("Total Images", total)
                c2.metric("With Defects", with_defects)
                c3.metric("Clean", clean)
                c4.metric("Skipped", skipped)
                c5.metric("Total Detections", total_dets)

                # Defect breakdown
                defect_counts = {}
                for r in results:
                    for d in r.detections:
                        defect_counts[d.class_name] = defect_counts.get(d.class_name, 0) + 1

                if defect_counts:
                    st.markdown("#### Defect Breakdown")
                    cols = st.columns(min(len(defect_counts), 6))
                    for i, (name, count) in enumerate(
                        sorted(defect_counts.items(), key=lambda x: -x[1])
                    ):
                        with cols[i % len(cols)]:
                            st.metric(name.capitalize(), count)

                # --- CSV download ---
                csv_path = Path(tmp_output) / "report.csv"
                write_csv_report(results, csv_path)
                with open(csv_path, "rb") as f:
                    st.download_button(
                        "Download CSV Report",
                        data=f.read(),
                        file_name="damage_report.csv",
                        mime="text/csv",
                    )

                # --- JSON download ---
                json_path = Path(tmp_output) / "report.json"
                write_json_report(results, json_path)
                with open(json_path, "rb") as f:
                    st.download_button(
                        "Download JSON Report",
                        data=f.read(),
                        file_name="damage_report.json",
                        mime="application/json",
                    )

                # --- Per-image results ---
                st.markdown("---")
                st.subheader("Per-Image Results")

                for idx, result in enumerate(results):
                    name = result.image_path.name
                    if result.error:
                        st.warning(f"{name}: {result.error}")
                        continue

                    defect_str = (
                        ", ".join(f"{d.class_name} ({d.confidence:.0%})" for d in result.detections)
                        if result.has_detections
                        else "No damage"
                    )
                    status = "[DAMAGED]" if result.has_detections else "[CLEAN]"

                    with st.expander(f"{status} {name} — {defect_str}", expanded=False):
                        c1, c2 = st.columns(2)

                        with c1:
                            orig = Image.open(result.image_path).convert("RGB")
                            st.image(orig, caption="Original", use_container_width=True)

                        with c2:
                            output_img_path = Path(tmp_output) / name
                            if output_img_path.exists():
                                ann = Image.open(output_img_path).convert("RGB")
                                st.image(ann, caption="Annotated", use_container_width=True)

                        if result.has_detections:
                            for i, d in enumerate(result.detections, 1):
                                st.write(
                                    f"**{i}.** {d.class_name.capitalize()} "
                                    f"— confidence: {d.confidence:.1%}"
                                )

        finally:
            Path(tmp_zip.name).unlink(missing_ok=True)
            shutil.rmtree(tmp_extract, ignore_errors=True)
            # Note: tmp_output cleaned up when Streamlit reruns or session ends
    else:
        st.info("Upload a ZIP file containing vehicle images to start batch processing.")
