# Vehicle Damage Detector

Batch vehicle damage detection system using a custom-trained YOLOv8 model. Detects dents, scratches, cracks, glass shatter, lamp broken, and tire flat in vehicle images.

## Setup

```bash
pip install -r requirements.txt
```

## Usage

### Web Interface

```bash
streamlit run app.py
```

Opens at **http://localhost:8501**. Supports single image upload or batch ZIP processing.

### Command Line (Batch)

```bash
python main.py vehicle_images.zip
python main.py vehicle_images.zip --conf 0.4 --output results
```

## Detectable Damage Types

- Dent
- Scratch
- Crack
- Glass shatter
- Lamp broken
- Tire flat

## Project Structure

```
models/
  best.pt              -- Trained YOLOv8 model
core/
  zip_handler.py       -- ZIP extraction and image filtering
  image_detector.py    -- Model loading, inference, bounding box drawing
  result_writer.py     -- CSV/JSON report generation
output/                -- Annotated images and reports
app.py                 -- Streamlit web UI
main.py                -- CLI batch processor
requirements.txt       -- Python dependencies
```
