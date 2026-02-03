# SkinShade AI

**Real-time skin-tone classification and personalized color-palette generation**

SkinShade AI detects a face from any uploaded photo, isolates skin pixels under varied lighting conditions, extracts the dominant skin color, classifies the tone on the Fitzpatrick scale (Type I-VI), and generates a 5-color personalized palette -- all on CPU in real time with zero image storage.

---

## Table of Contents

- [Problem Statement](#problem-statement)
- [Solution](#solution)
- [Features](#features)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Pipeline Overview](#pipeline-overview)
- [Algorithms and Design Choices](#algorithms-and-design-choices)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Privacy](#privacy)
- [Limitations](#limitations)
- [License](#license)

---

## Problem Statement

Skin-tone-aware UIs -- cosmetics virtual try-ons, avatar creators, accessibility themes -- frequently fail under changing illumination, busy backgrounds, and diverse skin tones. Existing approaches couple brightness with color, causing unstable tone detection when lighting shifts.

## Solution

SkinShade AI decouples illumination from chrominance by operating in the HSV color space. It focuses strictly on the facial region using MTCNN face detection, computes tone and palette from skin-only pixels via KMeans clustering, and maps the result to a medically-recognized Fitzpatrick classification. A lightweight TFLite neural network then generates a 5-color complementary palette from the detected tone.

## Features

- **Face Detection** -- MTCNN-based detection with configurable padding for reliable, aligned face crops
- **Lighting-Invariant Skin Segmentation** -- HSV color space thresholding separates hue/saturation from brightness
- **Dominant Color Extraction** -- KMeans clustering (k=4) isolates the true skin color from masked pixels
- **Fitzpatrick Scale Classification** -- ITA (Individual Typology Angle) calculation maps skin tone to Type I-VI
- **5-Color Palette Generation** -- Pre-trained TFLite neural network recommends complementary colors
- **Privacy-First Design** -- All processing runs locally; no images are uploaded or stored
- **Real-Time CPU Inference** -- No GPU required; runs on any machine including Apple Silicon

---

## Architecture

```
                        +---------------------+
                        |   User uploads image |
                        +----------+----------+
                                   |
                        +----------v----------+
                        |   MTCNN Face         |
                        |   Detection          |
                        +----------+----------+
                                   |
                        +----------v----------+
                        |   Face Cropping      |
                        |   (20% padding)      |
                        +----------+----------+
                                   |
                        +----------v----------+
                        |   HSV Skin Mask      |
                        |   Segmentation       |
                        +----------+----------+
                                   |
                        +----------v----------+
                        |   KMeans Dominant    |
                        |   Color Extraction   |
                        +----------+----------+
                                   |
                    +--------------+--------------+
                    |                              |
          +---------v---------+        +-----------v-----------+
          |  ITA Fitzpatrick  |        |  TFLite Palette       |
          |  Classification   |        |  Generation           |
          +-------------------+        +-----------------------+
                    |                              |
                    +--------------+---------------+
                                   |
                        +----------v----------+
                        |   Gradio Web UI      |
                        |   (Tone + Palette)   |
                        +----------------------+
```

---

## Tech Stack

| Layer               | Technology                              |
|---------------------|-----------------------------------------|
| Web UI              | Gradio                                  |
| Face Detection      | MTCNN (facenet-pytorch)                 |
| Computer Vision     | OpenCV                                  |
| Skin Color Clustering | scikit-learn (KMeans)                 |
| Palette Model       | TensorFlow Lite                         |
| Deep Learning Core  | PyTorch                                 |
| Numerical Computing | NumPy                                   |
| Image Handling      | Pillow (PIL)                            |
| Language            | Python 3.x                              |

---

## Pipeline Overview

### 1. Face Detection
MTCNN (Multi-task Cascaded Convolutional Networks) from `facenet-pytorch` detects faces and returns bounding boxes. Faces are cropped with 20% padding to include surrounding skin area.

### 2. Skin Segmentation
The cropped face is converted to HSV color space. A mask is created using thresholds (H: 0-25, S: 20-255, V: 70-255) to isolate skin-toned pixels. Morphological operations (erosion, dilation) and Gaussian blur clean the mask.

### 3. Dominant Color Extraction
Masked skin pixels are fed into KMeans clustering (k=4, 10 initializations). The largest cluster center represents the dominant skin color, returned as both RGB and hex.

### 4. Fitzpatrick Classification
The dominant RGB color is converted to CIE L\*a\*b\* color space. The Individual Typology Angle (ITA) is computed:

```
ITA = arctan((L* - 50) / b*) x (180 / pi)
```

ITA values map to Fitzpatrick Types I through VI.

### 5. Palette Generation
The detected hex color is fed into a quantized TFLite neural network (~21.6 KB) that outputs 5 complementary colors (base, neutral, complement, and two analogous tones) as hex codes.

### 6. Display
Results are rendered in a Gradio web interface with color swatches and tone labels.

---

## Algorithms and Design Choices

| Component             | Choice          | Rationale                                                                 |
|-----------------------|-----------------|---------------------------------------------------------------------------|
| Face detection        | MTCNN           | Reliable multi-stage detector with built-in alignment; no GPU needed      |
| Skin segmentation     | HSV thresholding | Separates chrominance from luminance; brightness shifts don't affect hue |
| Dominant color        | KMeans (k=4)    | Fast, interpretable clustering; captures skin tone despite minor noise   |
| Tone classification   | ITA formula     | Clinically-validated metric for Fitzpatrick scale mapping                |
| Palette model         | TFLite NN       | Quantized to ~21 KB for portable, sub-millisecond CPU inference          |
| UI framework          | Gradio          | Minimal code for interactive ML demos with image upload support          |
| Device                | CPU-only        | Maximizes portability; avoids MPS/CUDA compatibility issues              |

---

## Project Structure

```
SkinShade/
├── app.py                       # Production Gradio application (full pipeline)
├── ml_project_final.ipynb       # Development notebook (experiments & documentation)
├── palette_generator.tflite     # Pre-trained quantized palette model (~21.6 KB)
└── README.md                    # Project documentation
```

---

## Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/SkinShade.git
cd SkinShade

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install gradio numpy opencv-python torch facenet-pytorch scikit-learn tensorflow pillow
```

---

## Usage

```bash
python app.py
```

This launches a Gradio web interface (opens in your browser). Upload a photo containing a face, click **Detect Tone**, and the app will display:

1. The detected dominant skin tone (hex code)
2. The Fitzpatrick classification (Type I-VI)
3. A 5-color recommended palette with visual swatches

To share the app publicly (via a temporary Gradio link), the app runs with `share=True` by default.

---

## Privacy

SkinShade AI is designed with privacy as a core principle:

- **No image storage** -- uploaded images are processed in memory and discarded
- **No network calls** -- all ML inference runs locally on your machine
- **No telemetry** -- the application does not collect or transmit any data

---

## Limitations

- Processes only the first detected face in multi-face images
- HSV thresholds are tuned for common skin tones; extreme lighting (e.g., colored stage lights) may reduce accuracy
- Fitzpatrick classification uses ITA approximation, not dermatological ground truth
- No batch processing support in the current UI
