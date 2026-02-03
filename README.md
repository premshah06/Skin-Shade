SkinShade AI

Real-time skin-tone classification & personalized color-palette generation

SkinShade AI detects a face, isolates skin pixels robustly under varied lighting, extracts the dominant skin color, classifies tone into Light / Medium / Dark, and returns a 5-color personalized palette ready for UI use — all on CPU in real time.

Overview

Problem. Skintone-aware UIs (cosmetics try-ons, avatar creators, accessibility themes) often fail under changing illumination and busy backgrounds.

Solution. Separate illumination from color, focus strictly on the facial region, and compute tone and palette from skin-only pixels.

Outcome. Consistent tone labels and palettes across common lighting conditions, surfaced instantly via a lightweight interactive UI.

What the system does

Finds faces with MTCNN (facenet-pytorch) for reliable, aligned crops.

Segments skin in HSV color space (S/V-centric thresholds) so brightness changes do not derail color estimates.

Extracts the dominant skin color using KMeans on masked pixels only.

Maps the dominant color to a tone label (Light / Medium / Dark) with simple, tunable thresholds (calibrated during development with samples from CelebA).

Generates a 5-color palette (base, neutral, complement, two analogs) as hex swatches for immediate UI use.

Runs live in a minimal UI (Gradio), with a small tone component exported to TensorFlow Lite for portable CPU inference.

Algorithms and design choices
Face detection — MTCNN (facenet-pytorch)

Skin segmentation — HSV (“SV-based”)

Dominant color — KMeans

Tone mapping — Light / Medium / Dark

Palette generation — five colors

Deployment and UI — TFLite + Gradio

A small classifier component is exported to TensorFlow Lite; an interactive Gradio app displays the tone label and color swatches.
