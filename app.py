import gradio as gr
import numpy as np
import cv2
import torch
from facenet_pytorch import MTCNN
from sklearn.cluster import KMeans
from collections import Counter
import tensorflow as tf
from PIL import Image

# ——— 1. Load TFLite interpreter ———
interpreter = tf.lite.Interpreter(model_path="palette_generator.tflite")
interpreter.allocate_tensors()
_input_idx  = interpreter.get_input_details()[0]["index"]
_output_idx = interpreter.get_output_details()[0]["index"]

# ——— 2. Face Detection Branch ———
class FaceDetectionBranch:
    def __init__(self, image_size=160, margin=20, min_face_size=20,
                 thresholds=[0.6, 0.7, 0.7], factor=0.709):
        # Force CPU to avoid MPS adaptive‐pool bug
        self.device = torch.device('cpu')
        self.mtcnn = MTCNN(
            image_size=image_size,
            margin=margin,
            min_face_size=min_face_size,
            thresholds=thresholds,
            factor=factor,
            device=self.device
        )

    def detect_faces(self, image):
        # Ensure RGB numpy array
        if isinstance(image, np.ndarray) and image.shape[-1] == 3:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if image.dtype==np.uint8 else image
        else:
            rgb = np.array(image) if isinstance(image, Image.Image) else image

        boxes, _ = self.mtcnn.detect(rgb)
        faces = []
        if boxes is not None:
            for x1, y1, x2, y2 in boxes:
                faces.append([int(x1), int(y1), int(x2-x1), int(y2-y1)])
        return faces

    def crop_faces(self, image, faces, padding=0.2):
        img = np.array(image) if isinstance(image, Image.Image) else image
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if img.shape[-1]==3 else img
        h, w = img_bgr.shape[:2]
        crops = []
        for x, y, fw, fh in faces:
            pad_w = int(fw*padding); pad_h = int(fh*padding)
            x1 = max(0, x - pad_w); y1 = max(0, y - pad_h)
            x2 = min(w, x + fw + pad_w); y2 = min(h, y + fh + pad_h)
            face = img_bgr[y1:y2, x1:x2]
            if face.size: crops.append(face)
        return crops

# ——— 3. Skin Tone Extractor ———
class SkinToneExtractor:
    def __init__(self, n_clusters=4):
        self.n_clusters = n_clusters
        self.lower_hsv = np.array([0,20,70],dtype=np.uint8)
        self.upper_hsv = np.array([25,255,255],dtype=np.uint8)

    def extract_skin_mask(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_hsv, self.upper_hsv)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
        mask = cv2.erode(mask, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=1)
        return cv2.GaussianBlur(mask, (3,3), 0)

    def find_dominant_color(self, image, mask):
        skin = cv2.bitwise_and(image, image, mask=mask)
        pixels = skin.reshape(-1,3)
        pixels = pixels[np.all(pixels!=[0,0,0],axis=1)]
        if len(pixels)==0:
            return np.array([211,169,150])
        km = KMeans(n_clusters=self.n_clusters, n_init=10).fit(pixels)
        label = Counter(km.labels_).most_common(1)[0][0]
        return km.cluster_centers_[label].astype(int)

    def extract_skin_tone(self, face_image):
        mask = self.extract_skin_mask(face_image)
        dom = self.find_dominant_color(face_image, mask)
        # BGR → RGB
        r, g, b = dom[::-1]
        hex_code = f"#{r:02x}{g:02x}{b:02x}"
        return hex_code, (r, g, b)

# ——— 4. Fitzpatrick Classification ———
def classify_skin_tone(rgb_color):
    rgb_norm = np.array([[rgb_color]],dtype=np.float32)/255.0
    bgr = rgb_norm[:,:,::-1]
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2Lab)[0][0]
    L, b = lab[0], lab[2]-128
    ita = np.arctan((L-50)/(b if b!=0 else 0.01))*180/np.pi
    # if ita>55: return "Type I "
    # if 48<=ita<=55: return "Type II"
    # if 41<=ita<48: return "Type III"
    # if 30<=ita<41: return "Type IV"
    # if 19<=ita<30: return "Type V "
    # return "Type VI "

# ——— 5. TFLite Palette Prediction ———
def hex_to_int8_rgb(h):
    h = h.lstrip('#')
    if len(h)==3: h = ''.join(c*2 for c in h)
    vals = [int(h[i:i+2],16)/255.0 for i in (0,2,4)]
    return (np.array(vals)*255 - 128).astype(np.int8)

def predict_palette(hex_code):
    inp = np.expand_dims(hex_to_int8_rgb(hex_code),0)
    interpreter.set_tensor(_input_idx, inp)
    interpreter.invoke()
    raw = interpreter.get_tensor(_output_idx)[0]
    out = []
    for i in range(0,15,3):
        norm = (raw[i:i+3].astype(np.float32)+128)/255.0
        rgb = (norm*255).round().astype(int)
        out.append(f"#{rgb[0]:02X}{rgb[1]:02X}{rgb[2]:02X}")
    return out

# ——— 6. Full Processing Pipeline ———
def process_face_image(image):
    fd = FaceDetectionBranch()
    img_np = np.array(image) if isinstance(image, Image.Image) else image
    faces = fd.detect_faces(img_np)
    if not faces: return None, None, None
    crops = fd.crop_faces(img_np, faces)
    if not crops: return None, None, None
    ste = SkinToneExtractor()
    hex_code, rgb_color = ste.extract_skin_tone(crops[0])
    fitz = classify_skin_tone(rgb_color)
    palette = predict_palette(hex_code)
    return hex_code, fitz, palette

def display_output(image):
    if image is None:
        return "<p>Please upload an image.</p>", ""
    hex_code, fitz, palette = process_face_image(image)
    if hex_code is None:
        return "<p>No face detected.</p>", ""
    tone_html = (
        "<h3 style='text-align:center;'>Detected Skin Tone</h3>"
        f"<h2 style='text-align:center; color:#333;'>{hex_code}</h2>"
        f"<p style='text-align:center;'>{fitz}</p>"
    )
    swatches = "<div style='display:flex; gap:12px; justify-content:center;'>"
    for c in palette:
        swatches += (
            f"<div style='width:60px; height:60px; background:{c}; "
            "border-radius:8px; box-shadow:0 1px 4px rgba(0,0,0,0.1);'></div>"
        )
    swatches += "</div>"
    pal_html = "<h3 style='text-align:center;'>Recommended Palette</h3>" + swatches
    return tone_html, pal_html

# ——— 7. Gradio UI ———
custom_css = """
#upload-btn {margin:auto; border:2px dashed #ccc; padding:16px; background:white; border-radius:10px;}
.gr-button {background:#FFBC80; color:#222; font-weight:bold; border:none;}
.gr-button:hover {background:#ffaa55;}
"""
with gr.Blocks(css=custom_css, title="SkinShade AI") as demo:
    gr.Markdown("# Skin Tone & Color Palette Recommender")
    inp = gr.Image(type="pil", elem_id="upload-btn", label="Upload Your Photo")
    tone_out, pal_out = gr.HTML(), gr.HTML()
    btn = gr.Button("Detect Tone")
    btn.click(display_output, inputs=inp, outputs=[tone_out, pal_out])
    gr.Markdown("We don’t store any images—everything runs locally.")
if __name__ == "__main__":
    demo.launch(share=True)