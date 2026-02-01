import uvicorn
import tensorflow as tf
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Response
import io
import cv2

app = FastAPI(title="Skin Analysis Tool")

MODEL_PATH = "model/skin_texture_saved_model"
FACE_CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
IMG_SIZE = 224

resources = {
    "model": None,
    "inference_func": None,
    "face_cascade": None
}


@app.on_event("startup")
def startup_event():
    try:
        resources["model"] = tf.saved_model.load(MODEL_PATH)
        resources["inference_func"] = resources["model"].signatures["serving_default"]
        resources["face_cascade"] = cv2.CascadeClassifier(FACE_CASCADE_PATH)
        print("System resources loaded successfully.")
    except Exception as e:
        print(f"Error loading resources: {e}")


def calculate_inflammation_index(image_roi):
    hsv = cv2.cvtColor(image_roi, cv2.COLOR_BGR2HSV)

    lower_red1, upper_red1 = np.array([0, 60, 50]), np.array([10, 255, 255])
    lower_red2, upper_red2 = np.array([170, 60, 50]), np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    combined_mask = mask1 + mask2

    total_pixels = image_roi.shape[0] * image_roi.shape[1]
    affected_pixels = cv2.countNonZero(combined_mask)

    if total_pixels == 0:
        return 0.0

    return (affected_pixels / total_pixels) * 100.0


def calculate_pigmentation_index(image_roi):
    gray = cv2.cvtColor(image_roi, cv2.COLOR_BGR2GRAY)

    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )

    total_pixels = image_roi.shape[0] * image_roi.shape[1]
    spot_pixels = cv2.countNonZero(thresh)

    if total_pixels == 0:
        return 0.0

    return (spot_pixels / total_pixels) * 100.0


def perform_hybrid_analysis(image_roi):
    ai_score = 50.0
    try:
        img_resized = cv2.resize(image_roi, (IMG_SIZE, IMG_SIZE))
        img_normalized = img_resized.astype(np.float32) / 255.0
        img_batch = np.expand_dims(img_normalized, axis=0)

        prediction = resources["inference_func"](tf.constant(img_batch))
        output_key = list(prediction.keys())[0]
        raw_output = float(prediction[output_key].numpy()[0][0])

        ai_score = (1.0 - raw_output) * 100.0
    except Exception as e:
        print(f"Inference error: {e}")

    inflammation_pct = calculate_inflammation_index(image_roi)
    pigmentation_pct = calculate_pigmentation_index(image_roi)

    final_score = ai_score

    if inflammation_pct > 4.0:
        penalty = inflammation_pct * 8.0
        final_score = min(final_score, 100.0 - penalty)

        if inflammation_pct > 8.0:
            final_score = min(final_score, 45.0)

    if pigmentation_pct > 10.0:
        final_score -= (pigmentation_pct * 0.5)

    return max(10.0, min(100.0, final_score))


def draw_diagnostic_overlay(image, x, y, w, h, score):
    if score >= 80:
        color = (0, 255, 0)
    elif score >= 60:
        color = (0, 255, 255)
    else:
        color = (0, 0, 255)

    cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)

    label_text = f"%{int(score)}"
    font = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 0.9
    font_thickness = 1

    (text_w, text_h), baseline = cv2.getTextSize(label_text, font, font_scale, font_thickness)

    text_x = x + (w - text_w) // 2
    text_y = y - 10

    if text_y < 20:
        text_y = y + h + text_h + 20

    padding = 5
    cv2.rectangle(image,
                  (text_x - padding, text_y - text_h - padding),
                  (text_x + text_w + padding, text_y + padding),
                  (0, 0, 0), -1)

    cv2.putText(image, label_text, (text_x, text_y), font, font_scale, color, font_thickness, cv2.LINE_AA)


@app.post("/predict_image")
async def predict_image(file: UploadFile = File(...)):
    if resources["inference_func"] is None:
        return Response(content="Service not initialized", status_code=503)

    file_bytes = await file.read()
    image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    image_np = np.array(image)

    height, width = image_np.shape[:2]
    max_dimension = 1200
    if width > max_dimension:
        scale_ratio = max_dimension / width
        new_height = int(height * scale_ratio)
        image_np = cv2.resize(image_np, (max_dimension, new_height))
        height, width = image_np.shape[:2]

    annotated_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    gray_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

    faces = resources["face_cascade"].detectMultiScale(
        gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80)
    )

    regions_of_interest = []

    if len(faces) > 0:
        faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
        (fx, fy, fw, fh) = faces[0]

        cheek_width = int(fw * 0.22)
        cheek_height = int(fh * 0.22)
        cheek_y_pos = int(fy + fh * 0.55)

        regions_of_interest = [
            (int(fx + fw * 0.15), cheek_y_pos, cheek_width, cheek_height),
            (int(fx + fw * 0.63), cheek_y_pos, cheek_width, cheek_height)
        ]
    else:
        box_size = int(min(width, height) * 0.3)
        center_y = (height // 2) - (box_size // 2)

        regions_of_interest = [
            (int(width * 0.10), center_y, box_size, box_size),
            (int(width * 0.60), center_y, box_size, box_size)
        ]

        cv2.putText(annotated_image, "MACRO MODE", (20, 50),
                    cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 255), 1, cv2.LINE_AA)

    for (rx, ry, rw, rh) in regions_of_interest:
        rx = max(0, min(rx, width - rw))
        ry = max(0, min(ry, height - rh))

        roi = image_np[ry:ry + rh, rx:rx + rw]

        if roi.size > 0:
            health_score = perform_hybrid_analysis(roi)
            draw_diagnostic_overlay(annotated_image, rx, ry, rw, rh, health_score)

    _, encoded_image = cv2.imencode('.png', annotated_image)
    return Response(content=encoded_image.tobytes(), media_type="image/png")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
