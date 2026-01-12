import uvicorn
import tensorflow as tf
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile
import io

app = FastAPI()

MODEL_PATH = "model/skin_texture_saved_model"
IMG_SIZE = 224

model = None
inference_func = None


@app.on_event("startup")
def load_model():
    global model, inference_func
    print(f"Loading model from {MODEL_PATH}...")
    try:
        model = tf.saved_model.load(MODEL_PATH)
        inference_func = model.signatures["serving_default"]
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")


def preprocess_image(image_bytes):
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image = image.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
        img_array = np.asarray(image, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        return None


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if inference_func is None:
        return {"error": "Model not loaded"}

    contents = await file.read()
    input_tensor = preprocess_image(contents)

    if input_tensor is None:
        return {"error": "Invalid image format"}

    prediction = inference_func(tf.constant(input_tensor))

    output_key = list(prediction.keys())[0]
    score = float(prediction[output_key].numpy()[0][0])

    return {
        "filename": file.filename,
        "score": score
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
