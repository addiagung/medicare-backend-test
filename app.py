import os

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from PIL import Image
from google.cloud import storage

app = Flask(__name__)
app.config["ALLOWED_EXTENSIONS"] = set(['png', 'jpg', 'jpeg'])
project_id = 'medicare-test-407623'
bucket_name = 'medicare-test-407623'

# Configure Google Cloud Storage client
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r"C:\Users\addia\Downloads\key-file.json"

# Configure Google Cloud Storage client
client = storage.Client(project=project_id)
bucket = client.get_bucket(bucket_name)

# Function to check if a file has an allowed extension
def allowed_file(filename):
    return "." in filename and filename.split(".", 1)[1] in app.config["ALLOWED_EXTENSIONS"]

# Load Keras model for image classification
model = load_model("model.h5", compile=False)

# Read data obat from a CSV file and preprocess the description column
dataobat_df = pd.read_csv("dataobat.csv")
dataobat_df["deskripsi"] = dataobat_df["deskripsi"].str.replace('\n', '<br>')
columns = ["nama", "deskripsi", "dosis", "manfaat", "efek_samping", "kategori"]
labels = dataobat_df[columns].values.tolist()

# Define the index route for the API
@app.route("/")
def index():
    return jsonify({
        "status": {
            "code": 200,
            "message": "API fetched successfully",
        },
        "data": None
    }), 200

# Define the route for making detections based on input images
@app.route("/detection", methods=["POST"])
def detection():
    blob = None
    
    if request.method == "POST":
        image = request.files.get("image")
        if image and allowed_file(image.filename):
            # Save the input image to the designated folder
            filename = secure_filename(image.filename)

            # Upload the file to Google Cloud Storage
            blob = bucket.blob(filename)
            blob.upload_from_file(image)

            # Optionally, set the public access permission if you want the image to be publicly accessible
            blob.make_public()

            # Get the public URL of the uploaded image
            image_url = blob.public_url

            # Preprocess the input image for model detection
            img = Image.open(image).convert("RGB")  # Corrected line
            img = img.resize((150, 150))
            img_array = np.asarray(img)
            img_array = np.expand_dims(img_array, axis=0)
            normalized_image_array = (img_array.astype(np.float32) / 127.5) - 1
            data = np.ndarray(shape=(1, 150, 150, 3), dtype=np.float32)
            data[0] = normalized_image_array

            # Make detections using the loaded model
            prediction = model.predict(data)
            index = np.argmax(prediction)
            class_names = labels[index]
            confidence_score = prediction[0][index]

            return jsonify({
                "status": {
                    "code": 200,
                    "message": "Detection successful",
                },
                "data": {
                    "model": class_names,
                    "confidence": float(confidence_score),
                    "image_url": image_url,
                }
            }), 200
        else:
            return jsonify({
                "status": {
                    "code": 400,
                    "message": "Client-side error"
                },
                "data": None
            }), 400
    else:
        return jsonify({
            "status": {
                "code": 405,
                "message": "Method not allowed"
            },
            "data": None,
        }), 405

# Run the Flask application
if __name__ == "__main__":
    app.run()
