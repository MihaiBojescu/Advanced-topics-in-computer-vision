from flask import Flask, request, jsonify
from data.dataloader import SingleImageDataLoader
from nn.model.model import Model
import io
from PIL import Image

app = Flask(__name__)

model = Model()
# just hardcode path to weights ig
model.load_weights("./outputs/model_epochs25_loss162038.4062_val-loss243631.0156_1717439475669035715.weights.h5")

def get_bytes_image_data(request):
    if not request.data:
        raise ValueError("No image data found in the request")
    return io.BytesIO(request.data)

def get_form_image_data(request):
    if "image" not in request.files:
        raise ValueError("No image found in the request")
    image_file = request.files["image"]
    if not image_file or image_file.filename == "":
        raise ValueError("No image file provided")
    if image_file.content_type not in ["image/jpeg", "image/png"]:
        raise ValueError("Invalid image format. Only JPEG and PNG are supported")
    return image_file

def get_image_data(request):
    if "image" in request.files:
        return get_form_image_data(request)
    else:
        return get_bytes_image_data(request)

def predict_coordinates(image_file):
    dataloader = SingleImageDataLoader(image_file)
    image_array = dataloader.load_data()
    predictions = model.predict(image_array)
    normalized_x, normalized_y = predictions[0]

    return normalized_x, normalized_y

@app.route("/predict", methods=["POST"])
def predict():
    try:
        image_file = get_image_data(request)
        x, y = predict_coordinates(image_file)
        return jsonify({"x": float(x), "y": float(y)})

    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run()