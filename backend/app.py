from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50
from PIL import Image
import os

app = Flask(__name__)
CORS(app)

# Device config
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === TEXT MODEL SETUP ===
text_tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
text_model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
text_model.load_state_dict(torch.load("D:/MultimodalFakeNewsAI/model/text_model.pt", map_location=DEVICE))
text_model.eval().to(DEVICE)

# === IMAGE MODEL SETUP ===
image_model = resnet50(pretrained=False)
image_model.fc = torch.nn.Linear(image_model.fc.in_features, 2)
image_model.load_state_dict(torch.load("D:/MultimodalFakeNewsAI/model/image_model.pt", map_location=DEVICE))
image_model.eval().to(DEVICE)

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

@app.route("/predict", methods=["POST"])
def predict_text():
    data = request.get_json()
    text = data.get("text", "")
    if not text:
        return jsonify({"error": "No input text provided"}), 400

    inputs = text_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128).to(DEVICE)
    with torch.no_grad():
        outputs = text_model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        label = torch.argmax(probs, dim=1).item()

    return jsonify({
        "type": "text",
        "prediction": "Fake" if label == 1 else "Real",
        "confidence": round(float(probs[0][label]), 4)
    })

@app.route("/predict-image", methods=["POST"])
def predict_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image file uploaded"}), 400

    image_file = request.files['image']
    try:
        image = Image.open(image_file.stream).convert("RGB")
        image_tensor = transform(image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            outputs = image_model(image_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=-1)
            label = torch.argmax(probs, dim=1).item()
            label = torch.argmax(probs, dim=1).item()
            print('Label:', label)

        return jsonify({
            "type": "image",
            "prediction": "Real" if label == 1 else "Fake",
            "confidence": round(float(probs[0][label]), 4)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/", methods=["GET"])
def home():
    return "✅ Multimodal Fake News Detection API is running!"

if __name__ == "__main__":
    app.run(debug=True)
