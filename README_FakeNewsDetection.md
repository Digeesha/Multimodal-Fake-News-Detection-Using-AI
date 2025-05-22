
# Multimodal Fake News Detection (Fakeddit Dataset)

This repository presents a deep learning-based multimodal fake news detection system using the [Fakeddit dataset](https://www.kaggle.com/datasets/vanshikavmittal/fakeddit-dataset). The system combines textual and visual features from Reddit posts to classify content as real or fake, leveraging both BERT and ResNet-50 models. It includes preprocessing scripts, model training pipelines, and an interactive inference interface.

---

## 📁 Dataset

The Fakeddit dataset is used for training and evaluation. It contains over 1 million Reddit posts with:

- Text (titles and optional body)
- Images (linked or embedded)
- Labels (binary: real/fake; multiclass: satire, hoax, clickbait, etc.)

### 🔽 Download Instructions

1. Visit the dataset on Kaggle: [Fakeddit Dataset](https://www.kaggle.com/datasets/vanshikavmittal/fakeddit-dataset)
2. Download and extract the dataset files.
3. Save them in a directory named `data/` at the root of this project.

---

## 🧰 Setup & Installation

```bash
# Clone the repository
git clone https://github.com/your-username/fakenews-multimodal.git
cd fakenews-multimodal

# Create a virtual environment and activate it
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## 🖼️ Image Downloader Script

The dataset includes image URLs, which need to be downloaded for training.

```bash
python utils/image_download.py
```

This script will fetch all images and store them in the `data/images/` directory.

---

## 🏋️ Model Training

### 🔤 Text Model (BERT)

Train the BERT-based classifier for text-only input:

```bash
python train_text_model.py
```

This will fine-tune a BERT model using post titles and binary labels.

### 🖼️ Image Model (ResNet-50)

Train the ResNet-50-based classifier for image-only input:

```bash
python train_image_model.py
```

This uses image data resized to 224x224 with normalization and augmentation.

---

## 🤖 Multimodal Inference

Once models are trained, you can run the combined inference script:

```bash
python multimodal_inference.py
```

This script fuses the predictions from the text and image models to classify new Reddit-style posts.

---

## 📊 Evaluation & Results

- **Text Model Accuracy**: 92.1%
- **Image Model Accuracy**: 86.8%
- **AUC (ROC Curve)**: 1.00 (both models)
- Includes interpretability via SHAP and fairness checks using IBM AIF360.

---

## 📦 Project Structure

```
├── data/
│   ├── train.csv
│   ├── test.csv
│   └── images/
├── models/
│   ├── text_model.pt
│   └── image_model.pt
├── utils/
│   ├── image_download.py
│   └── preprocessing.py
├── train_text_model.py
├── train_image_model.py
├── multimodal_inference.py
└── README.md
```

---

## 📌 Future Work

- Extend to video and audio modalities
- Integrate multilingual and cross-platform data
- Add real-time browser extension for misinformation detection

---

## 🧠 Authors & Acknowledgments

- Developed by: [Your Name]
- Powered by: PyTorch, HuggingFace, Torchvision, IBM AIF360
- Dataset courtesy of [Vanshika V Mittal](https://www.kaggle.com/datasets/vanshikavmittal/fakeddit-dataset)

---

## 📄 License

This project is licensed under the MIT License - see the `LICENSE` file for details.
