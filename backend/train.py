import pandas as pd
import os
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet50
from torch import nn
from tqdm import tqdm

from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    get_scheduler
)

# CONFIG
TEXT_PATH = "D:/MultimodalFakeNewsAI/data/multimodal_train.tsv"
IMAGE_DIR = "D:/MultimodalFakeNewsAI/data/images"
MODEL_DIR = "D:/MultimodalFakeNewsAI/model"
os.makedirs(MODEL_DIR, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 8
EPOCHS = 3

# === TEXT CLASSIFICATION ===
print("\n=== TEXT CLASSIFICATION TRAINING ===")
text_df = pd.read_csv(TEXT_PATH, sep="\t")[["clean_title", "2_way_label"]].dropna()
text_df = text_df.sample(n=3000, random_state=42)
text_df.columns = ["text", "label"]

from datasets import Dataset as HFDataset
text_data = HFDataset.from_pandas(text_df)
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=128)

text_data = text_data.map(tokenize, batched=True)
text_data.set_format("torch", columns=["input_ids", "attention_mask", "label"])
text_loader = DataLoader(text_data, batch_size=BATCH_SIZE, shuffle=True)

text_model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2).to(DEVICE)
text_optimizer = torch.optim.AdamW(text_model.parameters(), lr=5e-5)
scheduler = get_scheduler("linear", optimizer=text_optimizer, num_warmup_steps=0, num_training_steps=len(text_loader) * EPOCHS)

# Training Text Model
text_model.train()
total_correct_text = 0
total_samples_text = 0

for epoch in range(EPOCHS):
    correct = total = 0
    loop = tqdm(text_loader, desc=f"Text Epoch {epoch+1}")
    for batch in loop:
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        outputs = text_model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["label"]
        )
        loss = outputs.loss
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1)
        correct += (preds == batch["label"]).sum().item()
        total += batch["label"].size(0)

        text_optimizer.zero_grad()
        loss.backward()
        text_optimizer.step()
        scheduler.step()
        loop.set_postfix(loss=loss.item())

    print(f"✅ Text Model Epoch {epoch+1} Accuracy: {correct / total:.4f}")
    total_correct_text += correct
    total_samples_text += total

overall_text_acc = total_correct_text / total_samples_text
print(f"📊 Overall Text Accuracy: {overall_text_acc:.4f}")
torch.save(text_model.state_dict(), os.path.join(MODEL_DIR, "text_model.pt"))
print("✅ Text model saved.")

# === IMAGE CLASSIFICATION ===
print("\n=== IMAGE CLASSIFICATION TRAINING ===")

img_df = pd.read_csv(TEXT_PATH, sep="\t")[["id", "2_way_label"]].dropna()
img_df.columns = ["image_id", "label"]
img_df["image_path"] = img_df["image_id"].apply(lambda x: os.path.join(IMAGE_DIR, f"{x}.jpg"))
img_df = img_df[img_df["image_path"].apply(os.path.exists)]

print(f"🖼️ Found {len(img_df)} images for training.")

class ImageDataset(Dataset):
    def __init__(self, df, transform):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        path = self.df.iloc[idx]["image_path"]
        label = self.df.iloc[idx]["label"]
        try:
            image = Image.open(path).convert("RGB")
            image = self.transform(image)
        except Exception:
            image = torch.zeros((3, 224, 224))  # fallback
        return image, torch.tensor(label)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

image_dataset = ImageDataset(img_df, transform)
image_loader = DataLoader(image_dataset, batch_size=BATCH_SIZE, shuffle=True)

img_model = resnet50(pretrained=True)
img_model.fc = nn.Linear(img_model.fc.in_features, 2)
img_model = img_model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
image_optimizer = torch.optim.AdamW(img_model.parameters(), lr=2e-5)

img_model.train()
total_correct_image = 0
total_samples_image = 0

for epoch in range(EPOCHS):
    correct = total = 0
    loop = tqdm(image_loader, desc=f"Image Epoch {epoch+1}")
    for images, labels in loop:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = img_model(images)
        loss = criterion(outputs, labels)
        preds = torch.argmax(outputs, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        image_optimizer.zero_grad()
        loss.backward()
        image_optimizer.step()
        loop.set_postfix(loss=loss.item())

    print(f"✅ Image Model Epoch {epoch+1} Accuracy: {correct / total:.4f}")
    total_correct_image += correct
    total_samples_image += total

overall_image_acc = total_correct_image / total_samples_image
print(f"📊 Overall Image Accuracy: {overall_image_acc:.4f}")

torch.save(img_model.state_dict(), os.path.join(MODEL_DIR, "image_model.pt"))
print("✅ Image model saved.")
