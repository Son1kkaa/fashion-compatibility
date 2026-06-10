import os
import pickle
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

MODEL_PATH = "models/best_smoothed_model.pth"
INDEX_PATH = "models/wardrobe_index.pkl"
WARDROBE_DIR = "static/wardrobe"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 224

img_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class SiameseClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet50(weights=None)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.projection = nn.Sequential(
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4)
        )
        self.classifier = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )

    def get_projection(self, img):
        feat = self.backbone(img).squeeze(-1).squeeze(-1)
        return self.projection(feat)


def main():
    print(f"Завантаження моделі: {MODEL_PATH}")
    model = SiameseClassifier().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    projections, items = [], []

    if not os.path.exists(WARDROBE_DIR):
        print(f"Папка {WARDROBE_DIR} не знайдена!")
        print(f"Створи папки: static/wardrobe/tops/  bottoms/  shoes/")
        return

    for category in sorted(os.listdir(WARDROBE_DIR)):
        cat_dir = os.path.join(WARDROBE_DIR, category)
        if not os.path.isdir(cat_dir):
            continue
        for fname in sorted(os.listdir(cat_dir)):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            path = os.path.join(cat_dir, fname)
            try:
                img = Image.open(path).convert("RGB")
            except Exception as e:
                print(f"  Помилка: {path} -- {e}")
                continue
            tensor = img_transform(img).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                proj = model.get_projection(tensor).cpu().squeeze(0)
            projections.append(proj)
            items.append({
                "filename": f"wardrobe/{category}/{fname}",
                "category": category,
            })
            print(f"  + {category}/{fname}")

    if not projections:
        print("Жодного зображення не знайдено!")
        return

    index = {"projections": torch.stack(projections), "items": items}
    with open(INDEX_PATH, "wb") as f:
        pickle.dump(index, f)
    print(f"\nПроіндексовано {len(items)} елементів -> {INDEX_PATH}")


if __name__ == "__main__":
    main()
