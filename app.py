import os
import pickle
import random

import torch
import torch.nn as nn
from flask import Flask, jsonify, render_template, request
from PIL import Image
from torchvision import models, transforms

app = Flask(__name__)

SIAMESE_MODEL_PATH = "models/best_smoothed_model.pth"
INDEX_PATH = "models/wardrobe_index.pkl"
WARDROBE_DIR = "static/wardrobe"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMG_SIZE = 224
COMPATIBILITY_THRESHOLD = 0.48

img_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])


def generate_recommendation(probability):
    if probability >= 90:
        tips = [
            "Елементи демонструють відмінну візуальну сумісність. Кольорова гама, текстура та стилістика гармонійно доповнюють одне одного, утворюючи цілісний образ.",
            "Модель визначила високий рівень сумісності. Поєднання кольорів та фактур виглядає збалансованим — такий образ підходить для повсякденного використання.",
        ]
        return random.choice(tips)
    elif probability >= 75:
        tips = [
            "Речі мають високий рівень візуальної сумісності. Поєднання виглядає гармонійно за кольором, фактурою та загальним стилістичним сприйняттям.",
            "Модель визначила сумісність між елементами одягу. Таке поєднання можна використовувати як основу цілісного образу.",
        ]
        return random.choice(tips)
    elif probability >= 60:
        return ("Речі мають середній рівень сумісності. Поєднання є допустимим, "
                "однак для покращення образу рекомендується звернути увагу на "
                "узгодженість кольорової гами, текстури та загального стилю елементів.")
    elif probability >= 40:
        return ("Модель визначила знижений рівень сумісності між елементами. "
                "Можливий конфлікт візерунків, кольорів або стилістичних напрямків. "
                "Рекомендується замінити один із елементів для створення більш гармонійного образу.")
    else:
        return ("Речі мають низький рівень візуальної сумісності. Модель визначила "
                "значну відмінність між елементами одягу за кольором, текстурою або стилем. "
                "Таке поєднання може виглядати негармонійно.")


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

    def classify_projections(self, proj1, proj2):
        diff = torch.abs(proj1 - proj2)
        return self.classifier(diff).squeeze(-1)

    def forward(self, img1, img2):
        feat1 = self.backbone(img1).squeeze(-1).squeeze(-1)
        feat2 = self.backbone(img2).squeeze(-1).squeeze(-1)
        proj1 = self.projection(feat1)
        proj2 = self.projection(feat2)
        diff = torch.abs(proj1 - proj2)
        return self.classifier(diff).squeeze(-1)


def load_state_if_exists(model, path):
    if not os.path.exists(path):
        return False
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.eval()
    return True


# Завантаження моделі
siamese_model = SiameseClassifier().to(DEVICE)
siamese_ready = load_state_if_exists(siamese_model, SIAMESE_MODEL_PATH)


def build_wardrobe_index():
    """Автоматична індексація бази гардероба при запуску."""
    if not os.path.exists(WARDROBE_DIR):
        return None
    projections, items = [], []
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
            except Exception:
                continue
            tensor = img_transform(img).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                proj = siamese_model.get_projection(tensor).cpu().squeeze(0)
            projections.append(proj)
            items.append({"filename": f"wardrobe/{category}/{fname}", "category": category})
    if not projections:
        return None
    index = {"projections": torch.stack(projections), "items": items}
    os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)
    with open(INDEX_PATH, "wb") as f:
        pickle.dump(index, f)
    print(f"  Проіндексовано {len(items)} елементів гардероба")
    return index


# Автоматична індексація при запуску
wardrobe = None
if siamese_ready:
    if os.path.exists(INDEX_PATH):
        with open(INDEX_PATH, "rb") as f:
            wardrobe = pickle.load(f)
        print(f"Індекс гардероба завантажено: {len(wardrobe['items'])} елементів")
    elif os.path.exists(WARDROBE_DIR):
        print("Індекс не знайдено, створюю автоматично...")
        wardrobe = build_wardrobe_index()


def preprocess_image(file_storage):
    image = Image.open(file_storage).convert("RGB")
    return img_transform(image).unsqueeze(0).to(DEVICE)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/recommend_page")
def recommend_page():
    return render_template("recommend.html")


@app.route("/predict", methods=["POST"])
def predict():
    if not siamese_ready:
        return jsonify({"error": "Модель сумісності не завантажена."}), 500
    if "image1" not in request.files or "image2" not in request.files:
        return jsonify({"error": "Завантажте два зображення для аналізу."}), 400
    file1 = request.files["image1"]
    file2 = request.files["image2"]
    if not file1.filename or not file2.filename:
        return jsonify({"error": "Одне із зображень не було обрано."}), 400
    try:
        tensor1 = preprocess_image(file1)
        tensor2 = preprocess_image(file2)
        with torch.no_grad():
            logits = siamese_model(tensor1, tensor2)
            probability = torch.sigmoid(logits).item()
        percent = round(probability * 100, 1)
        return jsonify({
            "probability": percent,
            "compatible": probability >= COMPATIBILITY_THRESHOLD,
            "recommendation": generate_recommendation(percent)
        })
    except Exception as exc:
        return jsonify({"error": f"Помилка обробки: {exc}"}), 500


@app.route("/recommend", methods=["POST"])
def recommend():
    if not siamese_ready:
        return jsonify({"error": "Модель не завантажена."}), 500
    if wardrobe is None:
        return jsonify({"error": "База гардероба порожня. Додайте фото у static/wardrobe/"}), 500
    if "image" not in request.files or not request.files["image"].filename:
        return jsonify({"error": "Завантажте зображення."}), 400

    target_category = request.form.get("category", "").strip().lower()
    top_k = int(request.form.get("k", 4))

    try:
        query_tensor = preprocess_image(request.files["image"])
        with torch.no_grad():
            query_proj = siamese_model.get_projection(query_tensor)

        all_projs = wardrobe["projections"].to(DEVICE)
        items = wardrobe["items"]

        idxs = list(range(len(items)))
        if target_category:
            idxs = [i for i in idxs if items[i]["category"] == target_category]
            if not idxs:
                return jsonify({"error": f"Немає елементів категорії '{target_category}'."}), 400

        scores = []
        with torch.no_grad():
            for i in idxs:
                item_proj = all_projs[i].unsqueeze(0)
                logit = siamese_model.classify_projections(query_proj, item_proj)
                prob = torch.sigmoid(logit).item()
                scores.append((i, round(prob * 100, 1)))

        scores.sort(key=lambda x: x[1], reverse=True)

        results = []
        for idx, score in scores[:top_k]:
            results.append({
                "filename": items[idx]["filename"],
                "category": items[idx]["category"],
                "score": score,
            })

        return jsonify({"recommendations": results})
    except Exception as exc:
        return jsonify({"error": f"Помилка обробки: {exc}"}), 500


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
