import argparse
from pathlib import Path

import torch
import cv2
from torchvision import transforms, models
from ultralytics import YOLO
from PIL import Image

# ======================
# PATHS
# ======================
HERE = Path(__file__).resolve().parent
CLASSIFIER_PATH = HERE / "best_model_v2.pt"
YOLO_PATH = HERE / "yolov8x.pt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======================
# LOAD YOLO
# ======================
yolo_model = YOLO(str(YOLO_PATH))

# ======================
# CLASS NAMES (IMPORTANT)
# ======================
class_names =  ['alligator', 'alpaca', 'anteater', 'antelope', 'arctic_fox', 'armadillo', 'axis_deer', 'baboon', 'badger', 'beaver', 'bison', 'black_bear', 'boar', 'bobcat', 'brown_bear', 'buffalo', 'cat', 'coyote', 'crocodile', 'dogs', 'donkey', 'elephant', 'elk', 'fallow_deer', 'gray_fox', 'groundhog', 'hippopotamus', 'horse', 'jackal', 'koala', 'lynx', 'mongoose', 'moose', 'mountain_goat', 'mountain_lion', 'opossum', 'otter', 'pig', 'polar_bear', 'porcupine', 'rabbit', 'raccoon', 'red_fox', 'reindeer', 'sika_deer', 'skunk', 'squirrel', 'sugar_glider', 'weasel', 'wolf', 'wombat', 'yak']


# ======================
# LOAD CLASSIFIER
# ======================
model = models.efficientnet_b4(weights=None)
model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, len(class_names))

model.load_state_dict(torch.load(CLASSIFIER_PATH, map_location=device))
model.to(device)
model.eval()

# ======================
# TRANSFORM
# ======================
transform = transforms.Compose([
    transforms.Resize((380, 380)),
    transforms.ToTensor(),
])

# ======================
# PIPELINE
# ======================
def run_pipeline(image_path, save_path=None, show=True, min_conf=0.0):
    img = cv2.imread(str(image_path))

    if img is None:
        print(f"Image not found ❌ ({image_path})")
        return

    results = yolo_model(img, conf=0.25, verbose=False)

    boxes = results[0].boxes

    if boxes is None or len(boxes) == 0:
        print(f"No animal detected ❌ ({image_path})")
        return

    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        crop = img[y1:y2, x1:x2]

        if crop.size == 0:
            continue

        # Convert to PIL
        crop_pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))

        # Transform
        input_tensor = transform(crop_pil).unsqueeze(0).to(device)

        # Predict
        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.softmax(outputs, dim=1)
            conf, pred = torch.max(probs, 1)

        label = class_names[pred.item()]
        confidence = conf.item() * 100
        if confidence < min_conf:
            continue
        print(f"{Path(image_path).name}: {label} ({confidence:.1f}%)")

        # Draw
        cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(img, f"{label} ({confidence:.1f}%)",
                    (x1, max(y1-10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0,255,0), 2)

    if save_path:
        cv2.imwrite(str(save_path), img)

    if show:
        cv2.imshow("Result", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# ======================
# RUN
# ======================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect and classify wildlife in images.")
    parser.add_argument("images", nargs="*", default=[str(HERE / "test.jpg")],
                        help="image file(s) to classify (default: test.jpg)")
    parser.add_argument("--save", metavar="DIR",
                        help="write annotated images to this folder")
    parser.add_argument("--min-conf", type=float, default=0.0, metavar="PCT",
                        help="hide predictions below this confidence (0-100)")
    parser.add_argument("--no-show", action="store_true",
                        help="don't open a preview window")
    args = parser.parse_args()

    if args.save:
        Path(args.save).mkdir(parents=True, exist_ok=True)

    for image in args.images:
        out = Path(args.save) / Path(image).name if args.save else None
        run_pipeline(image, save_path=out, show=not args.no_show, min_conf=args.min_conf)
