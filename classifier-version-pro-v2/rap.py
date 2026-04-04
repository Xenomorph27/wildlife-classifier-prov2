import torch
import cv2
from torchvision import transforms, models
from ultralytics import YOLO
from PIL import Image

# ======================
# PATHS
# ======================
CLASSIFIER_PATH = "best_model_v2.pt"
YOLO_PATH = "yolov8x.pt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======================
# LOAD YOLO
# ======================
yolo_model = YOLO(YOLO_PATH)

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
def run_pipeline(image_path):
    img = cv2.imread(image_path)

    if img is None:
        print("Image not found ❌")
        return

    results = yolo_model(img, conf=0.25)

    boxes = results[0].boxes

    if boxes is None or len(boxes) == 0:
        print("No animal detected ❌")
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

        # Draw
        cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(img, f"{label} ({confidence:.1f}%)",
                    (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0,255,0), 2)

    cv2.imshow("Result", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# ======================
# RUN
# ======================
if __name__ == "__main__":
    run_pipeline("test.jpg")   # change image here