# Wildlife Classifier Pro v2

Two-stage computer vision pipeline for multi-species wildlife classification 
across 52 animal classes.

## Results
- **Accuracy:** ~91.52% classification accuracy
- **Classes:** 52 animal species
- **Architecture:** YOLOv8x (detection) + EfficientNetB4 (classification)

## Pipeline
1. YOLOv8x detects animals and extracts regions of interest
2. EfficientNetB4 (fine-tuned, transfer learning) classifies each ROI
3. Softmax confidence scoring with multi-detection handling

## Preprocessing
- BGR → RGB conversion
- Resize to 380×380
- Bounding box extraction via OpenCV

## Tech Stack
- PyTorch, Ultralytics YOLOv8
- EfficientNetB4 (pretrained, fine-tuned)
- OpenCV

## Usage
```bash
pip install torch torchvision ultralytics opencv-python
python classifier-version-pro-v2/inference.py
```
