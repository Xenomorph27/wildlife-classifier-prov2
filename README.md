<div align="center">

# 🦌 Wildlife Classifier Pro v2

### Detect every animal in a photo, then name the species — 52 classes, ~91.5% accuracy

A two-stage computer-vision pipeline: **YOLOv8x** finds each animal and crops it,
then a fine-tuned **EfficientNet-B4** classifies every crop with a confidence score.

![Python](https://img.shields.io/badge/Python-3776ab?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white)
![YOLOv8](https://img.shields.io/badge/YOLOv8x-111f68?logo=yolo&logoColor=white)
![EfficientNet](https://img.shields.io/badge/EfficientNet--B4-4285f4)
![OpenCV](https://img.shields.io/badge/OpenCV-5c3ee8?logo=opencv&logoColor=white)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

<br/>

<img src="docs/results/mongoose-mob.jpg" alt="Six mongooses on a termite mound, each in its own box labelled mongoose with 84–99% confidence" width="80%">

<sub>Six mongooses, six detections, every one classified correctly (<code>--min-conf 50</code>).</sub>

</div>

---

## How it works

```mermaid
flowchart LR
  A[📷 Image] --> B[YOLOv8x<br/>animal detection]
  B -->|one crop per animal| C[BGR → RGB<br/>resize 380×380]
  C --> D[EfficientNet-B4<br/>fine-tuned, 52 classes]
  D --> E[Softmax<br/>label + confidence]
  E --> F[🖼️ Annotated image]
```

Why two stages? A whole-image classifier has to pick one label, even when a photo holds several
animals or one small animal in a big scene. Detecting first means each animal is classified on its
own tightly cropped region, at the classifier's full 380×380 input resolution.

| | |
|---|---|
| **Detector** | YOLOv8x (Ultralytics), confidence threshold 0.25 |
| **Classifier** | EfficientNet-B4, ImageNet-pretrained, fine-tuned with transfer learning |
| **Input** | 380 × 380 RGB crops |
| **Classes** | 52 species |
| **Accuracy** | ~91.5% classification accuracy |
| **Multi-animal** | Every detection gets its own box, label and confidence |

## Examples

<p align="center">
  <img src="classifier-version-pro-v2/example_images/img1.jpeg" alt="Polar bear and cub, both detected and labelled polar_bear" width="49%">
  <img src="classifier-version-pro-v2/example_images/img2.jpeg" alt="Opossum on a log, labelled opossum" width="35%">
</p>
<p align="center">
  <img src="classifier-version-pro-v2/example_images/img3.jpeg" alt="Opossum carrying babies, labelled opossum" width="49%">
  <img src="classifier-version-pro-v2/example_images/img4.jpeg" alt="Group of mongooses, each labelled mongoose" width="49%">
</p>

## Supported species

<details>
<summary><b>All 52 classes</b></summary>
<br/>

| | | | |
|---|---|---|---|
| alligator | alpaca | anteater | antelope |
| arctic fox | armadillo | axis deer | baboon |
| badger | beaver | bison | black bear |
| boar | bobcat | brown bear | buffalo |
| cat | coyote | crocodile | dog |
| donkey | elephant | elk | fallow deer |
| gray fox | groundhog | hippopotamus | horse |
| jackal | koala | lynx | mongoose |
| moose | mountain goat | mountain lion | opossum |
| otter | pig | polar bear | porcupine |
| rabbit | raccoon | red fox | reindeer |
| sika deer | skunk | squirrel | sugar glider |
| weasel | wolf | wombat | yak |

</details>

## Quick start

The model weights (`best_model_v2.pt`, `yolov8x.pt`) are stored with Git LFS, so install
[Git LFS](https://git-lfs.com) before cloning.

```bash
git lfs install
git clone https://github.com/Xenomorph27/wildlife-classifier-prov2.git
cd wildlife-classifier-prov2
pip install -r requirements.txt
```

```bash
# classify the bundled test image and open a preview window
python classifier-version-pro-v2/predict.py

# your own images: save annotated copies, skip the window, hide low-confidence guesses
python classifier-version-pro-v2/predict.py photo1.jpg photo2.jpg --save results/ --no-show --min-conf 50
```

| Option | |
|---|---|
| `images …` | One or more image files (default: `test.jpg`) |
| `--save DIR` | Write annotated images to `DIR` |
| `--min-conf PCT` | Hide predictions below this confidence, 0–100 |
| `--no-show` | Don't open the OpenCV preview window |

Runs on CPU out of the box and uses CUDA automatically when available.

## Project structure

```
classifier-version-pro-v2/
├── predict.py          # detection + classification pipeline (CLI)
├── best_model_v2.pt    # fine-tuned EfficientNet-B4 weights (Git LFS)
├── yolov8x.pt          # YOLOv8x detector weights (Git LFS)
├── test.jpg            # sample input
└── example_images/     # sample outputs
docs/results/           # README showcase output
```

## Related

[**edge-wildlife-classifier**](https://github.com/Xenomorph27/edge-wildlife-classifier) — the lightweight
predecessor: a 3-class ResNet18 for live webcam inference on edge hardware.

## License

[MIT](LICENSE) © Barnak Roy Nandy
