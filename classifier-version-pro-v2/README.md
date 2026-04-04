# Wildlife-Classifier-prov2
Advanced wildlife classification system supporting multiple species using deep learning and computer vision.

---

##  Supported Animal Classes
The model is trained to classify the following 52 wildlife species:

alligator, alpaca, anteater, antelope, arctic_fox, armadillo, axis_deer, baboon, badger, beaver, bison, black_bear, boar, bobcat, brown_bear, buffalo, cat, coyote, crocodile, dog, donkey, elephant, elk, fallow_deer, gray_fox, groundhog, hippopotamus, horse, jackal, koala, lynx, mongoose, moose, mountain_goat, mountain_lion, opossum, otter, pig, polar_bear, porcupine, rabbit, raccoon, red_fox, reindeer, sika_deer, skunk, squirrel, sugar_glider, weasel, wolf, wombat, yak 

##  Features

- Multi-species classification  
- Deep learning-based model  
- Scalable architecture  
- Handles real-world wildlife images  
- Extensible for new species  

---

##  System Overview

The system follows a complete machine learning pipeline:

1. Data collection and preprocessing  
2. Model training  
3. Model evaluation  
4. Prediction and inference  

---

##  Project Structure


wildlife-classifier-pro/         
│                     
├── rap.py  # Model training script                                  
├── best_model_v2.pt / # Trained models         
├── yolov8x.pt / # Output predictions       
├── requirements.txt # Dependencies               
└── README.md


---

##  Installation

```bash
git clone https://github.com/Xenomorph27/wildlife-classifier-prov2.git
cd wildlife-classifier-prov2
pip install -r requirements.txt
```
## Usage
In Terminal navigate to respective folder and and run:
```bash
python rap.py
```
