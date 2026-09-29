# Facial Skin Condition Detection Using Traditional Machine Learning

## MCA Project — Academic Use Only

---

## Project Overview

This system performs preliminary, AI-based classification of facial skin conditions using traditional
machine learning algorithms (no deep learning). It is built as a complete MCA-level academic project.

### Supported Conditions
| Class | Description |
|-------|-------------|
| Acne | Inflammatory skin condition affecting hair follicles |
| Rosacea | Chronic redness and facial flushing |
| Melasma | Dark pigmentation patches on facial skin |
| Facial Eczema | Atopic Dermatitis on facial region |
| Seborrheic Dermatitis | Scaly patches, red skin, stubborn dandruff |

---

## ⚠️ Medical Disclaimer

> This application is strictly for **academic and educational purposes**.
> It provides a **preliminary ML-based classification** and is **NOT a medical diagnosis**.
> Always consult a **qualified dermatologist** for professional skin evaluation.
> This system does **not** prescribe medicines or recommend treatments.

---

## Technology Stack

| Category | Technology |
|----------|------------|
| Language | Python 3.10+ |
| ML Framework | Scikit-learn, XGBoost |
| Image Processing | OpenCV, scikit-image |
| Data | NumPy, Pandas |
| Visualization | Matplotlib, Seaborn |
| Model Saving | Joblib |
| Web App | Flask, HTML, CSS, JavaScript |

---

## ML Pipeline

```
Image → Preprocessing → Feature Extraction → ML Model → Prediction
                              ↓
                 Color + GLCM + LBP Features
```

**No CNN, No Deep Learning, No PyTorch, No TensorFlow.**

---

## Project Structure

```
Facial_Skin_Condition_Detection/
│
├── dataset/                        ← Place your images here
│   ├── Acne/
│   ├── Rosacea/
│   ├── Melasma/
│   ├── Facial_Eczema/
│   └── Seborrheic_Dermatitis/
│
├── features/
│   └── features.csv                ← Auto-generated feature dataset
│
├── models/
│   ├── best_model.pkl              ← Saved best ML model
│   └── scaler.pkl                  ← Saved feature scaler
│
├── notebooks/
│   ├── 01_dataset_analysis.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_feature_extraction.ipynb
│   ├── 04_model_training.ipynb
│   └── 05_model_evaluation.ipynb
│
├── src/
│   ├── dataset_analysis.py         ← Dataset loading & statistics
│   ├── preprocessing.py            ← Image preprocessing pipeline
│   ├── feature_extraction.py       ← Color + GLCM + LBP features
│   ├── train_models.py             ← Train all 7 ML models
│   ├── evaluate_models.py          ← Compare & evaluate models
│   └── predict.py                  ← Single-image prediction
│
├── app/
│   ├── app.py                      ← Flask web application
│   ├── templates/index.html
│   └── static/
│       ├── style.css
│       └── script.js
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Setup Instructions

### 1. Clone / Download the project

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Add your dataset
Place images into the corresponding folders under `dataset/`:
```
dataset/Acne/          ← .jpg / .jpeg / .png images of Acne
dataset/Rosacea/       ← images of Rosacea
dataset/Melasma/       ← images of Melasma
dataset/Facial_Eczema/ ← images of Facial Eczema
dataset/Seborrheic_Dermatitis/ ← images of Seborrheic Dermatitis
```

### 5. Run dataset analysis
```bash
python src/dataset_analysis.py
```

### 6. Extract features
```bash
python src/feature_extraction.py
```

### 7. Train models
```bash
python src/train_models.py
```

### 8. Evaluate models
```bash
python src/evaluate_models.py
```

### 9. Run the web application
```bash
cd app
python app.py
```
Then open: http://localhost:5000

---

## ML Algorithms Compared

1. Support Vector Machine (SVM)
2. Random Forest
3. K-Nearest Neighbors (KNN)
4. Decision Tree
5. Logistic Regression
6. Naive Bayes
7. XGBoost

---

## Features Extracted

### Color Features (12)
- RGB: Mean + Std Dev for R, G, B channels
- HSV: Mean + Std Dev for H, S, V channels

### GLCM Texture Features (20)
- Contrast, Dissimilarity, Homogeneity, Energy, Correlation
- Extracted at multiple angles (0°, 45°, 90°, 135°) averaged

### LBP Features (26)
- Local Binary Pattern histogram (normalized)

**Total Feature Vector: ~58 features per image**

---

## Academic Information

- **Level**: MCA (Master of Computer Applications)
- **Domain**: Machine Learning + Computer Vision
- **Approach**: Traditional ML (no deep learning)
- **Purpose**: Educational / Academic

---

*This project was built for academic purposes. All results are experimental and not clinically validated.*
