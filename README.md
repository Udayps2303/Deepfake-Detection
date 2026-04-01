# DeepFake Detection: Combating AI-Generated Synthetic Media

A deep learning project to detect AI-generated fake faces using transfer learning with VGG16 on the 140K Real and Fake Faces dataset.

---

## Overview

This project builds a binary image classifier to distinguish between real and AI-generated (deepfake) human faces. It leverages the power of VGG16 — a pre-trained convolutional neural network — fine-tuned on a balanced subset of the 140K Real and Fake Faces dataset from Kaggle.

---

## Dataset

- **Source:** [140K Real and Fake Faces – Kaggle](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces)
- **Subset Used:**
  - Train: 20,000 images per class (Real / Fake)
  - Validation: 5,000 images per class
  - Test: 5,000 images per class

---

## Model Architecture

- **Base Model:** VGG16 (pre-trained on ImageNet, frozen layers)
- **Custom Head:**
  - GlobalAveragePooling2D
  - Dense(256, ReLU)
  - Dropout(0.5)
  - Dense(1, Sigmoid) — binary output

- **Loss:** Binary Crossentropy  
- **Optimizer:** Adam  
- **Epochs:** 8  
- **Input Size:** 224 × 224 × 3

---

## Project Structure

```
deepfake-detection/
│
├── Deepfake_VGG_16.ipynb     # Main Colab notebook
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation
```

---

## Results

| Metric         | Value        |
|----------------|--------------|
| Validation Accuracy | Evaluated post-training |
| AUC Score      | Evaluated on test set |
| Classes        | Real / Fake  |

> Detailed classification report and confusion matrix are generated inside the notebook.

---

## How to Run

### Option 1: Google Colab (Recommended)

1. Open the notebook in [Google Colab](https://colab.research.google.com/)
2. Install dependencies:
   ```bash
   !pip install tensorflow kagglehub opencv-python matplotlib scikit-learn
   ```
3. Set up your Kaggle API credentials to download the dataset via `kagglehub`
4. Run all cells in order

### Option 2: Local Setup

```bash
# Clone the repo
git clone https://github.com/your-username/deepfake-detection.git
cd deepfake-detection

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter
jupyter notebook Deepfake_VGG_16.ipynb
```

---

## Requirements

See `requirements.txt` for full dependencies.

---

## Testing on Custom Images

The notebook includes a section to upload and test your own image:

```python
predict_image("your_photo.jpeg")
```

Output will be either:
- `REAL FACE` with confidence score
- `FAKE FACE` with confidence score

---

## Tech Stack

- Python 3.x
- TensorFlow / Keras
- VGG16 (ImageNet weights)
- OpenCV
- Scikit-learn
- Matplotlib / Seaborn
- KaggleHub

---

## License

This project is for educational and research purposes only.
