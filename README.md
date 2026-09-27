# Concrete Crack Detection: SVM vs. CNN

Binary image classification that detects **cracks in concrete surfaces**. It compares a classical machine-learning baseline (Support Vector Machines on raw pixels) with a small convolutional neural network built in PyTorch.

This is a case study from **EMIA4110 (Practical Machine Learning)**. The course provided a notebook template for each part, and I completed the data loading, model, training and evaluation code.

## Why it matters

Automatically flagging cracks in images of infrastructure or factory equipment supports **predictive maintenance**. Real defects can be caught early and sent for inspection, instead of relying only on manual visual checks.

## Dataset

- 10,000 greyscale concrete surface images at **30×30 px**, perfectly balanced: 5,000 with cracks (`Positive_img`) and 5,000 without (`Negative_img`).
- A downscaled subset of [Concrete Crack Images for Classification](https://data.mendeley.com/datasets/5y9wdsg2zt/2) (Özgenel, Mendeley Data).
- The data is not included in this repo. See [`data/README.md`](data/README.md) for how to get it.

## What I did

### 1. SVM baseline: [`notebooks/01_svm_crack_classification.ipynb`](notebooks/01_svm_crack_classification.ipynb)

- Loaded every image in greyscale with OpenCV, resized it to 30×30 and **flattened it into a 900-value feature vector**. Labels are 1 = crack, 0 = no crack.
- Stratified **80/20 train/test split** (`random_state=42`).
- Standardised the features with `StandardScaler`, fitted on the training set only to avoid leakage.
- Trained an `SVC` with a **polynomial kernel**, then swapped in an **RBF kernel** to compare the two.

### 2. CNN: [`notebooks/02_cnn_crack_classification.ipynb`](notebooks/02_cnn_crack_classification.ipynb)

- Wrote a custom PyTorch `Dataset` that loads each image, scales its pixels to [0, 1] and returns a `(1, 30, 30)` tensor with its label.
- **80/10/10 train/validation/test split** (seeded) with batch size 32.
- Designed a deliberately tiny CNN with **1,122 trainable parameters**:

  ```
  Input (1×30×30)
  → Conv2d(1→4, 3×3, pad 1) → ReLU → MaxPool 2×2   # 4×15×15
  → Conv2d(4→8, 3×3, pad 1) → ReLU → MaxPool 2×2   # 8×7×7
  → Flatten (392) → Linear(392→2)                  # logits
  ```

- Trained for 10 epochs with **Adam (lr = 0.001)** and **cross-entropy loss**, checkpointing the model with the best validation accuracy. That best checkpoint was then evaluated on the held-out test set.

## Results

| Model | Features | Test accuracy |
|---|---|---|
| SVM, polynomial kernel | raw pixels (900-d) | 86.65% |
| SVM, RBF kernel | raw pixels (900-d) | 96.75% |
| **CNN (1,122 params)** | learned conv features | **99.0%** |

The CNN's best validation accuracy was 98.9% (epoch 7), and the test result is from that checkpoint.

**Takeaways**

- **The choice of kernel matters a lot.** Switching the SVM from polynomial to RBF gained about 10 percentage points, which shows the boundary between crack and no-crack in pixel space is highly non-linear.
- **Learned features beat raw pixels.** A CNN with only about 1.1k parameters beat the best SVM by more than 2 points. Its convolutional filters pick up local edge and line patterns, which is what a crack looks like, regardless of where the crack sits in the image.
- The training and validation losses stay close together throughout, so there's no sign of overfitting at this model size.

## Tools

- **Python 3**, run on **Google Colab**
- **scikit-learn**: `SVC`, `StandardScaler`, `train_test_split`, metrics
- **PyTorch**: `Dataset`/`DataLoader`, `nn.Conv2d`, Adam, `CrossEntropyLoss`
- **OpenCV** and **Pillow** for image loading, **NumPy**, **Matplotlib**, **tqdm**

## How to run

### Option A: Google Colab (easiest)

1. Open a notebook from `notebooks/` in [Google Colab](https://colab.research.google.com/) (**File → Upload notebook**).
2. Run all cells. When Step 2 asks for a file, upload `Concrete_Crack_Images.zip`.

A GPU isn't required. The SVMs take under a minute each, and the CNN trains in about 1–2 minutes on CPU.

### Option B: Locally

```bash
git clone <this-repo-url>
cd <repo-folder>
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

# put Concrete_Crack_Images.zip in data/ (see data/README.md)
cd notebooks
jupyter notebook
```

Then run `01_svm_crack_classification.ipynb` and `02_cnn_crack_classification.ipynb` top to bottom. Step 2 extracts the dataset from `../data/` automatically.

Results can vary slightly between runs, because the SVM split is seeded but the CNN's weight initialisation and batch shuffling are not.

## Project structure

```
.
├── notebooks/
│   ├── 01_svm_crack_classification.ipynb
│   └── 02_cnn_crack_classification.ipynb
├── data/
│   └── README.md          # how to get the dataset (data itself is git-ignored)
├── requirements.txt
└── README.md
```

## Acknowledgements

- Case study and notebook templates: EMIA4110, Practical Machine Learning.
- Dataset: Özgenel, Ç. F. *Concrete Crack Images for Classification*, Mendeley Data. [data.mendeley.com/datasets/5y9wdsg2zt/2](https://data.mendeley.com/datasets/5y9wdsg2zt/2)
