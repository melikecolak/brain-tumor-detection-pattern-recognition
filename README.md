# Brain Tumor Detection — Pattern Recognition Project

A study comparing **three different approaches** for brain tumor detection from MRI images (binary classification: tumor / healthy). The goal is not to find a single "best model," but to place deep learning, feature extraction, and traditional methods side by side on the same dataset and observe how they behave.

## Dataset

[Kaggle — Brain Tumor Detection MRI](https://www.kaggle.com/datasets/abhranta/brain-tumor-detection-mri) (Abhranta Panigrahi).
1500 tumor + 1500 healthy = 3000 labeled images were used (the `pred` folder was excluded since it is unlabeled).

## Methods

**Method 1 — Pure Deep Learning (hold-out 70/10/20)**
6 architectures were trained with ImageNet weights using two-phase transfer learning: EfficientNetB0, Xception, InceptionV3, VGG-19, ResNet-101, DenseNet-121.

**Method 2 — DL Feature Extraction + ML (5-Fold CV)**
The same 6 architectures were used as frozen feature extractors; the resulting vectors were classified with 5 ML algorithms (SVM, Random Forest, XGBoost, KNN, Logistic Regression). 6 × 5 = 30 combinations.

**Method 3 — Handcrafted Feature Extraction + ML (5-Fold CV)**
6 traditional methods (GLCM, Hu Moments, histogram statistics, HOG, Gabor, LBP) were extracted and classified with the same 5 ML algorithms. The individual contribution of each method was also examined (ablation).

## Folder Structure

```
veri_hazirlik/      data splitting notebook + split_index.json
yontemler/
├── yontem1/        pure DL: notebook, plots, results table, JSON results
├── yontem2/        DL FE + ML: notebook, plots, results table, FE metadata
└── yontem3/        handcrafted FE + ML: notebook, plots, results table
```

> Note: Trained model weights (`.keras`) and feature files (`.npy`) were excluded from the repository due to size. These files are regenerated when the notebooks are run from scratch in Google Colab.

## Running

The notebooks were written for Google Colab (T4 GPU) + Google Drive. Order:
1. `veri_hazirlik/01_data_prep.ipynb` — downloads and splits the data
2. `yontemler/yontem1/method1_DL.ipynb`
3. `yontemler/yontem2/method2_DL_FE_ML.ipynb`
4. `yontemler/yontem3/method3_handcrafted_FE_ML.ipynb`

`random_state = 42` was used throughout for reproducibility.
