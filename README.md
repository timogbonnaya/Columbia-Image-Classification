# 🧠 Columbia Photographic Image Classification using Neural Networks

**Author:** Timothy Ogbonnaya  
**Language:** R  

---

## 🎯 Project Overview

This project applies a **Neural Network (Multi-Layer Perceptron)** to classify images from the **Columbia Photographic Image Dataset** as *outdoor* or *not outdoor*.  

A baseline **Logistic Regression** model (using median RGB intensities) is compared against the **Neural Network** model (using a grid-based grayscale feature extraction).  

The project demonstrates model comparison using **Accuracy**, **Sensitivity**, **Specificity**, and **Misclassification Rate**.

---

## 📚 Background

The **Columbia Photographic Image Dataset** contains **800 real-world JPEG images** captured in **New York City** and **Boston** using Canon 10D and Nikon D70 cameras.  
Each image was labeled according to lighting and context conditions, such as:

- `outdoor-day`
- `indoor-dark`
- `indoor-light`
- `artificial`

The classification goal was **binary**:
> 🟩 1 = Outdoor-day  🟥 0 = Not outdoor-day

---

## 🧩 Methodology

The analysis was fully conducted in **R** using the following workflow:

### 1️⃣ Feature Extraction  
- Each image was resized to **100×100** pixels and converted to **grayscale**.  
- Then partitioned into a **10×10 grid**, producing **100 mean intensity features** per image.  
- This process retains **spatial brightness patterns** unlike simple color averages.

### 2️⃣ Metadata Loading  
- The dataset’s metadata (`photoMetaData.csv`) provided filenames and categorical labels for training.

### 3️⃣ Data Preparation  
- Images with missing pixel information were imputed using **mean imputation**.  
- Data was split: **70% training** and **30% testing**.

### 4️⃣ Model Training  
Two models were implemented:
- **Baseline Logistic Regression:** 3 predictors (median RGB values)
- **Neural Network:** 100 grayscale grid features, 1 hidden layer (5 nodes), sigmoid output

### 5️⃣ Evaluation Metrics  
- **Accuracy** = (TP + TN) / Total  
- **Sensitivity** = TP / (TP + FN)  
- **Specificity** = TN / (TN + FP)  
- **Misclassification Rate** = 1 − Accuracy  

### 6️⃣ Visualization  
- Distribution of image categories  
- Histogram of predicted probabilities for both models  

---

## 🧮 Model Comparison

| Metric              | Logistic Regression | Neural Network |
|---------------------|---------------------|----------------|
| **Accuracy**        | 73.38%              | 63.75%         |
| **Sensitivity**     | 50.69%              | 38.37%         |
| **Specificity**     | 86.04%              | 77.92%         |
| **Misclassification** | 26.61%             | 36.25%         |

🟢 **Result:** Logistic Regression outperformed the Neural Network.  
Despite the richer feature representation, grayscale processing made it difficult for the neural network to differentiate *indoor lighting* from *outdoor light*.

---

## 🔍 Discussion

Although Neural Networks are excellent for non-linear modeling, this study highlights that:

- The **quality of features** is more important than quantity.  
- Grayscale processing caused **loss of color cues** (e.g., sunlight vs. artificial light).  
- Simpler linear models can outperform complex networks on small datasets.  

### 🔮 Future Improvements
- Use **RGB 10×10 grids** (300 features) instead of grayscale.  
- Experiment with **Convolutional Neural Networks (CNNs)** for spatial awareness.  
- Apply **Principal Component Analysis (PCA)** to reduce feature redundancy.

---

## 🧠 Key Learnings

- **Feature representation drives model success.**  
- **Neural Networks need careful tuning** (architecture, regularization, scaling).  
- Logistic regression remains a **strong baseline** for interpretable image classification.

---

## 🧰 Tools and Libraries

| Library | Purpose |
|----------|----------|
| `jpeg` | Read and process JPEG images |
| `nnet` | Train single-layer neural networks |
| `caret` | Confusion matrix and model evaluation |
| `pROC` | ROC/AUC analysis |
| `ggplot2` | Data visualization |

---



