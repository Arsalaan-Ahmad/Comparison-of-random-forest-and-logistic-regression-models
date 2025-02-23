# Diabetes Prediction: Comparing Random Forest & Logistic Regression

## 📌 Project Overview
This project compares the performance of **Random Forest (RF)** and **Logistic Regression (LR)** on the Pima Indians Diabetes Dataset to predict the presence of diabetes in women aged 21+ of Pima Indian heritage. The goal is to evaluate model accuracy, computational efficiency, and robustness in handling imbalanced data. Results are benchmarked against the study by *Chang et al. (2022)*.

### Key Questions Addressed:
- Which model (RF or LR) performs better in terms of accuracy, AUC, and computational speed?
- How do feature importance and correlation impact predictions?
- Does hyperparameter tuning significantly improve performance?


## 📂 Repository Structure
├── data/ # Dataset files (Pima Indians Diabetes Dataset from Kaggle)

├── images/ # Visualizations (heatmaps, frequency distributions, ROC curves)

├── code/ # Scripts for data preprocessing, model training, and evaluation

├── models/ # Trained model files (RF and LR)

├── docs/ # Project presentation (INM431-coursework-A1-template.pptx)

└── README.md



---

## 🏥 Dataset Details
- **Source**: [Kaggle](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
- **Samples**: 768 women (8 predictors, 1 binary target)
- **Features**: Pregnancies, Glucose, Blood Pressure, Skin Thickness, Insulin, BMI, Diabetes Pedigree Function, Age
- **Preprocessing**:
  - Replaced missing values (encoded as `0`) with mean/median.
  - Normalized predictors using min-max scaling.
  - Split into 60:20:20 (train/validation/test).

---

## 🛠️ Methodology
1. **Data Cleaning & EDA**:
   - Handled missing values (mean for Glucose/BP, median for Skin Thickness/Insulin/BMI).
   - Analyzed correlations (e.g., Glucose and BMI strongly linked to diabetes).
2. **Model Training**:
   - **Random Forest**: Hyperparameter tuning (trees=150, max splits=40, leaf size=30) via 10-fold CV.
   - **Logistic Regression**: Grid search for lambda (best=0.001).
3. **Evaluation Metrics**:
   - Accuracy, AUC, Precision, Recall, F1-Score.


## 📊 Results
### Model Performance Comparison (Test Set)
| Metric               | Random Forest | Logistic Regression |
|----------------------|---------------|---------------------|
| **Accuracy**         | 78%           | 77%                 |
| **AUC**              | 0.82          | 0.75                |
| **Precision**        | 0.74          | 0.68                |
| **Recall**           | 0.65          | 0.72                |
| **F1-Score**         | 0.69          | 0.70                |
| **Training Speed**   | Slow          | Fast                |

### Key Findings:
- **RF Strengths**: Higher AUC (0.82), better precision.
- **LR Strengths**: Faster training, better recall.
- Both models struggled with class imbalance (more "no diabetes" predictions).
- RF overfits slightly on training data but generalizes well on test data.
- **Contradiction to Hypothesis**: RF had lower TP/FP rates than expected.


## 📸 Key Visualizations
### Figure 1: Frequency Distribution Before & After Imputation
Frequency Distribution

Left: Original dataset with missing values (encoded as 0).

Right: After replacing missing values with mean/median.

Shows the balanced distribution of the target variable (Outcome).

### Figure 2: Correlation Heatmap
Correlation Heatmap

Glucose, BMI, and Age show strong positive correlations with diabetes.

Skin Thickness and Insulin are less influential.

### Figure 3: Feature Importance (Random Forest vs. Logistic Regression)
Feature Importance

RF: Glucose and BMI are top predictors.

LR: Blood Pressure and Diabetes Pedigree Function dominate.

Negative correlations in LR suggest some features may hurt performance.

### Figure 4: Class Distribution
Class Distribution

Imbalanced dataset: ~65% "No Diabetes" vs. ~35% "Diabetes".

Explains why both models perform better on the majority class.

### Figure 5: ROC Curves
ROC Curves

RF AUC = 0.82: Better class separation.

LR AUC = 0.75: Moderate performance.

### Figure 6: Training vs. Validation Accuracy (Overfitting Check)
Overfitting Check

RF shows slight overfitting (higher training accuracy than validation).

LR generalizes better due to simplicity.

### Figure 7: Confusion Matrices
Confusion Matrices

RF: Higher precision (74%) but lower recall (65%).

LR: Better recall (72%) but lower precision (68%).

## 🖼️ All Visualizations

For full details, explore the images folder:

figure1.png: Frequency distributions

figure2.png: Correlation heatmap

figure3.png: Feature importance

figure4.png: Class imbalance

figure5.png: ROC curves

figure6.png: Overfitting analysis

figure7.png: Confusion matrices


## 📹 Presentation Video  
- [Download or view the video here](videos/project_explanation.mp4)  


## 📝 Lessons Learned & Future Work
### Lessons:
- **Feature Selection**: Critical for LR (e.g., excluding negatively correlated features improved accuracy).
- **Hyperparameter Tuning**: RF requires careful tuning (OOB error may outperform CV for small datasets).
- **Class Imbalance**: Addressing imbalance (e.g., SMOTE) could improve recall.

### Future Work:
- Experiment with feature engineering and advanced techniques (XGBoost, SVM).
- Compare OOB error vs. cross-validation for hyperparameter tuning.
- Expand dataset size and apply anomaly detection for outliers.

---

## 🔗 References
- [Chang et al. (2022)](https://doi.org/10.1007/s00521-022-07049-z)
- Full references in [docs/INM431-coursework-A1-template.pptx](docs/INM431-coursework-A1-template.pptx)
