**Heart Attack Prediction Project**

In this project, the goal was to predict the risk of heart attack using machine learning algorithms and compare their performance. The dataset was obtained from Kaggle. After trying multiple classification models including Logistic Regression, Decision Tree, Support Vector Machine (SVM), and Random Forest, the best performance was achieved with the Random Forest algorithm.

---

## Contents

1. [Project Steps](#project-steps)
2. [Techniques and Tools Used](#techniques-and-tools-used)
3. [Model Performances](#model-performances)
4. [Conclusion and Suggestions](#conclusion-and-suggestions)

---

## Project Steps

### Initial Setup

* Defined the objective of the project.
* Identified and downloaded the dataset from Kaggle.

### Data Understanding & Preprocessing

* Analyzed the dataset structure and recognized all numeric and categorical features.
* Examined missing values, unique values, and performed data cleaning where necessary.
* Conducted exploratory data analysis using distribution plots, pie charts, count plots, box plots, swarm plots, and heatmaps.
* Applied FacetGrid and PairPlot to understand variable relationships.
* Outliers were visualized and handled in features like `trtbps`, `thalach`, and `oldpeak`.
* Applied one-hot encoding for categorical features.
* Used RobustScaler to scale numeric features for machine learning compatibility.
* Split the dataset into training and testing sets.

### Feature Engineering

* Low-correlation columns were removed.
* Distribution transformations were applied for skewed data.
* A new DataFrame was created using the `melt()` function for comparative analysis.

---

## Modeling Phase

### Models Trained and Evaluated:

* **Logistic Regression**

  * Achieved 0.87 accuracy and 0.88 AUC.

* **Decision Tree**

  * Accuracy: 0.83, AUC: 0.76

* **Support Vector Machine (SVM)**

  * Accuracy: 0.83, AUC: 0.89

* **Random Forest**

  * Achieved the best performance with 0.87 accuracy and 0.93 AUC.

### Additional Techniques Used:

* Cross-validation was used for model reliability.
* ROC Curve and AUC values were visualized and compared.
* Hyperparameter optimization was conducted using GridSearchCV.

---

## Techniques and Tools Used

* **Languages & Platforms:** Python 3.x, Jupyter Notebook
* **Libraries:** pandas, numpy, matplotlib, seaborn, scikit-learn
* **Visualizations:** Distplot, Pair Plot, Heatmap, Count Plot, Boxplot, Pie Chart, Swarm Plot
* **Evaluation Metrics:** Accuracy, ROC-AUC, Cross-Validation

---

## Model Performances

| Model                  | Accuracy | AUC      |
| ---------------------- | -------- | -------- |
| Logistic Regression    | 0.87     | 0.88     |
| Decision Tree          | 0.83     | 0.76     |
| Support Vector Machine | 0.83     | 0.89     |
| Random Forest          | **0.87** | **0.93** |

---

## Conclusion and Suggestions

* Random Forest model provided the best results in terms of both accuracy and AUC.
* For further improvements, boosting algorithms like XGBoost, LightGBM, or CatBoost can be considered.
* It is recommended to evaluate additional metrics like precision, recall, and F1-score, especially in imbalanced datasets.
* Future work could involve implementing feature selection techniques and testing ensemble models.

---

**Project Repository and Dataset Link:** \[Insert Kaggle dataset and notebook link here]

*End of README.*
