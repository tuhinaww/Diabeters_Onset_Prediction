# Diabetes Onset Prediction Project

This project aims to predict the likelihood of diabetes onset in patients using machine learning models, with a special focus on **Logistic Regression** and **Feature Engineering**. We’ve also explored additional models, including **K-Nearest Neighbors**, **Decision Trees**, and **Random Forest**, comparing their performance to determine the most effective model for predicting diabetes onset. Based on our evaluation metrics, **Logistic Regression** showed the highest accuracy and efficiency for this dataset.

## Team Members

| Name                 | Registration Number     |
|----------------------|-------------------------|
| Tuhina Tripathi      | RA2211003010423         |
| Utkarshini Narayan   | RA2211003010403         |
| Krishna Jain         | RA2211003010429         |
| Akanksha Rathore     | RA2211003010396         |

---

## Project Overview

Diabetes is a significant health issue worldwide, affecting millions of people. Early prediction of diabetes onset can be critical for managing symptoms and improving patient outcomes. In this project, we applied machine learning techniques to analyze and predict the likelihood of diabetes onset using the **Pima Indians Diabetes dataset**. 

### Objectives

- To predict diabetes onset using patient health data.
- To compare multiple machine learning models and evaluate their performance.
- To demonstrate the effectiveness of **Logistic Regression** over other classification models.

---

## Methodology

### 1. **Data Preparation and Exploration**
   - **Data Cleaning**: Checked and handled missing values.
   - **Exploratory Data Analysis (EDA)**: Visualized feature distributions, correlations, and key patterns in the dataset.

### 2. **Feature Engineering**
   - **Feature Scaling**: Standardized data for model consistency.
   - **Feature Creation**: Tested with different feature combinations to improve prediction accuracy.

### 3. **Model Building**
   - **Logistic Regression**: Focus model for prediction.
   - **Additional Models**: Trained **K-Nearest Neighbors**, **Decision Trees**, and **Random Forest** for comparison.

### 4. **Evaluation Metrics**
   - **Accuracy**: Correct predictions over the total predictions.
   - **Precision, Recall, F1-score**: Evaluated model sensitivity and predictive power.
   - **ROC-AUC**: Assessed model’s ability to distinguish between classes.
   - **Confusion Matrix**: Visualized model performance in correct vs. incorrect predictions.

### 5. **Model Comparison**
   - **ROC Curves**: Plotted and compared ROC-AUC scores of all models.
   - **Model Comparison Graph**: Showcased the superior performance of Logistic Regression over other models.

---

## Installation and Setup

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-repo/diabetes_onset_prediction.git
   cd diabetes-prediction

2. **Generate requirements.txt**:
   ```bash
   pip freeze > requirements.txt


3. **Install Required Libraries**:
   ```bash
   pip install -r requirements.txt

4. **Open the notebook diabetes_onset_prediction.ipynb to view the analysis, graphs, and model results.**:

---

## Usage

The project notebook guides you through each step of the analysis:

1. **Data Loading**: Load and examine the data.
2. **Data Visualization**: View feature distributions and correlations.
3. **Model Training and Evaluation**: Train Logistic Regression, Decision Tree, and Random Forest models, then evaluate and compare their performance.
4. **Model Comparison**: View graphs comparing each model’s effectiveness.

### Included Graphs:
- Feature distribution histograms.
- Correlation heatmaps.
- Confusion matrices for each model.
- ROC Curves for model comparison.
- Model performance comparison plot.
 ---

## Results

| Model                | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|----------------------|----------|-----------|--------|----------|---------|
| **Logistic Regression** | 0.74     | 0.80      | 0.79   | 0.80     | 0.90    |
| **Decision Tree**      | 0.71     | 0.81      | 0.72   | 0.76     | 0.80    |
| **Random Forest**      | 0.74     | 0.80      | 0.80   | 0.80     | 0.88    |

---
## Conclusion

In this project, **Logistic Regression** with effective feature engineering was shown to be the best model for predicting diabetes onset. The analysis revealed that Logistic Regression outperformed other models, such as **Decision Trees** and **Random Forests**, in terms of accuracy, precision, recall, and ROC-AUC score. This highlights the importance of selecting the right model based on the dataset and the problem at hand.

The project demonstrates the significance of data preparation and feature engineering in achieving optimal model performance. By leveraging these techniques, we were able to enhance the predictive capabilities of our models, making them valuable tools for early diabetes diagnosis.

---

## Future Improvements

To further enhance this project, the following improvements can be considered:

- **Advanced Ensemble Methods**: Implementing advanced ensemble techniques such as **Gradient Boosting** or **XGBoost** could lead to better accuracy and performance by combining the strengths of multiple models.

- **Deep Learning Models**: Exploring deep learning architectures, such as neural networks, may capture more complex relationships in the data, potentially improving prediction accuracy.

- **Feature Engineering Enhancements**: Investigating additional feature engineering techniques, including polynomial features and interaction terms, could provide deeper insights and improve model performance.

- **Hyperparameter Tuning**: Systematically tuning hyperparameters using methods like Grid Search or Random Search can optimize model performance and enhance predictive accuracy.

- **Incorporating More Data**: Using larger and more diverse datasets could improve model generalization, leading to better performance in real-world applications.

- **Model Deployment**: Developing a web application to deploy the model would allow users to input patient data and receive instant predictions, making the tool more accessible for healthcare professionals.

---



