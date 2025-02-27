Task 1: Classification Problem
Dataset:
Wine Quality Dataset
Source: UCI Repository
Features: 11 features including acidity, alcohol, pH, etc.
Target: Wine quality, represented as an integer value between 0 and 10.
Problem Overview:
In this task, we aim to predict the quality of wines based on their chemical properties. Using the Wine Quality dataset, the objective is to classify wines into different quality categories using machine learning models.

Process:
Exploratory Data Analysis (EDA):

Analyzed the distribution of features and their correlations using visualization tools like Seaborn and Matplotlib.
Handled missing values, scaled the features, and removed any outliers to ensure clean data for modeling.
Preprocessing:

Standardized the data using StandardScaler to bring all features to a similar scale.
Split the dataset into training and testing sets for model evaluation.
Models Applied:

Logistic Regression: A simple linear model to estimate the probability of wine quality classes.
Support Vector Machine (SVM): A powerful model for high-dimensional data classification.
Random Forest Classifier: A robust ensemble method to capture complex patterns.
Decision Tree Classifier: A simple tree-based model for understanding the decision-making process.
Model Evaluation:

Evaluated the models based on accuracy, confusion matrix, precision, recall, and F1-score.
The Random Forest classifier achieved the highest accuracy of 82%.
Key Insights:
The Random Forest model outperformed other models, demonstrating its strength in handling imbalanced datasets and capturing complex relationships between features.
This model can assist in optimizing wine production processes and improving quality control.
