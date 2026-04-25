# DSAA2011_project
This repository contains the group project for **DSAA2011 (L01)** at HKUST(GZ).  
We apply machine learning techniques to the **Student Dropout Dataset** to explore, cluster, and predict student academic outcomes.

## Dataset

**Student Dropout Dataset** – 4,424 instances of students enrolled in different undergraduate degrees.  
Features include information known at enrollment (academic path, demographics, socio-economic factors) and academic performance at the end of the first and second semesters.  
The prediction target is the student’s outcome (e.g., dropout, enrolled, graduate).

## Project Objectives

Gain hands-on experience with a real-world machine learning workflow:
- Data preprocessing and exploratory analysis
- Dimensionality reduction and visualization
- Unsupervised clustering
- Supervised classification and model evaluation
- Open-ended exploration and improvement

## Mandatory Tasks

### 1. Data Preprocessing
- Handle missing values (imputation / indicator)
- Encode non-numeric features (one-hot encoding, boolean indicators)
- Further processing: standardisation, train/test split

### 2. Data Visualization (t-SNE)
- Project high-dimensional data into 2D/3D using t-SNE
- Colour points by class labels (dropout / enrolled / graduate)
- Identify patterns or clusters in the embedding

### 3. Clustering Analysis
- Apply **at least two** clustering algorithms (e.g., K-Means, Hierarchical Clustering)
- Evaluate clustering with multiple metrics
- Visualize clusters and compare algorithm performance
- Justify the best clustering result

### 4. Prediction: Model Training & Testing
- Define classification target (student outcome)
- Choose **at least two** simple model classes (e.g., Decision Tree, Logistic Regression)
- Split data (e.g., 70% train / 30% test)
- Train models and evaluate on train, test, and entire dataset
- Generate confusion matrices and visualizations

### 5. Model Evaluation & Improvement
- Compute accuracy, precision, recall, F1-score
- Plot ROC curves and calculate AUC for each model
- Improve models through validation
- Interpret strengths, weaknesses, and overfitting

## Open-ended Exploration

Further analysis is encouraged. Possible directions:
- **Model improvement** – polynomial features, regularisation (Ridge/Lasso)
- **Model comparison** – compare ≥3 models (e.g., SVM, Random Forest, Neural Networks) with cross-validation
- **Feature engineering/selection** – create new features or select by importance
- **Hyperparameter tuning** – grid search or random search
