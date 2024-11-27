# Hotel-Review-Analysis-and-Prediction-A-Machine-Learning-Approach

# Introduction: 

The hospitality industry heavily depends on customer reviews to understand their experiences and improve services. This project leverages machine learning and NLP to analyze hotel reviews, identify sentiments, and predict churn behavior. With an emphasis on handling large datasets, the project utilizes distributed processing with PySpark and visualization libraries to uncover critical insights.

# Key Features: 

1. Data Preprocessing:

Null value handling, outlier detection, and removal.
Feature engineering, including review length and sentiment analysis.
2. Natural Language Processing (NLP):

Sentiment analysis using VADER and TextBlob.
Word clouds for positive and negative reviews.
Polarity and subjectivity analysis of customer reviews.
3. Data Analytics:

SQL-like querying on PySpark DataFrames.
Analysis of trends like review scores, hotel ratings, and trip types.
Machine Learning:

4. Classification models: Random Forest, Logistic Regression, XGBoost.
Handling class imbalance using SMOTE and undersampling techniques.
Predictive modeling for customer churn.
Visualization:

5. Seaborn and Matplotlib plots for review trends, sentiment distribution, and hotel rankings.

# Technologies Used: 

1. Programming: Python
2. Data Processing: Pandas, PySpark
3. Machine Learning: Scikit-learn, XGBoost
4. Visualization: Matplotlib, Seaborn, WordCloud
5. Natural Language Processing (NLP): VADER, TextBlob
6. SQL Integration: pandasql, SparkSQL
7. Model Evaluation: Classification metrics, ROC-AUC, Precision-Recall Curve

# Accuracy: 

1. Random Forest Classifier
Initial Model:

Accuracy: 97.97%
Observations:
The model performed exceptionally well in classifying the majority class (non-churners).
Precision and recall for the minority class (churners) were low due to severe class imbalance.
After Applying Class Weights:

Adjusting for imbalance using class weights improved the model's sensitivity towards the minority class.
Precision for churners: Increased slightly.
Recall for churners: Improved to 33%.
F1-Score for churners: Improved but still limited due to data imbalance.

2. XGBoost Classifier
Handling Class Imbalance with scale_pos_weight:

Accuracy: 84%
ROC-AUC Score: 0.876 (good discriminatory power between churners and non-churners).
Precision for churners: 9%.
Recall for churners: 74%.
This model focused more on capturing the minority class, even at the cost of overall accuracy.
After Adjusting Thresholds:

Optimizing for F1-score yielded a significant improvement.
F1-Score for churners: 0.29.
Precision and recall became more balanced
