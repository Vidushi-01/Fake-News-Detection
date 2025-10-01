# Fake-News-Detection

Fake News Detection using Machine Learning Algorithms

This Project is to solve the problem with fake news. In this we have used two datasets named "Fake" and "True" from Kaggle. You can download the file from here https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset I have used five classifiers in this project the are Naive Bayes, Random Forest, Decision Tree, SVM, Logistic Regression.

#Report:
Fake news has become a significant problem in today’s digital world. The spread of misinformation through social media and online platforms can influence public opinion and create confusion. Machine Learning provides automated techniques to classify news articles as either ‘fake’ or ‘true’ based on their content. This report explores the process of building a fake news detection model using Python, Pandas, and Scikit-learn.
Dataset Description
We use the Fake and True News Dataset, which contains two files: Fake.csv – News articles identified as fake. True.csv – News articles identified as true. Each file has the following columns: title – Headline of the news article. text – The full content of the article. subject – Category of the article (politics, world, etc.). date – Date of publication. The dataset is balanced between fake and true news, allowing for effective training and evaluation of classification models.
Data Preprocessing
Before training the model, the following preprocessing steps are applied: Loading and combining Fake and True news datasets. Labeling fake news as 0 and true news as 1. Cleaning the text by removing punctuation, numbers, and stopwords. Tokenizing and normalizing the text (lowercasing, stemming/lemmatization). Converting text into numerical features using TF-IDF Vectorization.
Model Building
We implemented several machine learning models for fake news classification: Logistic
Regression – A simple baseline classifier that performs well on text classification tasks. Naive
Bayes – Suitable for word-frequency-based models, efficient for text classification. Random Forest
– An ensemble method that improves performance by combining multiple decision trees. Support Vector Machine (SVM) – Effective for high-dimensional spaces like TF-IDF vectors. The models are trained on 70% of the dataset and evaluated on 30% for testing.
Model Evaluation
The models are evaluated using the following metrics: Accuracy – Percentage of correctly classified news articles. Precision – Ability of the model to correctly identify true positives. Recall – Ability of the model to capture all relevant cases. F1 Score – Harmonic mean of precision and recall. Confusion Matrix – Distribution of true/false positives and negatives. Typically, Logistic Regression and SVM achieve accuracies above 95% on this dataset, making them effective solutions for fake news detection.
Conclusion
Fake news detection using machine learning is an essential application in the fight against misinformation. This project demonstrates how preprocessing, feature engineering, and machine learning models can be combined to build an automated fake news classifier. While traditional models achieve high accuracy, future work can explore deep learning approaches (e.g., LSTMs, Transformers) for improved generalization on diverse datasets.



