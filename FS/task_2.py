import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import SelectKBest, chi2, SequentialFeatureSelector, RFE, SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from task_1 import feature_names
data = pd.read_csv("sms.tsv", sep="\t", names=["class", "text"], header=0)
data['class'] = data['class'].map({'ham': 0, 'spam': 1})

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(data['text'])
y = data['class']

filter_method = SelectKBest(score_func=chi2, k=30)
X_filter_selected = filter_method.fit_transform(X, y)
filter_features = [feature_names[i] for i in filter_method.get_support(indices=True)]
print('Фильтрующий метод completed')
print("Фильтрующий метод:", filter_features)

embedded_model = LogisticRegression(max_iter=1000)
embedded_model.fit(X, y)
coefficients = embedded_model.coef_[0]
embedded_features = [feature_names[i] for i in coefficients.argsort()[-30:]]
print('Встроенный метод completed')
print("Встроенный метод:", embedded_features)

wrapper_model = RandomForestClassifier(n_estimators=100, random_state=42)
wrapper_model.fit(X, y)
feature_importances = wrapper_model.feature_importances_
top_30_indices = np.argsort(feature_importances)[-30:]
wrapper_features = [feature_names[i] for i in top_30_indices]
print("Метод-обёртка completed")
print("Метод-обёртка:", wrapper_features)

