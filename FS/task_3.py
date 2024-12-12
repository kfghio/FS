import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

data = pd.read_csv("sms.tsv", sep="\t", names=["class", "text"], header=0)
data['class'] = data['class'].map({'ham': 0, 'spam': 1})

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(data['text'])
y = data['class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
filter_method = SelectKBest(score_func=chi2, k=30)
X_filter_selected = filter_method.fit_transform(X, y)

classifiers = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "RandomForest": RandomForestClassifier(random_state=42),
    "KNN": KNeighborsClassifier()
}

for name, clf in classifiers.items():
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(f"{name} до выбора признаков: {accuracy_score(y_test, y_pred):.2f}")

for name, clf in classifiers.items():
    clf.fit(X_filter_selected[:len(y_train)], y_train)
    y_pred = clf.predict(X_filter_selected[len(y_train):])
    print(f"{name} после фильтрующего метода: {accuracy_score(y_test, y_pred):.2f}")
