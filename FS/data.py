import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score, make_scorer, confusion_matrix, ConfusionMatrixDisplay

data = pd.read_csv("sms.tsv", sep="\t", names=["class", "text"], header=0)

data = data[data['text'].notnull()]
data = data[data['text'].str.strip() != '']

data['class'] = data['class'].map({'ham': 0, 'spam': 1})

X = data['text']
y = data['class']

vectorizer = CountVectorizer(stop_words=None)
X_vectorized = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)

model = MultinomialNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
f1 = f1_score(y_test, y_pred)
print(f"F1-мера на тестовых данных: {f1:.2f}")

scorer = make_scorer(f1_score)
f1_cv = cross_val_score(model, X_vectorized, y, cv=5, scoring=scorer)
print(f"Средняя F1-мера по кросс-валидации: {f1_cv.mean():.2f}")

print(data['class'].value_counts())

cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["ham", "spam"])
disp.plot()
plt.show()
