from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

data = pd.read_csv("sms.tsv", sep="\t", names=["class", "text"], header=0)
data['class'] = data['class'].map({'ham': 0, 'spam': 1})

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(data['text'])
y = data['class']

feature_names = vectorizer.get_feature_names_out()
