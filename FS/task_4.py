import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split

data = pd.read_csv("sms.tsv", sep="\t", names=["class", "text"], header=0)
data['class'] = data['class'].map({'ham': 0, 'spam': 1})

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(data['text'])
y = data['class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
filter_method = SelectKBest(score_func=chi2, k=30)
X_filter_selected = filter_method.fit_transform(X, y)

kmeans = KMeans(n_clusters=2, random_state=42)
clusters_before = kmeans.fit_predict(X)
silhouette_before = silhouette_score(X, clusters_before)

clusters_after = kmeans.fit_predict(X_filter_selected)
silhouette_after = silhouette_score(X_filter_selected, clusters_after)

print(f"Silhouette до выбора признаков: {silhouette_before:.2f}")
print(f"Silhouette после выбора признаков: {silhouette_after:.2f}")
