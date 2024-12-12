import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

data = pd.read_csv("sms.tsv", sep="\t", names=["class", "text"], header=0)
data['class'] = data['class'].map({'ham': 0, 'spam': 1})

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(data['text'])
y = data['class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
filter_method = SelectKBest(score_func=chi2, k=30)
X_filter_selected = filter_method.fit_transform(X, y)

pca = PCA(n_components=2)
X_pca_before = pca.fit_transform(X.toarray())
X_pca_after = pca.fit_transform(X_filter_selected)


tsne = TSNE(n_components=2, random_state=42, init="random")
X_tsne_before = tsne.fit_transform(X.toarray())
X_tsne_after = tsne.fit_transform(X_filter_selected)


plt.figure(figsize=(12, 6))
#PCA
plt.subplot(1, 2, 1)
plt.scatter(X_pca_before[:, 0], X_pca_before[:, 1], c=y, cmap='viridis', alpha=0.5)
plt.title("PCA до выбора признаков")

plt.subplot(1, 2, 2)
plt.scatter(X_pca_after[:, 0], X_pca_after[:, 1], c=y, cmap='viridis', alpha=0.5)
plt.title("PCA после выбора признаков")

#t-SNE
#plt.subplot(1, 2, 1)
#plt.scatter(X_tsne_before[:, 0], X_pca_before[:, 1], c=y, cmap='viridis', alpha=0.5)
#plt.title("t-SNE до выбора признаков")

#plt.subplot(1, 2, 1)
#plt.scatter(X_tsne_after[:, 0], X_pca_after[:, 1], c=y, cmap='viridis', alpha=0.5)
#plt.title("t-SNE после выбора признаков")

plt.show()