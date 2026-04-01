import os, numpy as np, joblib
from sklearn.neighbors import KNeighborsClassifier

DATA = "hybrid_data"
X, y, labels = [], [], []

for i, cls in enumerate(sorted(os.listdir(DATA))):
    cls_dir = os.path.join(DATA, cls)
    if not os.path.isdir(cls_dir):
        continue
    labels.append(cls)
    for f in os.listdir(cls_dir):
        X.append(np.load(os.path.join(cls_dir, f)))
        y.append(i)

X = np.array(X)
y = np.array(y)

knn = KNeighborsClassifier(n_neighbors=5, weights="distance")
knn.fit(X, y)

joblib.dump(knn, "hybrid_model.pkl")
print("Hybrid KNN trained")
