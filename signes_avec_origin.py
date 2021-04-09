from utils_avec_origin import *
from sklearn.neighbors import KNeighborsClassifier

def initialize_knn(x_train, y_train, n_neighbors=5):
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    return knn

def train_knn(knn, x_train, y_train):
    x_train_normalized = normalize_data(x_train)
    knn.fit(x_train_normalized, y_train)

def predictions_knn(knn, x_test):
    x_test_normalized = normalize_data(x_test)
    predictions = knn.predict(x_test_normalized)
    return predictions
