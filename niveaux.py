from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

'''def initialize_decision_tree_classifier():
    tree = DecisionTreeClassifier(random_state=1)
    return tree

def train_decision_tree_classifier(tree, x_train, y_train):
    tree.fit(x_train, y_train)

def predictions_decision_tree_classifier(tree, x_test):
    predictions = tree.predict(x_test)
    return predictions'''


def initialize_knn_niveau(x_train, y_train, n_neighbors=3):
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    return knn

def train_knn_niveau(knn, x_train, y_train):
    knn.fit(x_train, y_train)

def predictions_knn_niveau(knn, x_test):
    predictions = knn.predict(x_test)
    return predictions

