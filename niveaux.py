<<<<<<< HEAD
from sklearn.tree import DecisionTreeClassifier

def initialize_decision_tree_classifier():
    tree = DecisionTreeClassifier(random_state=1)
    return tree

def train_decision_tree_classifier(tree, x_train, y_train):
    tree.fit(x_train, y_train)

def predictions_decision_tree_classifier(tree, x_test):
    predictions = tree.predict(x_test)
    return predictions
=======
from sklearn.tree import DecisionTreeClassifier

def initialize_decision_tree_classifier():
    tree = DecisionTreeClassifier(random_state=1)
    return tree

def train_decision_tree_classifier(tree, x_train, y_train):
    tree.fit(x_train, y_train)

def predictions_decision_tree_classifier(tree, x_test):
    predictions = tree.predict(x_test)
    return predictions
>>>>>>> c9ff0ab8f8b2e63cddd478911dd1c90068c48ae4
