from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import numpy as np

class BR_RandomForest:
    def __init__(self):
        base_classifier = RandomForestClassifier(n_estimators=365,random_state=42)
        self.classifier = MultiOutputClassifier(base_classifier)
        self.is_trained = False

    def train(self, X_train, Y_train):

        self.classifier.fit(X_train, Y_train)
        self.is_trained = True

    def test(self, X_test, proba=False):

        if not self.is_trained:
            raise ValueError("The model has not been trained yet. Call train() first.")

        if proba:
            y_pred = np.array([prob[:, 1] for prob in self.classifier.predict_proba(X_test)]).T
        else:
            y_pred = self.classifier.predict(X_test)

        return y_pred
class BR_SVM:
    def __init__(self):
        base_classifier = SVC(kernel='rbf')
        self.classifier = MultiOutputClassifier(base_classifier)
        self.is_trained = False

    def train(self, X_train, Y_train):

        self.classifier.fit(X_train, Y_train)
        self.is_trained = True

    def test(self, X_test, proba=False):

        if not self.is_trained:
            raise ValueError("The model has not been trained yet. Call train() first.")

        if proba:
            y_pred = np.array([prob[:, 1] for prob in self.classifier.predict_proba(X_test)]).T
        else:
            y_pred = self.classifier.predict(X_test)

        return y_pred
class BR_LogisticRegression:
    def __init__(self):

        base_classifier = LogisticRegression()
        self.classifier = MultiOutputClassifier(base_classifier)
        self.is_trained = False

    def train(self, X_train, Y_train):

        self.classifier.fit(X_train, Y_train)
        self.is_trained = True

    def test(self, X_test, proba=False):

        if not self.is_trained:
            raise ValueError("The model has not been trained yet. Call train() first.")

        if proba:
            y_pred = np.array([prob[:, 1] for prob in self.classifier.predict_proba(X_test)]).T
        else:
            y_pred = self.classifier.predict(X_test)

        return y_pred


