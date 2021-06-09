import joblib
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import ComplementNB
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_validate


class MLengine:
    def __init__(self, x, y, mode, model_filename):
        self.BASE_PATH = "data/"
        self.filename = model_filename
        self.mode = mode

        if self.mode == 'train':
            self.X_train = x
            self.y_train = y
        else:
            self.X_test = x
            self.y_test = y
            self.model = self.load_model()

    def validate(self):
        classifier = ComplementNB()

        pr_scoring = ['precision', 'recall']

        def confusion_matrix_scorer(clf, X, y):
            y_pred = clf.predict(X)
            cm = confusion_matrix(y, y_pred)
            return {'tn': cm[0, 0], 'fp': cm[0, 1],
                    'fn': cm[1, 0], 'tp': cm[1, 1]}

        # 5-fold cross-validation used to decide classification algo and params
        scores = cross_validate(classifier, self.X_train.toarray(), self.y_train, scoring=confusion_matrix_scorer)
        return scores

    def train(self):
        classifier = ComplementNB()
        classifier.fit(self.X_train.toarray(), self.y_train)
        self.model = classifier
        self.save_model()
        return

    def predict(self):
        classifier = self.load_model()
        y_pred = classifier.predict(self.X_test)
        return y_pred

    def save_model(self):
        joblib.dump(self.model, self.BASE_PATH + self.filename)

    def load_model(self):
        return joblib.load(self.BASE_PATH + self.filename)
