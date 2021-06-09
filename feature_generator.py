import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer


class FeatureGenerator:
    def __init__(self, clean_data, mode, feature_filename):
        self.data = clean_data
        self.BASE_PATH = "data/"
        self.filename = feature_filename
        if mode == 'test':
            self.vectorizer = self.load_model()

    def generate_count_vector(self):
        self.vectorizer = CountVectorizer()
        self.vectorizer.fit(self.data)

        self.save_model()
        return self.vectorizer.transform(self.data)

    def generate_tfidf_vector(self):
        self.vectorizer = TfidfVectorizer()
        self.vectorizer.fit(self.data)

        self.save_model()
        return self.vectorizer.transform(self.data)

    def convert_to_count_vector(self):
        return self.vectorizer.transform(self.data)

    def convert_to_tfidf(self):
        return self.vectorizer.transform(self.data)

    def save_model(self):
        joblib.dump(self.vectorizer, self.BASE_PATH + self.filename)

    def load_model(self):
        return joblib.load(self.BASE_PATH + self.filename)
