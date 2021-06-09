import string
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


class Preprocessor:
    def __init__(self, path):
        self.path = path
        self.data = pd.read_csv(self.path)

    def clean_data(self):
        self.data['MESSAGE'] = self.data['MESSAGE'].str.lower()  # convert all test to lower
        self.data['MESSAGE'] = self.data['MESSAGE'].str.replace(r'http\S+', '')  # remove hyperlinks
        self.data['MESSAGE'] = self.data['MESSAGE'].str.replace(r'\d+', '')  # remove numbers
        self.data['MESSAGE'] = self.data['MESSAGE'].str.replace('\n', ' ')  # remove new line characters
        self.data['MESSAGE'] = self.data['MESSAGE'].str.translate(
            str.maketrans('', '', string.punctuation))  # remove punctuations

        return self.data

    def clean_tokens(self):
        lemmatizer = WordNetLemmatizer()
        messages = []
        for message in self.data['MESSAGE']:
            tokens = message.split()

            # remove stopwords
            tokens = [token for token in tokens if not token in set(stopwords.words('english'))]
            #  convert words to their base form (e.g. studied -> study)
            tokens = [lemmatizer.lemmatize(token) for token in tokens]

            clean_message = ' '.join(tokens)
            messages.append(clean_message)
        return messages
