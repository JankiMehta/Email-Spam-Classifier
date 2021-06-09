from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from data_preprocessor import Preprocessor
from feature_generator import FeatureGenerator
from ml_engine import MLengine

path = 'data/Spam Email raw text for NLP.csv'
feature_filename = 'tfidf_vectorizer.sav'
classifier_filename = 'complementNB_classifier.sav'

# read and clean data
print("Cleaning data...")
prep = Preprocessor(path)
dataset = prep.clean_data()
messages = prep.clean_tokens()
y = dataset['CATEGORY'].to_list()

# train and test split
msg_train, msg_test, y_train, y_test = train_test_split(messages, y, test_size=0.2, random_state=0)

# Generate features for training set
fg = FeatureGenerator(msg_train, 'train', feature_filename)
print("Generating features...")
X_train = fg.generate_tfidf_vector()

# Train and validate
ml_engine = MLengine(X_train, y_train, 'train', classifier_filename)
# 5-fold cross validation
score = ml_engine.validate()
print("Validation Score: ")
print("Ham marked as ham: ", score['test_tn'])
print("Ham marked as spam: ", score['test_fp'])
print("Spam marked as ham: ", score['test_fn'])
print("Spam marked as spam: ", score['test_tp'])

# Train and save model
print("Training classifier...")
ml_engine.train()

# Generate features for test set using already trained vectorizer
print("Testing...")
fg = FeatureGenerator(msg_test, 'test', feature_filename)
X_test = fg.convert_to_tfidf()

# Predict
ml_engine = MLengine(X_test, y_test, 'test', classifier_filename)
y_pred = ml_engine.predict()

# Evaluate performance on test set
cnf_mat = confusion_matrix(y_test, y_pred)
print("\nTest Score: ")
print("Ham marked as ham: ", cnf_mat[0, 0])
print("Ham marked as spam: ", cnf_mat[0, 1])
print("Spam marked as ham: ", cnf_mat[1, 0])
print("Spam marked as spam: ", cnf_mat[1, 1])
