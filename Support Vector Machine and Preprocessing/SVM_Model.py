# Step 1: Prepare the dataset
import pandas as pd
tweets_df = pd.read_csv('tweet_emotions.csv')
# Clean and preprocess the text data here

# Step 2: Convert the text data into a numerical format
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer()
X = tfidf.fit_transform(tweets_df['content'])
y = tweets_df['sentiment']

# Step 3: Split the dataset into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Step 4: Train the SVM model
from sklearn.svm import SVC
svm_classifier = SVC(kernel='linear', C=1, gamma='auto')
svm_classifier.fit(X_train, y_train)

# Step 5: Evaluate the performance of the trained model
y_pred = svm_classifier.predict(X_test)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
print('Accuracy:', accuracy_score(y_test, y_pred))
print('Precision:', precision_score(y_test, y_pred, average='weighted'))
print('Recall:', recall_score(y_test, y_pred, average='weighted'))
print('F1 score:', f1_score(y_test, y_pred, average='weighted'))
