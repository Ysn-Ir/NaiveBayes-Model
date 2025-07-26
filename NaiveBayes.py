import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import re
from imblearn.over_sampling import SMOTE

class NaiveBayes:
    def __init__(self):
        self.phi_y = None
        self.phi_x_0 = None
        self.phi_x_1 = None
        self.feature_names = None

    def fit(self, x, y, feature_names=None):
        X = np.array(x)
        Y = np.array(y)
        self.phi_y = np.mean(Y)
        self.feature_names = feature_names

        X_y1 = X[Y == 1]
        X_y0 = X[Y == 0]

        # Multinomial: Use counts with Laplace smoothing
        total_words_y1 = np.sum(X_y1) + X.shape[1]  # Total words + vocab size
        total_words_y0 = np.sum(X_y0) + X.shape[1]
        self.phi_x_1 = (np.sum(X_y1, axis=0) + 1) / total_words_y1
        self.phi_x_0 = (np.sum(X_y0, axis=0) + 1) / total_words_y0

        eps = 1e-9
        self.phi_x_1 = np.clip(self.phi_x_1, eps, 1 - eps)
        self.phi_x_0 = np.clip(self.phi_x_0, eps, 1 - eps)

    def predict_prob(self, x):
        x = np.array(x)
        log_phi = np.log(self.phi_y)
        log_phi_opp = np.log(1 - self.phi_y)

        # Multinomial: Use word counts
        log_phi_x_1 = np.sum(x * np.log(self.phi_x_1), axis=1)
        log_phi_x_0 = np.sum(x * np.log(self.phi_x_0), axis=1)

        log_p1 = log_phi_x_1 + log_phi
        log_p0 = log_phi_x_0 + log_phi_opp

        log_max = np.maximum(log_p1, log_p0)
        log_p1 = log_p1 - log_max
        log_p0 = log_p0 - log_max

        p1 = np.exp(log_p1)
        p0 = np.exp(log_p0)

        prob_y1 = p1 / (p1 + p0)
        return prob_y1

    def predict(self, x):
        prob = self.predict_prob(x)
        return (prob >= 0.5).astype(int)

    def debug_message(self, x, message):
        x = np.array(x)
        # Ensure x is 2D
        if x.ndim == 1:
            x = x.reshape(1, -1)
        
        log_phi = np.log(self.phi_y)
        log_phi_opp = np.log(1 - self.phi_y)
        
        log_phi_x_1 = x * np.log(self.phi_x_1) + (1 - x) * np.log(1 - self.phi_x_1)
        log_phi_x_0 = x * np.log(self.phi_x_0) + (1 - x) * np.log(1 - self.phi_x_0)
        
        print(f"\nDebugging message: {message}")
        print(f"Prior log prob (spam): {log_phi:.4f}, (ham): {log_phi_opp:.4f}")
        print("Top contributing features:")
        if self.feature_names is not None:
            indices = np.where(x[0] > 0)[0]  # Get indices for the first (and only) sample
            if len(indices) == 0:
                print("No features present in the message.")
            for idx in indices:
                word = self.feature_names[idx]
                print(f"Word: {word}, log P(word|spam): {log_phi_x_1[0, idx]:.4f}, log P(word|ham): {log_phi_x_0[0, idx]:.4f}")

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text

# Load data
try:
    df = pd.read_csv("spam.csv", encoding="latin1", usecols=[0, 1])
    df.columns = ['label', 'message']
except FileNotFoundError:
    print("Error: spam.csv not found. Please provide the dataset.")
    exit(1)

# Check class distribution
print("Class distribution:")
print(df['label'].value_counts())

# Clean messages
df['cleaned_message'] = df['message'].apply(clean_text)

# Convert labels
df['label'] = df['label'].map({'ham': 0, 'spam': 1}).astype(int)

# Vectorize cleaned text
vectorizer = CountVectorizer(binary=False, max_features=5000)
X = vectorizer.fit_transform(df['cleaned_message']).toarray()
y = df['label'].values
feature_names = vectorizer.get_feature_names_out()

# Check key words in vocabulary
key_words = ['free', 'prize', 'win', 'text', '12345']
print("\nKey words in vocabulary:", [word for word in feature_names if word in key_words])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Train model
nb = NaiveBayes()
nb.fit(X_train_resampled, y_train_resampled, feature_names=feature_names)

# Evaluate
preds = nb.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, preds, target_names=['Ham', 'Spam']))

# Test messages
test_msgs = [
    "Free prize! Text WIN to 12345 now!",
    "Congratulations! You've won $1000!",
    "Meeting at 3 PM tomorrow?",
    "Claim your free gift now!",
    "check this free iphone",
    "free money now",
    "i have a great offer for you and you only , register now ,congrats"
]
test_cleaned = [clean_text(m) for m in test_msgs]
x_test = vectorizer.transform(test_cleaned).toarray()
probs = nb.predict_prob(x_test)
preds = nb.predict(x_test)

for i, (msg, prob, pred) in enumerate(zip(test_msgs, probs, preds)):
    print(f"\nMessage: {msg}")
    print(f"Spam probability: {prob:.4f}")
    print(f"Prediction: {'Spam' if pred == 1 else 'Ham'}")
    nb.debug_message(x_test[i:i+1], msg)