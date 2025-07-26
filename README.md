
# 📧 Naive Bayes Spam Classifier

This project implements a **Naive Bayes classifier from scratch** in Python to detect spam messages using the classic SMS Spam Collection Dataset. It includes:

* Preprocessing of SMS text
* Feature extraction with `CountVectorizer`
* Oversampling using `SMOTE`
* Training & evaluation of a custom `NaiveBayes` class
* Debugging utilities to inspect prediction behavior

---

## 📁 Dataset

The project expects a CSV file named `spam.csv` with the following structure:

| label | message                            |
| ----- | ---------------------------------- |
| ham   | Hello, how are you?                |
| spam  | Free prize! Text WIN to 12345 now! |

You can download the dataset from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection).

---

## 📦 Dependencies

Install the required packages via:

```bash
pip install numpy pandas scikit-learn imbalanced-learn
```

---

## 🚀 How It Works

1. **Text Preprocessing**

   * Lowercasing
   * Removing punctuation/special characters using regex

2. **Vectorization**

   * `CountVectorizer` converts cleaned text into feature vectors (bag-of-words model)

3. **SMOTE Oversampling**

   * Balances dataset by generating synthetic samples for the minority class (spam)

4. **Naive Bayes Training**

   * Custom `NaiveBayes` class computes conditional probabilities using multinomial distribution and Laplace smoothing

5. **Evaluation**

   * Outputs a classification report (precision, recall, F1-score)
   * Tests sample messages and prints detailed debug info

---

## 🧠 Example Output

```text
Class distribution:
ham     4825
spam     747

Key words in vocabulary: ['free', 'prize', 'win', 'text', '12345']

Classification Report:
              precision    recall  f1-score   support

         Ham       0.97      0.99      0.98       965
        Spam       0.93      0.82      0.87       150

    accuracy                           0.96      1115
   macro avg       0.95      0.90      0.93      1115
weighted avg       0.96      0.96      0.96      1115
```

---

## 🛠 Debugging Example

```text
Message: Free prize! Text WIN to 12345 now!
Spam probability: 0.9991
Prediction: Spam

Debugging message: Free prize! Text WIN to 12345 now!
Prior log prob (spam): -0.8210, (ham): -0.1973
Top contributing features:
Word: free, log P(word|spam): -3.6512, log P(word|ham): -6.4718
Word: prize, log P(word|spam): -5.0748, log P(word|ham): -7.2087
...
```

---

## 🧪 Test It Yourself

You can customize and add your own messages to test against the trained model. These are defined near the bottom of the script:

```python
test_msgs = [
    "Free prize! Text WIN to 12345 now!",
    "Congratulations! You've won $1000!",
    "Meeting at 3 PM tomorrow?",
    ...
]
```

---

## 📌 Notes

* The model uses **Multinomial Naive Bayes** for count-based features.
* Uses **Laplace smoothing** to handle zero probabilities.
* Outputs spam probabilities and debug info for interpretability.

---

## 📄 License

This project is open-source and free to use under the [MIT License](https://opensource.org/licenses/MIT).



