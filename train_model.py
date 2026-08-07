import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# 1. Încărcarea datasetului

df = pd.read_csv("products.csv")

# Curățarea numelor coloanelor
df.columns = df.columns.str.strip()


# 2. Curățarea datelor

df["Product Title"] = (
    df["Product Title"]
    .fillna("")
    .astype(str)
    .str.lower()
    .str.strip()
)

df = df[
    (df["Product Title"] != "") &
    (df["Category Label"].notna())
].copy()

df = df.drop_duplicates().copy()


# 3. Pregătirea datelor

X = df["Product Title"]
y = df["Category Label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# 4. Modelul Logistic Regression

logistic_model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95
        )
    ),
    (
        "classifier",
        LogisticRegression(max_iter=1000)
    )
])

logistic_model.fit(X_train, y_train)

pred_lr = logistic_model.predict(X_test)


# 5. Modelul Multinomial Naive Bayes

nb_model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95
        )
    ),
    (
        "classifier",
        MultinomialNB()
    )
])

nb_model.fit(X_train, y_train)

pred_nb = nb_model.predict(X_test)


# 6. Evaluarea modelelor

def evaluate_model(name, y_true, predictions):

    accuracy = accuracy_score(y_true, predictions)

    precision = precision_score(
        y_true,
        predictions,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_true,
        predictions,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        predictions,
        average="weighted",
        zero_division=0
    )

    print(f"\n{name}")
    print("-" * len(name))
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

    return f1


f1_lr = evaluate_model(
    "Logistic Regression",
    y_test,
    pred_lr
)

f1_nb = evaluate_model(
    "Multinomial Naive Bayes",
    y_test,
    pred_nb
)


# 7. Alegerea modelului final

if f1_lr >= f1_nb:

    best_model = logistic_model
    best_model_name = "Logistic Regression"

else:

    best_model = nb_model
    best_model_name = "Multinomial Naive Bayes"


print("\n================================")
print("MODEL FINAL:", best_model_name)
print("================================")


# 8. Salvarea modelului

with open("product_classifier.pkl", "wb") as file:
    pickle.dump(best_model, file)

print("\nModelul a fost salvat:")
print("product_classifier.pkl")