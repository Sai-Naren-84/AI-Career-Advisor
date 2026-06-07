import pandas as pd
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = pd.read_csv("dataset.csv")

print(data["role"].value_counts())

vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(data["skills"])

y = data["role"]

model = LogisticRegression()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("Actual:", list(y_test))
print("Predicted:", list(predictions))
print("Accuracy:", accuracy)

model.fit(X, y)

pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

skills = input("Enter skills: ")

new_resume = [skills]

new_resume_vector = vectorizer.transform(new_resume)

prediction = model.predict(new_resume_vector)

print("Predicted Role:", prediction[0])