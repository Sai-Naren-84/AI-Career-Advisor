import pickle

model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

skills = input("Enter skills: ")

resume_vector = vectorizer.transform([skills])

prediction = model.predict(resume_vector)

print("Predicted Role:", prediction[0])