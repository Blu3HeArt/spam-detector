from pathlib import Path
from collections import Counter
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix

data_path = Path(__file__).parent / "data" / "SMSSpamCollection"

labels = []
messages = []

with data_path.open(encoding="utf-8") as file:
    for line in file:
        label, message = line.rstrip("\n").split("\t", maxsplit=1)
        labels.append(label)
        messages.append(message)

print("Total messages:", len(messages))
print("Label counts:", Counter(labels))
from sklearn.model_selection import train_test_split

unique_messages = {}

for label, message in zip(labels, messages):
    if message in unique_messages and unique_messages[message] != label:
        raise ValueError("The same message has conflicting labels.")
    unique_messages[message] = label

clean_messages = list(unique_messages.keys())
clean_labels = list(unique_messages.values())

print("Duplicates removed:", len(messages) - len(clean_messages))

X_train, X_test, y_train, y_test = train_test_split(
    clean_messages,
    clean_labels,
    test_size=0.2,
    random_state=42,
    stratify=clean_labels,
)

print("Training messages:", len(X_train))
print("Test messages:", len(X_test))
print("Training labels:", Counter(y_train))
print("Test labels:", Counter(y_test))

model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("classifier", MultinomialNB()),
])

model.fit(X_train, y_train)

print("Training complete!")
print("Vocabulary size:", len(model.named_steps["tfidf"].vocabulary_))

predictions = model.predict(X_test)

print("\nClassification report:")
print(classification_report(y_test, predictions, digits=3, zero_division=0))

matrix = confusion_matrix(y_test, predictions, labels=["ham", "spam"])

print("Actual ham predicted as ham:", matrix[0, 0])
print("Actual ham predicted as spam:", matrix[0, 1])
print("Actual spam predicted as ham:", matrix[1, 0])
print("Actual spam predicted as spam:", matrix[1, 1])

custom_messages = [
    "Hey, are we still meeting for lunch tomorrow?",
    "Congratulations! You won a FREE cash prize. Call now to claim!",
    "Your verification code is 482910. Do not share it.",
]

custom_predictions = model.predict(custom_messages)

print("\nCustom message predictions:")

for message, prediction in zip(custom_messages, custom_predictions):
    print("Message:", message)
    print("Prediction:", prediction)
    print()
spam_variants = [
    "Congratulations! You won a FREE cash prize. Call now to claim!",
    "Congratulations! You won a FR33 c@sh pr1ze. Call now to claim!",
    "Hey, you were selected to receive some money. Give us a ring to collect it.",
]

variant_predictions = model.predict(spam_variants)

print("\nSpam wording experiment:")

for message, prediction in zip(spam_variants, variant_predictions):
    print("Message:", message)
    print("Prediction:", prediction)
    print()