import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.pipeline import make_pipeline

# Load the dataset
dataset = pd.read_csv('textual_malaria_patient_data_10000.csv')  # Replace with your file path

# Combine symptom columns into a single text column
dataset['Symptoms_Text'] = dataset[['High Temperature', 'Sweats', 'Chills', 'Headaches', 'Confusion', 'Very Tired', 'Sleepy', 'Tummy Pain', 'Diarrhoea', 'Loss of Appetite', 'Muscle Pains', 'Yellow Skin', 'White of Eyes', 'Sore Throat', 'Cough', 'Difficulty Breathing']].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)  # List all symptom columns

# Split the dataset into features (X) and target label (y)
X = dataset['Symptoms_Text']
y = dataset['Diagnosis']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a pipeline with TfidfVectorizer and MultinomialNB
pipeline = make_pipeline(TfidfVectorizer(), MultinomialNB())

# Train the model
pipeline.fit(X_train, y_train)

# Predict on the test set
predictions = pipeline.predict(X_test)

# Evaluate the model
report = classification_report(y_test, predictions)
print(report)
