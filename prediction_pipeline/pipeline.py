# import os

# # This tells us where this script is running from
# current_location = os.getcwd()
# print(f"Currently running from: {current_location}\n")

# # Let's check if the models folder exists
# models_folder = os.path.join(current_location, 'models')
# print(f"Looking for models folder at: {models_folder}")
# print(f"Does it exist? {os.path.exists(models_folder)}\n")

# # If the folder exists, let's see what's inside
# if os.path.exists(models_folder):
#     print("Files inside models folder:")
#     files = os.listdir(models_folder)
#     if files:
#         for file in files:
#             print(f"  - {file}")
#     else:
#         print("  (folder is empty)")
# else:
#     print("The models folder doesn't exist yet!")
#     print("This means you haven't saved your model yet.")

# # Let's also check your project structure
# print("\n" + "="*50)
# print("Your project structure:")
# print("="*50)
# for item in os.listdir(current_location):
#     item_path = os.path.join(current_location, item)
#     if os.path.isdir(item_path):
#         print(f"📁 {item}/")
#     else:
#         print(f"📄 {item}")

import joblib
import os

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Go up one level to project root, then into models folder
project_root = os.path.dirname(script_dir)

# Use the ACTUAL filenames that exist in your models folder
model_path = os.path.join(project_root, 'models', 'spam_classifier_model.joblib')
vectorizer_path = os.path.join(project_root, 'models', 'tfidf_vectorizer.joblib')

print("Loading model and vectorizer...")
print(f"Model path: {model_path}")
print(f"Vectorizer path: {vectorizer_path}\n")

# Load the trained model and vectorizer
model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)
print("Successfully loaded!\n")

# New email to predict
new_email = ["Congratulations! You've won a $1,000 Walmart gift card. Click here to claim your prize."]
print(f"Testing with email: {new_email[0]}\n")

# Convert the new email to numbers using the vectorizer
new_email_vec = vectorizer.transform(new_email)

# Make the prediction
# This returns 0 for not spam, 1 for spam
prediction = model.predict(new_email_vec)[0]

# Get the confidence scores
# This returns two numbers: [confidence_not_spam, confidence_spam]
probabilities = model.predict_proba(new_email_vec)[0]

# Display the results in a readable way
print("="*50)
print("PREDICTION RESULTS")
print("="*50)
print(f"Prediction: {'SPAM ⚠️' if prediction == 1 else 'NOT SPAM ✓'}")
print(f"Confidence in NOT SPAM: {probabilities[0] * 100:.2f}%")
print(f"Confidence in SPAM: {probabilities[1] * 100:.2f}%")