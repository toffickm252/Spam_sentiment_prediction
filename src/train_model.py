# import libaries 
import pandas as pd
import os
import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC



# --- 2. DATA LOADING & SETUP ---
base_dir = os.getcwd()
data_path = os.path.join(base_dir, 'data', 'Cleaned_Enron_new.csv')
output_path = os.path.join(base_dir, 'models', 'spam_classifier_model.joblib')

# Load the balanced cleaned data
# balanced_data = pd.read_csv(data_path)

# Load the balanced cleaned data
balanced_data = pd.read_csv(data_path)

# FIX: remove NaN or empty text
balanced_data = balanced_data.dropna(subset=['cleaned_email'])
balanced_data = balanced_data[balanced_data['cleaned_email'].str.strip() != ""]

# Train test split
from sklearn.model_selection import train_test_split
X = balanced_data['cleaned_email']
y = balanced_data['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Vectorization
from sklearn.feature_extraction.text import TfidfVectorizer

# 1. Create vectorizer
tfidf = TfidfVectorizer(max_features=5000)

# 2. Fit ONLY on training data
tfidf.fit(X_train)

joblib.dump(tfidf, 'models/tfidf_vectorizer.joblib')
print("Vectorizer saved!")

# 3. Transform BOTH train and test
X_train_tfidf = tfidf.transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# The data is now vectorized, so we can proceed with these features.
# The following lines are commented out as StandardScaler is not meant for sparse text data.
# If you have other numerical features, they should be scaled and combined here.

# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()
# scaler.fit(X_train)  # Fit on train
# X_train_scaled = scaler.transform(X_train)  # Transform train
# X_test_scaled = scaler.transform(X_test)    # Transform test

# from scipy.sparse import csr_matrix
# X_train_num_sparse = csr_matrix(X_train_scaled)
# X_test_num_sparse = csr_matrix(X_test_scaled)

# from scipy.sparse import hstack
# X_train_combined = hstack([X_train_tfidf, X_train_num_sparse])
# X_test_combined = hstack([X_test_tfidf, X_test_num_sparse])

# --- 3. MODEL TRAINING ---
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns

# After scaling, train multiple models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Linear SVM': CalibratedClassifierCV(LinearSVC(), cv=3),
    'Naive Bayes': MultinomialNB()
}

results = {}

for name, model in models.items():
    print(f"\n{'='*50}")
    print(f"Training {name}...")
    print('='*50)
    
    # Train
    model.fit(X_train_tfidf, y_train)
    
    # Predict
    y_pred = model.predict(X_test_tfidf)
    y_pred_proba = model.predict_proba(X_test_tfidf)[:, 1] if hasattr(model, 'predict_proba') else None
    
    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nAccuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Non-Spam', 'Spam']))
    
    # Store results
    results[name] = {
        'accuracy': accuracy,
        'predictions': y_pred,
        'probabilities': y_pred_proba
    }

# Compare models
print("\n" + "="*50)
print("MODEL COMPARISON")
print("="*50)
for name, result in results.items():
    print(f"{name}: {result['accuracy']:.4f}")


# Save the best model (based on accuracy)
best_model_name = max(results, key=lambda x: results[x]['accuracy'])
best_model = models[best_model_name]
print(f"\nBest model: {best_model_name} with accuracy {results[best_model_name]['accuracy']:.4f}")

# Save the model and vectorizer
# --- SAVE BEST MODEL + VECTORIZER ---
os.makedirs(os.path.dirname(output_path), exist_ok=True)
joblib.dump(best_model, output_path)
print(f"Best model saved to {output_path}")