import re
import pandas as pd
import nltk
from nltk.corpus import stopwords

# Download stopwords
nltk.download('stopwords', quiet=True)

# Load data
email_df = pd.read_csv('C:\\Users\\Surface\\OneDrive\\Documentos\\GitHub\\Spam_sentiment_prediction\\data\\Enron.csv')

# Explore data
print(email_df.head())
print(email_df.info())
print(f"Duplicates: {email_df.duplicated().sum()}")

# Extract dates from text
def extract_date(text):
    """Extract date from text in common formats."""
    if not isinstance(text, str):
        return None
    
    # Month name format: "May 25, 2001"
    match = re.search(r'(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2}\s*,?\s*\d{4}', 
                      text, re.IGNORECASE)
    if match:
        return match.group(0)
    
    # Numeric format: "05/29/2001"
    match = re.search(r'\d{1,2}/\d{1,2}/\d{2,4}', text)
    if match:
        return match.group(0)
    
    return None

# Extract dates
email_df['final_date'] = email_df['subject'].apply(extract_date).combine_first(
    email_df['body'].apply(extract_date))

# Combine subject and body
email_df['full_email'] = email_df['subject'].fillna('') + ' ' + email_df['body'].fillna('')

# Clean text function
def clean_text(text):
    """Comprehensive text cleaning."""
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove hashtags and special characters
    text = re.sub(r'#\S+', '', text)
    text = re.sub(r'[^A-Za-z\s]', '', text)
    
    # Convert to lowercase
    text = text.lower()
    
    # Expand common contractions
    contractions = {
        "can't": "cannot", "won't": "will not", "n't": " not",
        "'re": " are", "'s": " is", "'ll": " will", "'ve": " have",
        "'m": " am", "gonna": "going to", "wanna": "want to"
    }
    for contraction, expansion in contractions.items():
        text = text.replace(contraction, expansion)
    
    # Replace common slang
    slang = {
        "brb": "be right back", "lol": "laughing out loud",
        "idk": "i do not know", "btw": "by the way",
        "fyi": "for your information", "omg": "oh my god"
    }
    words = text.split()
    text = ' '.join([slang.get(word, word) for word in words])
    
    # Standardize repeated characters (e.g., "hellooo" -> "helloo")
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = [word for word in text.split() if word not in stop_words]
    text = ' '.join(words)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# Apply cleaning
email_df['cleaned_email'] = email_df['full_email'].apply(clean_text)

# Display results
print("\nCleaning sample:")
print(email_df[['full_email', 'cleaned_email']].head())

# Keep only necessary columns
email_df = email_df[['final_date', 'cleaned_email', 'label']]

# Save cleaned data
output_path = 'C:\\Users\\Surface\\OneDrive\\Documentos\\GitHub\\Spam_sentiment_prediction\\data\\Cleaned_Enron.csv'

try:
    email_df.to_csv(output_path, index=False)
    print(f"\nCleaned data saved to: {output_path}")
except PermissionError:
    # Try alternative filename if file is locked
    import os
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    alt_path = output_path.replace('.csv', f'_{timestamp}.csv')
    email_df.to_csv(alt_path, index=False)
    print(f"\nOriginal file was locked. Saved to: {alt_path}")
    print("Please close the file if it's open and rename it manually.")