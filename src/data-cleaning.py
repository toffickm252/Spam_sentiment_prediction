import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Download stopwords
nltk.download('stopwords', quiet=True)

# --- 1. GLOBAL CONFIGURATION & OPTIMIZATIONS ---

# Define Contractions Dictionary
contractions = {
    "can't": "cannot", "won't": "will not", "n't": " not",
    "'re": " are", "'s": " is", "'ll": " will", "'ve": " have",
    "'m": " am", "gonna": "going to", "wanna": "want to"
}

# Compile Regex Pattern for Contractions (Efficient one-pass replacement)
contractions_pattern = re.compile('|'.join(map(re.escape, contractions.keys())))

# Define Slang Dictionary
slang = {
    "brb": "be right back", "lol": "laughing out loud",
    "idk": "i do not know", "btw": "by the way",
    "fyi": "for your information", "omg": "oh my god"
}

# Define Stopwords globally
stop_words = set(stopwords.words('english'))


# --- 2. DATA LOADING & SETUP ---

# Use relative paths for portability
base_dir = os.getcwd()
data_path = os.path.join(base_dir, 'data', 'Enron.csv')
output_path = os.path.join(base_dir, 'data', 'Full_and_Cleaned_Enron.csv')
figure_path = os.path.join(base_dir, 'figures', 'balanced_distribution.png')

# Handle potential encoding errors
try:
    email_df = pd.read_csv(data_path, encoding='latin-1') 
except FileNotFoundError:
    print(f"Error: File not found at {data_path}. Check your working directory.")
    exit()


# --- 3. DATA BALANCING ---

ham_msg = email_df[email_df['label'] == 'ham']
spam_msg = email_df[email_df['label'] == 'spam']

# Safer sampling: use the minimum length of either class
min_sample = min(len(ham_msg), len(spam_msg))
ham_msg_balanced = ham_msg.sample(n=min_sample, random_state=42)
spam_msg_balanced = spam_msg.sample(n=min_sample, random_state=42)

balanced_data = pd.concat([ham_msg_balanced, spam_msg_balanced]).reset_index(drop=True)

# Visualize
sns.countplot(x='label', data=balanced_data)
plt.title("Balanced Distribution")
os.makedirs(os.path.dirname(figure_path), exist_ok=True) # Ensure folder exists
plt.savefig(figure_path)
plt.close()


# --- 4. FEATURE EXTRACTION ---

def extract_date(text):
    if not isinstance(text, str): return None
    # Month name format
    match = re.search(r'(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2}\s*,?\s*\d{4}', text, re.IGNORECASE)
    if match: return match.group(0)
    # Numeric format
    match = re.search(r'\d{1,2}/\d{1,2}/\d{2,4}', text)
    if match: return match.group(0)
    return None

balanced_data['final_date'] = balanced_data['subject'].apply(extract_date).combine_first(balanced_data['body'].apply(extract_date))
balanced_data['full_email'] = balanced_data['subject'].fillna('') + ' ' + balanced_data['body'].fillna('')


# --- 5. TEXT CLEANING ---

def clean_text(text):
    """Comprehensive text cleaning using global patterns."""
    
    # 1. Lowercase
    text = text.lower()
    
    # 2. Expand Contractions (Using pre-compiled global pattern)
    text = contractions_pattern.sub(lambda match: contractions[match.group(0)], text)

    # 3. Remove HTML, URLs, Hashtags
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'#\S+', '', text)

    # 4. Remove special characters (punctuation)
    # Note: Done AFTER contractions so we don't break "can't" -> "cant"
    text = re.sub(r'[^A-Za-z\s]', '', text)
    
    # 5. Handle Slang (Using global dictionary)
    words = text.split()
    text = ' '.join([slang.get(word, word) for word in words])
    
    # 6. Standardize repeated characters
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    
    # 7. Remove stopwords (Using global set)
    words = [word for word in text.split() if word not in stop_words]
    text = ' '.join(words)
    
    # 8. Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

print("Cleaning text... (this may take a moment)")
balanced_data['cleaned_email'] = balanced_data['full_email'].apply(clean_text)


# --- 6. SAVE RESULTS ---

balanced_data = balanced_data[['final_date', 'cleaned_email', 'label']]


balanced_data.to_csv(output_path, index=False)
print(f"Done! Cleaned data saved to: {output_path}")    
