# import libraries 
import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- 2. DATA LOADING & SETUP ---

# Use relative paths for portability
base_dir = os.getcwd()
data_path = os.path.join(base_dir, 'data', 'Enron.csv')
output_path = os.path.join(base_dir, 'data', 'Cleaned_Enron_new.csv')
figure_path = os.path.join(base_dir, 'figures', 'balanced_distribution_new.png')

# Handle potential encoding errors
try:
    email_df = pd.read_csv(data_path) # read the data from from the path
except FileNotFoundError:
    print(f"Error: File not found at {data_path}. Check your working directory.")
    exit()

# Data investigation 
print(f"Total rows: {len(email_df)}") # print total rows
# print(f"\nColumn names: {email_df.columns.tolist()}") # print column names
# print(f"\nLabel value counts:\n{email_df['label'].value_counts()}") # print label value counts
print(email_df.head(3)) # print first few rows 
print("\n")
print(email_df.info()) # print data info
print("\n")
print(email_df['label'].value_counts()) # print label value counts

# --- 3. DATA BALANCING ---
ham_msg = email_df[email_df['label'] == 0]
spam_msg = email_df[email_df['label'] == 1]

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
    if not isinstance(text, str): 
        return None
    # Month name format
    match = re.search(r'(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2}\s*,?\s*\d{4}', text, re.IGNORECASE)
    if match: 
        return match.group(0)
    # Numeric format
    match = re.search(r'\d{1,2}/\d{1,2}/\d{2,4}', text)
    if match: 
        return match.group(0)
    return None

# Extract dates from both columns separately
balanced_data['subject_date'] = balanced_data['subject'].apply(extract_date)
balanced_data['body_date'] = balanced_data['body'].apply(extract_date)

# Combine with subject taking precedence (use fillna, not combine_first)
balanced_data['final_date'] = balanced_data['subject_date'].fillna(balanced_data['body_date'])

# combine subject and body into full email text
balanced_data['full_email'] = balanced_data['subject'].fillna('') + ' '+ balanced_data['body'].fillna('')

print('Columns after feature extraction:', balanced_data.columns.tolist())
print("\n")
print(balanced_data[['subject_date', 'body_date', 'final_date']].head(10)) #
print("\n")
print(balanced_data[['subject_date', 'body_date', 'final_date']].isnull().sum()) #

# drop subject and body date columns
balanced_data.drop(columns=['subject_date', 'body_date','body','subject'], inplace=True)
print("\n")
# columns after dropping
print('Columns after dropping unnecessary columns:', balanced_data.columns.tolist())


# --- 5. TEXT CLEANING ---
nltk.download('stopwords')
# define stopwords
stop_words = set(stopwords.words('english'))

# Define Contractions Dictionary
contractions = {
    "can't": "cannot", "won't": "will not", "n't": " not",
    "'re": " are", "'s": " is", "'ll": " will", "'ve": " have",
    "'m": " am", "gonna": "going to", "wanna": "want to", "gotta": "got to"
    , "im": "i am", "dont": "do not", "doesnt": "does not", "didnt": "did not",
    "isnt": "is not", "shouldnt": "should not", "couldnt": "could not"
}

# Compile Regex Pattern for Contractions (Efficient one-pass replacement)
contractions_pattern = re.compile('|'.join(map(re.escape, contractions.keys())))

# Define Slang Dictionary
slang = {
    "brb": "be right back", "lol": "laughing out loud",
    "idk": "i do not know", "btw": "by the way",
    "fyi": "for your information", "omg": "oh my god", "u": "you",
    "ur": "your", "thx": "thanks", "pls": "please", "plz": "please"
}

# Text cleaning function
def clean_text(text):
    if not isinstance(text, str):
        return ""
    
    # Lowercase
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

# investigate cleaned text
print(balanced_data['cleaned_email'].head(3))
print("\n")
print(balanced_data.info())
print("\n")
print(balanced_data['cleaned_email'].isnull().sum())
print("\n")
print(balanced_data.columns.tolist())

# --- 6. SAVE RESULTS ---
balanced_data.to_csv(output_path, index=False)
print(f"Done! Cleaned data saved to: {output_path}")





