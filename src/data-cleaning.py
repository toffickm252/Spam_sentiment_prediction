# import libraries
import re
import pandas as pd
import numpy as np
import html
import json
import itertools
import string
import nltk
from nltk.corpus import stopwords

# Download stopwords once
nltk.download('stopwords', quiet=True)

email_df = pd.read_csv('C:\\Users\\Surface\\OneDrive\\Documentos\\GitHub\\Spam_sentiment_prediction\\data\\Enron.csv')

# small investigation of the data
print(email_df.head())
print(email_df.info())
print(email_df.duplicated().sum())

# DATA CLEANING
# extract date features
def extract_date(text):
    """
    Extract date from text in common formats.
    
    Args:
        text: String that may contain a date
        
    Returns:
        str: Extracted date string if found, None otherwise
    """
    if not isinstance(text, str):
        return None
    
    # Match month name format: "May 25, 2001" or "May 25 2001"
    match_text = re.search(
        r'(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2}\s*,?\s*\d{4}', 
        text, 
        re.IGNORECASE
    )
    if match_text:
        return match_text.group(0)
    
    # Match numeric format: "05/29/2001" or "5/29/01"
    match_num = re.search(r'\d{1,2}\s*/\s*\d{1,2}\s*/\s*\d{2,4}', text)
    if match_num:
        return match_num.group(0)
    
    return None

# apply date extraction to subject and body columns
print("Extracting dates...")
email_df['subject_date'] = email_df['subject'].apply(extract_date)
email_df['body_date'] = email_df['body'].apply(extract_date)

# combine subject_date and body_date into a single final_date column
email_df['final_date'] = email_df['subject_date'].combine_first(email_df['body_date'])

print(email_df[['subject_date', 'body_date', 'final_date']].head())

# combine subject and content into a single text column
email_df['Full_email'] = email_df['subject'].fillna('') + ' ' + email_df['body'].fillna('')
print(email_df['Full_email'].head())

# escaping out HTML characters
# convert HTML entities to normal characters
email_df['Full_email'] = email_df['Full_email'].apply(html.unescape)

# remove HTML tags
email_df['Full_email'] = email_df['Full_email'].apply(lambda x: re.sub(r'<.*?>', '', x))

# Removing URLs, Hashtags and Styles
def clean_text(text):
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remove hashtags
    text = re.sub(r'#\S+', '', text)
    # Remove special characters and digits
    text = re.sub(r'[^A-Za-z\s]', '', text)
    # Convert to lowercase
    text = text.lower()
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    # Remove 'RT' at the beginning of the text
    text = re.sub(r'^rt[\s]+', '', text)
    # normalize unicode characters
    text = text.encode('ascii', 'ignore').decode('ascii')
    # split attached words (e.g., helloworld -> hello world)
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    return text

email_df['Cleaned_email'] = email_df['Full_email'].apply(clean_text)
print(email_df['Cleaned_email'].head())

# contraction replacement
# Mapping of contractions to their expanded forms
contraction_mapping = {
    "can't": "cannot",
    "won't": "will not",
    "n't": " not",
    "'re": " are",
    "'s": " is",
    "'d": " would",
    "'ll": " will",
    "'ve": " have",
    "'m": " am",
    "'t": " not",
    "'cause": "because",
    "'em": "them",
    "'bout": "about",
    "'til": "until",
    "'n": "and",
    "'ya": "you",
    "'kay": "okay",
    "'cuz": "because",
    "'cos": "because",
    "'dunno": "do not know",
    "'gonna": "going to",
    "'gotta": "got to",
    "'lemme": "let me",
    "'outta": "out of",
    "'wanna": "want to"
}

# Compile regex pattern for efficient matching
contractions_re = re.compile('(%s)' % '|'.join(contraction_mapping.keys()))

def expand_contractions(text, contractions_dict=contraction_mapping):
    """Expand contractions in text (e.g., can't -> cannot)."""
    return contractions_re.sub(lambda match: contractions_dict[match.group(0)], text)

# Apply expansion to entire column
email_df['Cleaned_email'] = email_df['Cleaned_email'].apply(expand_contractions)

# slang replacement
# Mapping of slang to their expanded forms
slang_mapping = {
    "brb": "be right back",
    "lol": "laughing out loud",
    "idk": "I don't know",
    "smh": "shaking my head",
    "btw": "by the way",
    "imo": "in my opinion",
    "fyi": "for your information",
    "omg": "oh my god",
    "ttyl": "talk to you later",
    "np": "no problem",
    "tbh": "to be honest",
    "lmk": "let me know",
    "rofl": "rolling on the floor laughing",
    "ikr": "I know right",
    "wyd": "what are you doing",
    "yolo": "you only live once"
}

def replace_slang(text, slang_dict=slang_mapping):
    """Replace slang terms with their standard meanings."""
    words = text.split()
    return ' '.join([slang_dict.get(word.lower(), word) for word in words])

# Apply to DataFrame column
email_df['Cleaned_email'] = email_df['Cleaned_email'].apply(replace_slang)

# standardizing - one letter in a word should not be present more than twice in continuation
email_df['Cleaned_email'] = email_df['Cleaned_email'].apply(lambda text: ''.join(''.join(s)[:2] for _, s in itertools.groupby(text)))
print("After standardizing the text is:-\n{}".format(email_df['Cleaned_email'].iloc[0]))

# spell check (commented out - computationally expensive)
# from autocorrect import Speller 
# spell = Speller(lang='en')
# email_df['Cleaned_email'] = email_df['Cleaned_email'].apply(spell)
# print("After Spell check the text is:-\n{}".format(email_df['Cleaned_email'].iloc[0]))

# remove stopwords 
stop_words = set(stopwords.words('english'))
email_df['Cleaned_email'] = email_df['Cleaned_email'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))
print("After removing stopwords the text is:-\n{}".format(email_df['Cleaned_email'].iloc[0]))

# remove punctuations
clean_tweet = [word for word in email_df['Cleaned_email'].iloc[0] if word not in string.punctuation]
print("clean_tweet = {}".format(clean_tweet))

email_df['Cleaned_email'] = email_df['Cleaned_email'].apply(lambda x: ''.join([char for char in x if char not in string.punctuation]))
print("After removing punctuations the text is:-\n{}".format(email_df['Cleaned_email'].iloc[0]))

# final cleaned data
print(email_df[['Full_email', 'Cleaned_email']].head()) 

# save full_email and cleaned_email to a new CSV file
email_df[['Full_email', 'Cleaned_email']].to_csv('C:\\Users\\Surface\\OneDrive\\Documentos\\GitHub\\Spam_sentiment_prediction\\data\\Full_and_Cleaned_Enron.csv', index=False)

# save cleaned data to a new CSV file
email_df.to_csv('C:\\Users\\Surface\\OneDrive\\Documentos\\GitHub\\Spam_sentiment_prediction\\data\\Cleaned_Enron.csv', index=False)