# import libraries
import re
import pandas as pd
import numpy as np

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
import html
email_df['Full_email'] = email_df['Full_email'].apply(lambda x: html.unescape(x))

# remove HTML tags
email_df['Full_email'] = email_df['Full_email'].apply(lambda x: re.sub(r'<.*?>', '', x))

# detect encoded text (e.g., =C3=A9 for é) using chardet
import chardet

# # Fetch the web page content
# response = requests.get('https://www.geeksforgeeks.org/&/#39;)
# html_content = response.content

# Detect the encoding
# result = chardet.detect(html_content)
# encoding = result['encoding']
# print(f'Detected encoding: {encoding}')

results = email_df['Full_email'].apply(lambda x: chardet.detect(x.encode()))
email_df['encoding'] = results.apply(lambda x: x['encoding'])
# print detected encodings
print(email_df['encoding'].value_counts())