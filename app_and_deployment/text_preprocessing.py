import string
import re
import nltk
from nltk.corpus import stopwords

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

def clean_email(text):
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