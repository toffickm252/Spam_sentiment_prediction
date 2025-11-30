import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import pandas as pd
import os

base_dir = os.getcwd()
data_path = os.path.join(base_dir, 'data', 'Cleaned_Enron_new.csv')
figure_path = os.path.join(base_dir, 'figures', 'wordcloud_distribution.png')

# Load the data
balanced_data = pd.read_csv(data_path)
print("Data loaded successfully.")
print(f"Total rows: {len(balanced_data)}")
print("\n")
print(f"Column names: {balanced_data.columns.tolist()}")
print("\n")
print(f"Label value counts:\n{balanced_data['label'].value_counts()}")
print("\n")
print(f"First few rows:\n{balanced_data.head()}")
print(f"\nNull values in 'cleaned_email' column: {balanced_data['cleaned_email'].isnull().sum()}")

# def plot_word_cloud(data, typ):
#     # Safety checks
#     if len(data) == 0:
#         print(f"Warning: No data found for {typ} emails")
#         return
    
#     # Remove NaN values and convert to string
#     text_data = data['cleaned_email'].dropna().astype(str)
    
#     if len(text_data) == 0:
#         print(f"Warning: No valid text found for {typ} emails")
#         return
    
#     email_corpus = " ".join(text_data)
    
    # # Check if corpus is empty
    # if not email_corpus.strip():
    #     print(f"Warning: Empty corpus for {typ} emails")
    #     return
    
    # print(f"Generating word cloud for {typ} with {len(text_data)} emails")
    
    # wc = WordCloud(background_color='black', max_words=100, width=800, height=400).generate(email_corpus)
    # plt.figure(figsize=(7, 7))
    # plt.imshow(wc, interpolation='bilinear')
    # plt.title(f'WordCloud for {typ} Emails', fontsize=15)
    # plt.axis('off')
    # plt.show()


# plot_word_cloud(balanced_data[balanced_data['label'] == 0], typ='Non-Spam')
# plot_word_cloud(balanced_data[balanced_data['label'] == 1], typ='Spam')

# save the two wordclouds side by side
fig, axes = plt.subplots(1, 2, figsize=(15, 7))
def generate_wordcloud(data, typ):
    text_data = data['cleaned_email'].dropna().astype(str)
    email_corpus = " ".join(text_data)
    wc = WordCloud(background_color='black', max_words=100, width=800, height=400).generate(email_corpus)
    return wc

wc_non_spam = generate_wordcloud(balanced_data[balanced_data['label'] == 0], typ='Non-Spam')
wc_spam = generate_wordcloud(balanced_data[balanced_data['label'] == 1], typ='Spam')
axes[0].imshow(wc_non_spam, interpolation='bilinear')
axes[0].set_title('WordCloud for Non-Spam Emails', fontsize=15)
axes[1].imshow(wc_spam, interpolation='bilinear')
axes[1].set_title('WordCloud for Spam Emails', fontsize=15)
plt.tight_layout()
plt.savefig(figure_path)
plt.show()