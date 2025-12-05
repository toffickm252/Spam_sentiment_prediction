# Email Spam Detection & Sentiment Analysis System

A machine learning pipeline that intelligently filters spam emails and analyzes the emotional sentiment of legitimate messages.

## 🎯 Overview

This project combines spam detection with sentiment analysis to provide a two-stage email analysis system:
1. **Spam Detection**: Identifies and filters spam emails with 98% accuracy
2. **Sentiment Analysis**: Analyzes the emotional tone (positive, negative, neutral) of non-spam emails

The system is designed for computational efficiency - sentiment analysis only runs on emails that pass the spam filter, saving processing resources.

## ✨ Key Features

- **High Accuracy**: 98% spam detection accuracy using Support Vector Machine (SVM)
- **Smart Pipeline**: Two-stage analysis that processes sentiment only for legitimate emails
- **Multiple Model Training**: Evaluated Logistic Regression, Random Forest, SVM, and Naive Bayes to select the best performer
- **Human-Readable Output**: Converts sentiment scores into interpretable emotions (Happy, Sad, Neutral, etc.)
- **Clean Architecture**: Separated training code from deployment code for maintainability

## 📊 Model Performance

| Model | Accuracy |
|-------|----------|
| **SVM (Selected)** | **98%** |
| Logistic Regression | ~97% |
| Random Forest | ~96% |
| Naive Bayes | ~95% |

## 🛠️ Tech Stack

- **Python 3.x**
- **scikit-learn**: Model training (SVM, TF-IDF vectorization)
- **TextBlob**: Sentiment analysis
- **NLTK**: Text preprocessing
- **pandas**: Data manipulation
- **joblib**: Model serialization

## 📁 Project Structure

```
SPAM_SENTIMENT_PREDICTION/
├── data/
│   ├── Enron.csv                    # Original dataset
│   └── Cleaned_Enron_new.csv        # Preprocessed data
├── models/
│   ├── spam_classifier_model.joblib # Trained SVM model
│   └── tfidf_vectorizer.joblib      # Fitted TF-IDF vectorizer
├── src/
│   ├── cleaning_data.py             # Data preprocessing
│   ├── train_model.py               # Model training pipeline
│   └── visualization.py             # Performance visualizations
├── app_and_deployment/
│   ├── text_preprocessing.py        # Text cleaning functions
│   └── email_analyzer.py            # Main analyzer class
├── figures/
│   ├── balanced_distribution_new.png
│   └── wordcloud_distribution.png
└── requirements.txt
```

## 🚀 Installation

### 1. Clone the repository
```bash
git clone https://github.com/toffickm252/spam_sentiment_prediction.git
cd spam_sentiment_prediction
```

### 2. Create virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Download NLTK data
```python
import nltk
nltk.download('stopwords')
```

## 💻 Usage

### Quick Start

```python
from app_and_deployment.email_analyzer import EmailAnalyzer

# Initialize the analyzer (loads model automatically)
analyzer = EmailAnalyzer()

# Analyze a single email
email_text = "Hey, can we reschedule our meeting to 3pm tomorrow?"
result = analyzer.check_email(email_text)

print(f"Spam Status: {result['label']}")
print(f"Confidence: {result['confidence']:.1f}%")

if result['sentiment']:
    print(f"Sentiment: {result['sentiment']['interpretation']}")
    print(f"Polarity: {result['sentiment']['polarity']:.2f}")
```

### Expected Output

```
Spam Status: NOT SPAM
Confidence: 99.7%
Sentiment: Neutral/Moderate
Polarity: 0.00
```

### Batch Processing

```python
emails = [
    "Congratulations! You've won $1,000,000!",
    "Thanks for the report. I'll review it today."
]

results = analyzer.check_multiple_emails(emails)
for i, result in enumerate(results, 1):
    print(f"Email {i}: {result['label']}")
```

## 🔧 How It Works

### 1. Text Preprocessing
Emails are cleaned using multiple steps:
- Lowercasing
- Contraction expansion (can't → cannot)
- Removal of HTML, URLs, special characters
- Stopword removal
- Whitespace normalization

### 2. Spam Detection
- Text is vectorized using **TF-IDF** (5000 features)
- **Linear SVM** classifier predicts spam probability
- Threshold: 50% (adjustable)

### 3. Sentiment Analysis (Non-Spam Only)
- Uses **TextBlob** on raw (uncleaned) text
- Extracts polarity (-1 to +1) and subjectivity (0 to 1)
- Maps polarity to human-readable emotions:
  - **0.6 to 1.0**: Very Positive (Excited/Enthusiastic)
  - **0.3 to 0.6**: Positive (Happy/Pleased)
  - **-0.1 to 0.3**: Neutral/Moderate
  - **-0.5 to -0.1**: Negative (Sad/Disappointed)
  - **-1.0 to -0.5**: Very Negative (Angry/Upset)

## 📈 Training Your Own Model

```bash
# 1. Prepare your dataset (CSV with 'email' and 'label' columns)
# 2. Run data cleaning
python src/cleaning_data.py

# 3. Train models
python src/train_model.py

# 4. Models will be saved to models/ directory
```

## 🎯 Design Decisions

### Why Two-Stage Processing?
Running sentiment analysis on all emails is computationally wasteful. Since spam emails are discarded anyway, analyzing their sentiment provides no value. This architecture saves ~40-60% of sentiment processing time depending on spam prevalence.

### Why Raw Text for Sentiment?
TextBlob needs punctuation, capitalization, and grammar to accurately assess emotion:
- "THANK YOU!!!" vs "thank you" have different emotional intensities
- Cleaning removes these crucial sentiment indicators

### Why Separate Training & Deployment?
- **Training code** (`src/`): Experimental, messy, runs once
- **Deployment code** (`app_and_deployment/`): Clean, reusable, production-ready
- This separation improves maintainability and makes the codebase more professional

## 🔮 Next Steps

### Phase 1: Batch Processing ✅ (In Progress)
- Build CLI tool to process email datasets
- Generate analysis reports (CSV/JSON output)
- Add progress bars for large datasets

### Phase 2: REST API 🔄 (Planned)
- Flask/FastAPI endpoint for email analysis
- Docker containerization
- API documentation with Swagger
- Rate limiting and authentication

### Phase 3: Web Interface 📱 (Future)
- Streamlit dashboard
- Real-time email analysis
- Visualization of results
- Export functionality

## 📊 Dataset

**Enron Email Dataset**
- Source: Public domain dataset from Enron Corporation
- Size: ~33,000 emails
- Classes: Spam (1) and Ham/Not-Spam (0)
- Preprocessing: Balanced dataset to prevent class imbalance

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/improvement`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author

**Your Name**
- GitHub: (https://github.com/toffickm252)
- X: [@elzer252](https://twitter.com/elzer252)

## 🙏 Acknowledgments

- Enron Email Dataset
- scikit-learn documentation
- TextBlob library
- The open-source community

---

⭐ If you found this project helpful, please consider giving it a star!

## 📧 Contact

Have questions or suggestions? Feel free to:
- Open an issue
- Submit a pull request
- Reach out via email: toffickm252@gmail.com
