import joblib
import os
from text_preprocessing import clean_email
from textblob import TextBlob
import re


class EmailAnalyzer:
    """
    A reusable spam detection tool.
    
    Think of this as a spam-checking machine that you set up once,
    then you can feed it as many emails as you want to check.
    """
    
    def __init__(self):
        """
        This runs once when you first create the detector.
        It loads the model and vectorizer into memory so they're ready to use.
        """
        # Figure out where the model files are
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        
        model_path = os.path.join(project_root, 'models', 'spam_classifier_model.joblib')
        vectorizer_path = os.path.join(project_root, 'models', 'tfidf_vectorizer.joblib')
        
        print("Loading spam detector...")
        # Load both files into memory
        self.model = joblib.load(model_path)
        self.vectorizer = joblib.load(vectorizer_path)
        print("Spam detector ready!\n")
    
    def interpret_sentiment(self, polarity):
        """
        Internal function to interpret sentiment polarity.
        """
        if polarity >= 0.6:
            return 'Very Positive (Excited/Enthusiastic)'
        elif polarity >= 0.3:
            return 'Positive (Happy/Pleased)'
        elif polarity >= -0.1:
            return 'Slightly Positive (Content/Optimistic)'
        elif polarity > -0.5:
            return 'Slightly Negative (Disappointed/Unhappy)'
        else: 
            return 'Very Negative (Angry/Upset)'
    
    def check_email(self, email_text):
        """
        Check if a single email is spam.
        
        This is the function you'll call each time you want to check an email.
        It takes the email text and returns a simple answer about whether it's spam.
        """
        # Handle the case where someone passes an empty email
        if not email_text or email_text.strip() == '':
            return {
                'is_spam': False,
                'label': 'NOT SPAM',
                'confidence': 0.0,
                'message': 'Empty email provided'
            }
        
        # First, clean the email text
        cleaned_email = clean_email(email_text)
        # Convert the email text to numbers
        email_vectorized = self.vectorizer.transform([cleaned_email])
        
        # Get the prediction (0 or 1)
        prediction = self.model.predict(email_vectorized)[0]

        # probabilities = self.model.predict_proba(email_vectorized)[0]

        # spam_score = probabilities[1]
        # threshold = 0.75

        # # Recalculate prediction based on threshold
        # prediction = 1 if spam_score >= threshold else 0

        # Initialize sentiment_result to None
        sentiment_result = None

        # Only analyze sentiment if NOT spam
        if prediction == 0:
            blob = TextBlob(email_text)
            sentiment = blob.sentiment
            sentiment_result = {
                'polarity': float(sentiment.polarity),
                'subjectivity': float(sentiment.subjectivity),
                'interpretation': self.interpret_sentiment(sentiment.polarity)
            }

        # Get the confidence scores
        probabilities = self.model.predict_proba(email_vectorized)[0]
        

        # print("\n--- DEBUG ---")
        # print("Original:", email_text[:60])
        # print("Cleaned:", cleaned_email[:60])
        # print("Prediction:", prediction)
        # print("Probabilities:", probabilities)
        # print("Sentiment:", sentiment_result)
        # print("--- END DEBUG ---\n")

        # Package everything into a dictionary that's easy to understand
        return {
            'is_spam': bool(prediction == 1),
            'label': 'SPAM' if prediction == 1 else 'NOT SPAM',
            'confidence': float(max(probabilities) * 100),
            'spam_probability': float(probabilities[1] * 100),
            'not_spam_probability': float(probabilities[0] * 100),
            'sentiment': sentiment_result if prediction == 0 else None
        }
    
    def check_multiple_emails(self, email_list):
        """
        Check multiple emails at once.
        
        This is useful if you have a list of emails and want to check all of them.
        It returns a list of results, one for each email.
        """
        return [self.check_email(email) for email in email_list]


# This part only runs if you run this file directly (not if you import it)
if __name__ == "__main__":
    # Create the detector (this loads the model)
    detector = EmailAnalyzer()
    
    # Test it with a few different emails
    test_emails = [
        "Congratulations! You've won a $1,000 Walmart gift card. Click here to claim your prize.",
        "Hey John, can we reschedule our meeting to Thursday at 3pm? Let me know if that works.",
        "URGENT: Your bank account has been compromised. Click here immediately to verify your identity.",
        "Thanks for sending over the quarterly report. I'll review it and get back to you by Friday."
    ]
    
    print("Testing spam detector with multiple emails:")
    print("=" * 70)
    
    for i, email in enumerate(test_emails, 1):
        result = detector.check_email(email)
        print(f"\nEmail {i}:")
        print(f"Text: {email[:60]}...")
        print(f"Result: {result['label']} (Confidence: {result['confidence']:.1f}%)")
        
        if result['sentiment']:
            print(f"Sentiment: {result['sentiment']['interpretation']}")
            print(f"  Polarity: {result['sentiment']['polarity']:.2f}, Subjectivity: {result['sentiment']['subjectivity']:.2f}")
        print("-" * 70)