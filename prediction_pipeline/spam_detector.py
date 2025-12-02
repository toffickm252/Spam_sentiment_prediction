import joblib
import os

class SpamDetector:
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
        
        # Convert the email text to numbers
        email_vectorized = self.vectorizer.transform([email_text])
        
        # Get the prediction (0 or 1)
        prediction = self.model.predict(email_vectorized)[0]
        
        # Get the confidence scores
        probabilities = self.model.predict_proba(email_vectorized)[0]
        
        # Package everything into a dictionary that's easy to understand
        return {
            'is_spam': bool(prediction == 1),
            'label': 'SPAM' if prediction == 1 else 'NOT SPAM',
            'confidence': float(max(probabilities) * 100),
            'spam_probability': float(probabilities[1] * 100),
            'not_spam_probability': float(probabilities[0] * 100)
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
    detector = SpamDetector()
    
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
        print("-" * 70)