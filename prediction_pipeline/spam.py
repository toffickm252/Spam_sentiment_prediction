from spam_detector import SpamDetector

detector = SpamDetector()
result = detector.check_email("Thank you for the wonderful and overwhelming response to our Black Friday Sale! Your trust and enthusiasm have made this our biggest learning event of the year. We are thrilled to have you as part of our community and look forward to bringing you more exciting offers in the future. Stay tuned for upcoming deals and exclusive discounts. Happy shopping!")
print(result['label'])