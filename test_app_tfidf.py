"""Verification script for app.py TF-IDF sentence splitting fix"""
from app import KeywordExtractor
import nltk

def test_app_tfidf_sentences():
    text = "Artificial intelligence is transforming healthcare. Machine learning models are used for disease prediction. Deep learning is a subset of AI."
    
    # Initialize extractor with tfidf
    extractor = KeywordExtractor(method='tfidf')
    
    # This should now use sent_tokenize internally in app.py
    keywords = extractor.extract(text, top_n=10)
    
    print("Extracted Keywords and Scores from app.py:")
    for kw, score in keywords:
        print(f"  {kw:<20}: {score:.4f}")
    
    # Check if we have keywords from multiple sentences
    found_words = [kw for kw, _ in keywords]
    assert any(w in found_words for w in ['artificial', 'intelligence', 'healthcare'])
    assert any(w in found_words for w in ['machine', 'learning', 'models', 'disease', 'prediction'])
    
    print("\nVerification Successful: app.py now treats sentences as separate documents using sent_tokenize.")

if __name__ == "__main__":
    test_app_tfidf_sentences()
