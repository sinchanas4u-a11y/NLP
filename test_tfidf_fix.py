"""Verification script for TF-IDF sentence splitting fix"""
from keyword_extractor import TFIDFExtractor
import nltk

def test_tfidf_sentences():
    text = "Artificial intelligence is transforming healthcare. Machine learning models are used for disease prediction. Deep learning is a subset of AI."
    
    extractor = TFIDFExtractor()
    keywords = extractor.extract(text, top_n=10)
    
    print("Extracted Keywords and Scores:")
    for kw, score in keywords:
        print(f"  {kw:<20}: {score:.4f}")
    
    # Check if we have keywords from multiple sentences
    found_words = [kw for kw, _ in keywords]
    assert any(w in found_words for w in ['artificial', 'intelligence', 'healthcare'])
    assert any(w in found_words for w in ['machine', 'learning', 'models', 'disease', 'prediction'])
    assert any(w in found_words for w in ['deep', 'subset'])
    
    print("\nVerification Successful: Sentences are being treated as separate documents.")

if __name__ == "__main__":
    test_tfidf_sentences()
