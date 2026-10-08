"""Verification script for KeyBERT loading in app.py"""
from keyword_extractor import KeywordExtractor

def test_keybert_load():
    print("Testing KeyBERT loading...")
    text = "Artificial intelligence is transforming healthcare with machine learning."
    
    # Initialize extractor with keybert
    # This will trigger loading the model
    extractor = KeywordExtractor(method='keybert')
    
    print("Attempting to extract keywords with KeyBERT...")
    keywords = extractor.extract(text, top_n=5)
    
    if keywords:
        print("KeyBERT Extracted Keywords:")
        for kw, score in keywords:
            print(f"  {kw:<20}: {score:.4f}")
        print("\nVerification Successful: KeyBERT model loaded and extracted keywords successfully.")
    else:
        print("\nVerification Note: KeyBERT did not return keywords. Check if it's still because of dependencies OR if the text is too short.")
        # Try a longer text just in case
        long_text = "Artificial intelligence is a subset of computer science that deals with the creation of intelligent agents. Machine learning is a field of inquiry devoted to understanding and building methods that 'learn'."
        keywords = extractor.extract(long_text, top_n=5)
        if keywords:
             print("KeyBERT Extracted Keywords (Long Text):")
             for kw, score in keywords:
                 print(f"  {kw:<20}: {score:.4f}")
             print("\nVerification Successful: KeyBERT model loaded successfully with longer text.")
        else:
             print("\nVerification Failed: KeyBERT still not returning keywords or failing to load.")

if __name__ == "__main__":
    test_keybert_load()
