"""
Example usage of the Keyword Extraction System
"""

from keyword_extractor import KeywordExtractor, extract_keywords, create_wordcloud  # type: ignore

# Sample text for testing
SAMPLE_TEXT = """
Machine learning is a subset of artificial intelligence that enables systems to learn 
and improve from experience without being explicitly programmed. Machine learning algorithms 
build mathematical models based on training data to make predictions or decisions. 
Deep learning is a specialized form of machine learning that uses neural networks with 
many layers. These neural networks can learn complex patterns in large amounts of data. 
Natural language processing is another important field that combines linguistics and 
machine learning to help computers understand human language. Training data quality is 
crucial for building effective machine learning models. Supervised learning, unsupervised 
learning, and reinforcement learning are the main types of machine learning approaches.
"""

def main():
    print("=" * 70)
    print("KEYWORD EXTRACTION SYSTEM - EXAMPLE USAGE")
    print("=" * 70)
    
    # Initialize extractor
    extractor = KeywordExtractor()
    
    print("\n📄 Sample Text:")
    print("-" * 50)
    print(SAMPLE_TEXT[:200] + "...")
    print("-" * 50)
    
    # Method 1: Quick extraction using convenience function
    print("\n🔍 Method 1: Quick extraction with KeyBERT (default)")
    print("-" * 50)
    keywords = extract_keywords(SAMPLE_TEXT, method='keybert', top_n=5)
    print(f"Top 5 keywords: {keywords}")
    
    # Method 2: Using all extraction methods
    print("\n🔍 Method 2: Compare all extraction methods")
    print("-" * 50)
    
    methods = ['tfidf', 'rake', 'keybert', 'spacy']
    
    for method in methods:
        print(f"\n📌 {method.upper()}:")
        keywords = extractor.extract(SAMPLE_TEXT, method=method, top_n=5)
        for kw, score in keywords:
            print(f"   • {kw}: {score:.4f}")
    
    # Method 3: Extract with all methods at once
    print("\n🔍 Method 3: Extract with all methods (extract_all)")
    print("-" * 50)
    all_results = extractor.extract_all(SAMPLE_TEXT, top_n=3)
    for method, keywords in all_results.items():
        print(f"\n{method.upper()}: {keywords}")
    
    # Method 4: Different n-gram ranges
    print("\n🔍 Method 4: Multi-word keyphrases (bigrams/trigrams)")
    print("-" * 50)
    
    # Unigrams only
    keywords_unigram = extractor.extract(
        SAMPLE_TEXT, method='keybert', top_n=5, ngram_range=(1, 1)
    )
    print(f"\nUnigrams (single words): {[kw for kw, _ in keywords_unigram]}")
    
    # Bigrams and trigrams
    keywords_phrases = extractor.extract(
        SAMPLE_TEXT, method='keybert', top_n=5, ngram_range=(2, 3)
    )
    print(f"Bigrams/Trigrams (multi-word): {[kw for kw, _ in keywords_phrases]}")
    
    # Method 5: Generate word cloud
    print("\n☁️ Method 5: Generate word cloud visualization")
    print("-" * 50)
    
    # Get keywords for visualization
    viz_keywords = extractor.extract(SAMPLE_TEXT, method='keybert', top_n=20)
    
    # Create word cloud
    try:
        create_wordcloud(viz_keywords, output_path='keyword_wordcloud.png')
        print("✅ Word cloud saved to 'keyword_wordcloud.png'")
    except Exception as e:
        print(f"⚠️ Could not create word cloud: {e}")
    
    # Edge case handling
    print("\n🛡️ Edge Case Handling")
    print("-" * 50)
    
    test_cases = [
        ("", "Empty string"),
        ("Hi", "Too short"),
        ("This is a test sentence that should work fine.", "Valid short text"),
        ("这是一个中文文本测试", "Non-English text"),
    ]
    
    for test_text, description in test_cases:
        print(f"\nTest: {description}")
        result = extractor.extract(test_text, top_n=3)
        if result:
            print(f"   Result: {result}")
        else:
            print(f"   Result: (empty - handled gracefully)")
    
    print("\n" + "=" * 70)
    print("EXAMPLE COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    main()
