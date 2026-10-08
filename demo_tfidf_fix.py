"""
Demonstration of the new TF-IDF keyword extraction implementation.
"""
from keyword_extractor import KeywordExtractor

def main():
    # Example input text block with varying word frequencies.
    # The word "learning" and "machine" are intentionally repeated within the same paragraph/document 
    # to demonstrate that TF values are actually reflective of frequency now.
    text = """
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

    print("=" * 70)
    print("TF-IDF KEYWORD EXTRACTION DEMONSTRATION")
    print("=" * 70)
    print("\n[Input Text]:")
    print(text.strip())
    print("\n" + "-" * 70)

    # Initialize unified KeywordExtractor (which under the hood uses TFIDFExtractor)
    extractor = KeywordExtractor(method='tfidf')
    
    # Extract top 15 keywords to show varying values
    keywords = extractor.extract(text, top_n=15)
    
    print("\n[Extracted Keywords and TF-IDF Scores]:")
    print(f"{'Rank':<6} | {'Keyword':<20} | {'Score/TF-IDF'}")
    print("-" * 50)
    
    for i, (kw, score) in enumerate(keywords, 1):
        print(f"{i:<6} | {kw:<20} | {score:.4f}")

    print("\nAs shown above, the scores vary widely because their occurrences (Term Frequencies)")
    print("within this single text dataset are correctly counted and scaled by the custom TF-IDF formula!")

if __name__ == "__main__":
    main()
