"""
Stopword Finder
A program that accepts text input and displays all stopwords found in the text.
"""

import re
from typing import List, Tuple, Counter
from collections import Counter

# NLP Libraries
try:
    from nltk.corpus import stopwords  # type: ignore
    import nltk  # type: ignore
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', quiet=True)
    NLTK_STOPWORDS = set(stopwords.words('english'))
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    # Fallback stopwords
    NLTK_STOPWORDS = {
        'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours',
        'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers',
        'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
        'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are',
        'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',
        'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
        'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into',
        'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down',
        'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here',
        'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',
        'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
        'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now'
    }


class StopwordFinder:
    """Finds and displays stopwords in text."""
    
    def __init__(self):
        self.stopwords = NLTK_STOPWORDS
    
    def clean_and_tokenize(self, text: str) -> List[str]:
        """
        Clean text and tokenize into words.
        
        Args:
            text: Raw input text
            
        Returns:
            List of words
        """
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ' ', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', ' ', text)
        
        # Remove special characters, keep only letters and spaces
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        
        # Remove extra whitespace and split into words
        words = text.split()
        
        return words
    
    def find_stopwords(self, text: str) -> Tuple[List[str], List[str], int]:
        """
        Find all stopwords in the text.
        
        Args:
            text: Input text
            
        Returns:
            Tuple of (all_stopwords, unique_stopwords, total_count)
        """
        # Tokenize
        words = self.clean_and_tokenize(text)
        
        # Lowercase and find stopwords
        all_stopwords = []
        for word in words:
            word_lower = word.lower()
            if word_lower in self.stopwords:
                all_stopwords.append(word_lower)
        
        # Get unique stopwords
        unique_stopwords = sorted(list(set(all_stopwords)))
        
        return all_stopwords, unique_stopwords, len(all_stopwords)
    
    def display_stopwords(self, text: str):
        """
        Display all stopwords found in the text.
        
        Args:
            text: Input text
        """
        all_stopwords, unique_stopwords, total_count = self.find_stopwords(text)
        
        print("\n" + "=" * 70)
        print("STOPWORD ANALYSIS RESULTS")
        print("=" * 70)
        
        # Total count
        print(f"\n📊 TOTAL STOPWORDS FOUND: {total_count}")
        
        if total_count == 0:
            print("\nNo stopwords found in the input text.")
            return
        
        # All stopwords with duplicates
        print("\n" + "-" * 70)
        print("ALL STOPWORDS (with duplicates):")
        print("-" * 70)
        print(", ".join(all_stopwords))
        
        # Unique stopwords
        print("\n" + "-" * 70)
        print("UNIQUE STOPWORDS:")
        print("-" * 70)
        print(", ".join(unique_stopwords))
        print(f"\nTotal unique stopwords: {len(unique_stopwords)}")
        
        # Frequency count
        print("\n" + "-" * 70)
        print("STOPWORD FREQUENCY:")
        print("-" * 70)
        freq = Counter(all_stopwords)
        print(f"{'Stopword':<20} {'Count':<10} {'Visual'}")
        print("-" * 70)
        for word, count in freq.most_common():
            bar = "█" * count
            print(f"{word:<20} {count:<10} {bar}")
        
        print("\n" + "=" * 70)


def main():
    """Main interactive program."""
    print("=" * 70)
    print("STOPWORD FINDER")
    print("=" * 70)
    print("\nThis program identifies and displays all stopwords in your text.")
    print("Stopwords are common words like 'the', 'is', 'and', 'a', etc.")
    
    finder = StopwordFinder()
    
    # Main loop
    while True:
        print("\n" + "-" * 70)
        print("Enter or paste your text (press Enter to process):")
        print("-" * 70)
        
        try:
            text = input()
        except EOFError:
            break
        
        if not text.strip():
            print("No text entered. Exiting...")
            break
        
        # Process and display
        finder.display_stopwords(text)
        
        # Ask to continue
        print("\n" + "-" * 70)
        choice = input("Analyze another text? (y/n): ").strip().lower()
        if choice != 'y':
            print("\nThank you for using the Stopword Finder!")
            break


if __name__ == "__main__":
    main()
