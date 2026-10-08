"""
Interactive Keyword Extractor
A command-line program that accepts user text input and extracts keywords using NLP.
"""

import re
import string
from typing import List, Tuple
from collections import Counter

# NLP Libraries
try:
    from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    from rake_nltk import Rake  # type: ignore
    import nltk  # type: ignore
    # Download required NLTK data
    for resource in ['punkt', 'punkt_tab', 'stopwords']:
        try:
            if resource == 'punkt_tab':
                nltk.data.find('tokenizers/punkt_tab')
            elif resource == 'punkt':
                nltk.data.find('tokenizers/punkt')
            else:
                nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download(resource, quiet=True)
    RAKE_AVAILABLE = True
except ImportError:
    RAKE_AVAILABLE = False

try:
    from keybert import KeyBERT  # type: ignore
    KEYBERT_AVAILABLE = True
except ImportError:
    KEYBERT_AVAILABLE = False


class TextProcessor:
    """Handles text cleaning and preprocessing."""
    
    def __init__(self):
        self.stopwords = self._load_stopwords()
    
    def _load_stopwords(self) -> set:
        """Load English stopwords."""
        try:
            from nltk.corpus import stopwords  # type: ignore
            return set(stopwords.words('english'))
        except:
            # Fallback stopwords
            return {
                'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours',
                'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers',
                'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
                'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are',
                'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',
                'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
                'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into',
                'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down',
                'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once'
            }
    
    def clean_text(self, text: str) -> str:
        """
        Clean text by removing punctuation, special characters, and extra spaces.
        
        Args:
            text: Raw input text
            
        Returns:
            Cleaned text
        """
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ' ', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', ' ', text)
        
        # Remove special characters and digits, keep only letters and spaces
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def lowercase(self, text: str) -> str:
        """Convert text to lowercase."""
        return text.lower()
    
    def remove_stopwords(self, text: str) -> str:
        """Remove stopwords from text."""
        words = text.split()
        filtered_words = [w for w in words if w not in self.stopwords and len(w) > 2]
        return ' '.join(filtered_words)
    
    def lemmatize(self, text: str) -> str:
        """
        Apply lemmatization to reduce words to their base form.
        Falls back to stemming if lemmatization is not available.
        """
        try:
            from nltk.stem import WordNetLemmatizer  # type: ignore
            try:
                nltk.data.find('corpora/wordnet')
            except LookupError:
                nltk.download('wordnet', quiet=True)
            
            lemmatizer = WordNetLemmatizer()
            words = text.split()
            lemmatized = [lemmatizer.lemmatize(word) for word in words]
            return ' '.join(lemmatized)
        except:
            # Fallback to simple stemming
            return self.stem(text)
    
    def stem(self, text: str) -> str:
        """Apply stemming to reduce words to their root form."""
        try:
            from nltk.stem import PorterStemmer  # type: ignore
            stemmer = PorterStemmer()
            words = text.split()
            stemmed = [stemmer.stem(word) for word in words]
            return ' '.join(stemmed)
        except:
            return text
    
    def preprocess(self, text: str, use_lemmatization: bool = True) -> str:
        """
        Full preprocessing pipeline.
        
        Args:
            text: Raw input text
            use_lemmatization: Whether to use lemmatization (True) or stemming (False)
            
        Returns:
            Preprocessed text
        """
        # Step 1: Clean text (remove punctuation, special chars, extra spaces)
        text = self.clean_text(text)
        
        # Step 2: Lowercase all words
        text = self.lowercase(text)
        
        # Step 3: Remove stopwords
        text = self.remove_stopwords(text)
        
        # Step 4: Apply stemming or lemmatization
        if use_lemmatization:
            text = self.lemmatize(text)
        else:
            text = self.stem(text)
        
        return text


class KeywordExtractor:
    """Extracts keywords from text using various NLP methods."""
    
    def __init__(self, method: str = 'tfidf'):
        """
        Initialize the keyword extractor.
        
        Args:
            method: Extraction method ('tfidf', 'rake', or 'keybert')
        """
        self.method = method.lower()
        self.processor = TextProcessor()
        self.keybert_model = None
        
        # Validate method
        available_methods = []
        if SKLEARN_AVAILABLE:
            available_methods.append('tfidf')
        if RAKE_AVAILABLE:
            available_methods.append('rake')
        if KEYBERT_AVAILABLE:
            available_methods.append('keybert')
        
        if self.method not in available_methods:
            if available_methods:
                self.method = available_methods[0]
                print(f"Method not available. Using '{self.method}' instead.")
            else:
                raise ImportError("No keyword extraction libraries available. Please install requirements.")
    
    def extract_tfidf(self, text: str, top_n: int = 10) -> List[Tuple[str, float]]:
        """
        Extract keywords using TF-IDF.
        
        Args:
            text: Preprocessed text
            top_n: Number of top keywords to return
            
        Returns:
            List of (keyword, score) tuples
        """
        # Create a corpus with the text and a portion of it
        corpus = [text, text[:len(text)//2] if len(text) > 20 else text]  # type: ignore
        
        # Initialize TF-IDF vectorizer
        vectorizer = TfidfVectorizer(
            max_features=1000,
            ngram_range=(1, 2),
            stop_words='english',
            min_df=1,
            max_df=1.0
        )
        
        try:
            # Fit and transform
            tfidf_matrix = vectorizer.fit_transform(corpus)
            feature_names = vectorizer.get_feature_names_out()
            
            # Get scores for the first document
            scores = tfidf_matrix[0].toarray().flatten()
            
            # Create keyword-score pairs
            keywords = [(feature_names[i], float(scores[i])) for i in range(len(feature_names))]
            
            # Sort by score descending
            keywords.sort(key=lambda x: x[1], reverse=True)
            
            return keywords[:top_n]  # type: ignore
        except Exception as e:
            print(f"TF-IDF extraction error: {e}")
            return []
    
    def extract_rake(self, text: str, top_n: int = 10) -> List[Tuple[str, float]]:
        """
        Extract keywords using RAKE.
        
        Args:
            text: Original text (RAKE works better with original text)
            top_n: Number of top keywords to return
            
        Returns:
            List of (keyword, score) tuples
        """
        try:
            r = Rake(max_length=3)
            r.extract_keywords_from_text(text)
            keywords = r.get_ranked_phrases_with_scores()
            
            result = []
            for score, phrase in keywords[:top_n]:
                normalized_score = min(score / 10.0, 1.0) if score > 0 else 0.0
                result.append((phrase.lower(), normalized_score))
            
            return result
        except Exception as e:
            print(f"RAKE extraction error: {e}")
            return []
    
    def extract_keybert(self, text: str, top_n: int = 10) -> List[Tuple[str, float]]:
        """
        Extract keywords using KeyBERT.
        
        Args:
            text: Original text
            top_n: Number of top keywords to return
            
        Returns:
            List of (keyword, score) tuples
        """
        try:
            if self.keybert_model is None:
                print("Loading KeyBERT model (first time only)...")
                self.keybert_model = KeyBERT()
            
            if self.keybert_model is None:
                return []
                
            keywords = self.keybert_model.extract_keywords(  # type: ignore
                text,
                keyphrase_ngram_range=(1, 2),
                stop_words='english',
                top_n=top_n,
                use_mmr=True,
                diversity=0.7
            )
            
            return [(kw, float(score)) for kw, score in keywords]
        except Exception as e:
            print(f"KeyBERT extraction error: {e}")
            return []
    
    def extract(self, text: str, top_n: int = 10) -> List[Tuple[str, float]]:
        """
        Extract keywords using the selected method.
        
        Args:
            text: Input text
            top_n: Number of top keywords to return
            
        Returns:
            List of (keyword, score) tuples
        """
        if not text or len(text.strip()) < 10:
            print("Error: Text too short. Please provide at least 10 characters.")
            return []
        
        # Preprocess text for TF-IDF
        if self.method == 'tfidf':
            processed_text = self.processor.preprocess(text)
            return self.extract_tfidf(processed_text, top_n)
        
        # RAKE works better with original text
        elif self.method == 'rake':
            return self.extract_rake(text, top_n)
        
        # KeyBERT works better with original text
        elif self.method == 'keybert':
            return self.extract_keybert(text, top_n)
        
        return []


def display_keywords(keywords: List[Tuple[str, float]], method: str):
    """Display extracted keywords in a formatted way."""
    print("\n" + "=" * 60)
    print(f"EXTRACTED KEYWORDS (using {method.upper()})")
    print("=" * 60)
    
    if not keywords:
        print("No keywords extracted.")
        return
    
    print(f"\n{'Rank':<6} {'Keyword':<30} {'Relevance Score'}")
    print("-" * 60)
    
    for i, (keyword, score) in enumerate(keywords, 1):
        print(f"{i:<6} {keyword:<30} {score:.4f}")
    
    print("\n" + "=" * 60)


def main():
    """Main interactive program."""
    print("=" * 60)
    print("INTERACTIVE KEYWORD EXTRACTOR")
    print("=" * 60)
    print("\nThis program extracts keywords from your text using NLP.")
    
    # Show available methods
    available = []
    if SKLEARN_AVAILABLE:
        available.append("TF-IDF")
    if RAKE_AVAILABLE:
        available.append("RAKE")
    if KEYBERT_AVAILABLE:
        available.append("KeyBERT")
    
    print(f"\nAvailable methods: {', '.join(available)}")
    
    # Select method
    while True:
        method_choice = input("\nSelect method (tfidf/rake/keybert) [default: tfidf]: ").strip().lower()
        if not method_choice:
            method_choice = 'tfidf'
        if method_choice in ['tfidf', 'rake', 'keybert']:
            break
        print("Invalid choice. Please select 'tfidf', 'rake', or 'keybert'.")
    
    # Initialize extractor
    try:
        extractor = KeywordExtractor(method=method_choice)
    except Exception as e:
        print(f"Error initializing extractor: {e}")
        return
    
    print(f"\nUsing method: {method_choice.upper()}")
    print("\n" + "-" * 60)
    
    # Main loop
    while True:
        print("\nEnter or paste your text (press Enter to process):")
        print("-" * 60)
        
        try:
            text = input()
        except EOFError:
            break
        
        if not text.strip():
            print("No text entered. Exiting...")
            break
        
        # Get number of keywords
        while True:
            try:
                top_n_input = input("\nHow many keywords to extract? [default: 10]: ").strip()
                top_n = int(top_n_input) if top_n_input else 10
                if top_n > 0:
                    break
                print("Please enter a positive number.")
            except ValueError:
                print("Please enter a valid number.")
        
        # Extract keywords
        print("\nProcessing text...")
        keywords = extractor.extract(text, top_n=top_n)
        
        # Display results
        display_keywords(keywords, method_choice)
        
        # Ask to continue
        print("\n" + "-" * 60)
        choice = input("Extract keywords from another text? (y/n): ").strip().lower()
        if choice != 'y':
            print("\nThank you for using the Keyword Extractor!")
            break


if __name__ == "__main__":
    main()
