"""
Handles text cleaning and preprocessing.
"""
import re
import nltk  # type: ignore
from typing import List, Tuple, Optional, Set, Any, Dict

# Download required NLTK data
def download_nltk_resources():
    for resource in ['punkt', 'punkt_tab', 'stopwords', 'wordnet']:
        try:
            if resource == 'punkt_tab':
                nltk.data.find('tokenizers/punkt_tab')
            elif resource == 'punkt':
                nltk.data.find('tokenizers/punkt')
            elif resource == 'wordnet':
                nltk.data.find('corpora/wordnet')
            else:
                nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download(resource, quiet=True)

class TextProcessor:
    """Handles text cleaning and preprocessing."""
    
    def __init__(self, custom_stopwords: Optional[Set[str]] = None, domain: str = 'general'):
        download_nltk_resources()
        self.domain = domain
        self.stopwords = self._load_stopwords(custom_stopwords)
    
    def _load_stopwords(self, custom_stopwords: Optional[Set[str]] = None) -> Set[str]:
        """Load English stopwords with optional custom additions."""
        try:
            from nltk.corpus import stopwords  # type: ignore
            base_stopwords = set(stopwords.words('english'))
        except:
            base_stopwords = {
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

        base_stopwords.update({'used', 'applied', 'and', 'or', 'to', 'is', 'are', 'by', 'for', 'such', 'as', 'from', 'helps', 'the', 'a', 'an'})
        
        domain_stopwords = {
            'technical': {'using', 'use', 'method', 'methods', 'approach', 'approaches'},
            'product': set(),
            'general': set()
        }
        
        if self.domain in domain_stopwords:
            base_stopwords.update(domain_stopwords[self.domain])
        
        if custom_stopwords:
            base_stopwords.update(custom_stopwords)
        
        return base_stopwords
    
    def add_stopwords(self, words: set):
        self.stopwords.update(words)
    
    def clean_text(self, text: str) -> str:
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-f_A-F][0-9a-f_A-F]))+', ' ', text)
        text = re.sub(r'\S+@\S+', ' ', text)
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        text = ' '.join(text.split())
        return text
    
    def lowercase(self, text: str) -> str:
        return text.lower()
    
    def remove_stopwords(self, text: str) -> str:
        words = text.split()
        filtered_words = [w for w in words if w not in self.stopwords and len(w) > 2]
        return ' '.join(filtered_words)
    
    def lemmatize(self, text: str) -> str:
        try:
            from nltk.stem import WordNetLemmatizer  # type: ignore
            lemmatizer = WordNetLemmatizer()
            words = text.split()
            lemmatized = []
            for word in words:
                if word.lower() == 'models':
                    lemmatized.append(word)
                else:
                    lemmatized.append(lemmatizer.lemmatize(word))
            return ' '.join(lemmatized)
        except:
            return text
    
    def stem(self, text: str) -> str:
        try:
            from nltk.stem import PorterStemmer  # type: ignore
            stemmer = PorterStemmer()
            words = text.split()
            stemmed = [stemmer.stem(word) for word in words]
            return ' '.join(stemmed)
        except:
            return text
    
    def preprocess(self, text: str, use_lemmatization: bool = True) -> str:
        text = self.clean_text(text)
        text = self.lowercase(text)
        text = self.remove_stopwords(text)
        if use_lemmatization:
            text = self.lemmatize(text)
        else:
            text = self.stem(text)
        return text
