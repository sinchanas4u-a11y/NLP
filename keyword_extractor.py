"""
Unified Keyword Extraction Interface
Aggregates TF-IDF, RAKE, KeyBERT, and spaCy methods.
"""
import os
import re
from typing import List, Tuple, Optional, Dict, Any
from collections import Counter

# Import local modules
from preprocessing import TextProcessor
from tfidf_extraction import TFIDFExtractor
from rake_extraction import RAKEExtractor
from keybert_extraction import KeyBERTExtractor

class KeywordExtractor:
    """Main interface for keyword extraction using multiple methods."""
    
    def __init__(self, method: str = 'keybert'):
        self.method = method.lower()
        self.processor = TextProcessor()
        self.extractors = {
            'tfidf': TFIDFExtractor(self.processor),
            'rake': RAKEExtractor(self.processor),
            'keybert': KeyBERTExtractor(self.processor)
        }
        self.nlp = None  # Lazy load spaCy

    def _load_spacy(self):
        """Lazy load spaCy model."""
        if self.nlp is None:
            try:
                import spacy
                try:
                    self.nlp = spacy.load("en_core_web_sm")
                except OSError:
                    print("spaCy model 'en_core_web_sm' not found. Downloading...")
                    os.system("python -m spacy download en_core_web_sm")
                    self.nlp = spacy.load("en_core_web_sm")
            except Exception as e:
                print(f"Error loading spaCy: {e}")
                return None
        return self.nlp

    def extract_spacy(self, text: str, top_n: int = 10) -> List[Tuple[str, float]]:
        """Extract keywords using spaCy noun chunks."""
        nlp = self._load_spacy()
        if not nlp:
            return []
        
        doc = nlp(text)
        # Extract noun chunks that aren't just stopwords
        keywords = []
        for chunk in doc.noun_chunks:
            chunk_text = chunk.text.lower().strip()
            # Basic filtering
            if len(chunk_text) > 2 and chunk_text not in self.processor.stopwords:
                # Remove leading/trailing stopwords from chunks
                words = chunk_text.split()
                if words and words[0] not in self.processor.stopwords and words[-1] not in self.processor.stopwords:
                    keywords.append(chunk_text)
        
        # Count frequencies and normalize
        counts = Counter(keywords)
        total = sum(counts.values())
        
        if total == 0:
            return []
            
        result = [(kw, round(count/total, 4)) for kw, count in counts.most_common(top_n)]
        return result

    def extract(self, text: str, method: Optional[str] = None, top_n: int = 10, **kwargs) -> List[Tuple[str, float]]:
        """Generic extract method supporting multiple backends."""
        target_method = (method or self.method).lower()
        
        if target_method == 'spacy':
            return self.extract_spacy(text, top_n)
        
        extractor = self.extractors.get(target_method)
        if not extractor:
            print(f"Method '{target_method}' not recognized. Falling back to KeyBERT.")
            extractor = self.extractors['keybert']
            
        return extractor.extract(text, top_n=top_n)

    def extract_all(self, text: str, top_n: int = 10) -> Dict[str, List[Tuple[str, float]]]:
        """Extract keywords using all available methods for comparison."""
        results = {}
        for m in ['tfidf', 'rake', 'keybert', 'spacy']:
            results[m] = self.extract(text, method=m, top_n=top_n)
        return results

def extract_keywords(text: str, method: str = 'keybert', top_n: int = 10) -> List[Tuple[str, float]]:
    """Convenience function for quick extraction."""
    extractor = KeywordExtractor(method=method)
    return extractor.extract(text, top_n=top_n)

def create_wordcloud(keywords: List[Tuple[str, float]], output_path: str = 'keyword_wordcloud.png'):
    """Generate and save a word cloud from keywords."""
    try:
        from wordcloud import WordCloud
        import matplotlib.pyplot as plt
        
        if not keywords:
            print("No keywords provided for word cloud.")
            return

        # Convert list of tuples to dictionary for WordCloud
        word_freq = {kw: score for kw, score in keywords}
        
        wc = WordCloud(
            width=800, 
            height=400, 
            background_color='white',
            colormap='viridis',
            max_words=100
        ).generate_from_frequencies(word_freq)
        
        plt.figure(figsize=(10, 5))
        plt.imshow(wc, interpolation='bilinear')
        plt.axis('off')
        plt.tight_layout(pad=0)
        plt.savefig(output_path)
        plt.close()
    except ImportError:
        print("wordcloud or matplotlib not installed. Skipping word cloud generation.")
    except Exception as e:
        print(f"Error creating word cloud: {e}")
