"""
RAKE (Rapid Automatic Keyword Extraction) method.
"""
import re
from typing import List, Tuple, Optional, Set, Any, Dict
from preprocessing import TextProcessor  # type: ignore

class RAKEExtractor:
    """Extracts keywords using the RAKE algorithm."""
    
    def __init__(self, processor: Optional[TextProcessor] = None):
        self.processor = processor or TextProcessor()
    
    def calculate_metrics(self, text: str) -> dict:
        """Calculate word frequencies, degrees, and scores for RAKE."""
        # Clean and split into words
        text_lower = re.sub(r'[^a-zA-Z\s]', ' ', text.lower())
        words = text_lower.split()
        
        # Word counts
        freq: Dict[str, int] = {}
        for w in words:
            if w not in self.processor.stopwords and len(w) > 2:
                freq[w] = freq.get(w, 0) + 1
        
        content_words_list = [w for w in words if w in freq]
        unique_content_words = set(content_words_list)
        
        # Split text into phrases by stopwords
        phrases = []
        current_phrase = []
        stopwords = self.processor.stopwords
        for word in words:
            if word in stopwords or len(word) <= 2:
                if current_phrase:
                    phrases.append(current_phrase)
                    current_phrase = []
            else:
                current_phrase.append(word)
        if current_phrase:
            phrases.append(current_phrase)
        
        degree: Dict[str, int] = {}
        for phrase in phrases:
            unique_words_in_phrase = set(phrase)
            for word in unique_words_in_phrase:
                if word in unique_content_words:
                    co_occurring = [w for w in unique_words_in_phrase if w != word and w in unique_content_words]
                    degree[word] = degree.get(word, 0) + len(co_occurring)
        
        scores: Dict[str, float] = {}
        for word in freq:
            word_freq = freq[word]
            word_degree = degree.get(word, 0)
            scores[word] = float(word_degree) / float(word_freq) if word_freq > 0 else 0.0
        
        return {
            'frequencies': freq,
            'degrees': degree,
            'scores': scores,
            'content_words': content_words_list
        }
    
    def get_phrases(self, text: str) -> List[str]:
        """Split text into phrases using stopwords as delimiters."""
        text_lower = re.sub(r'[^a-zA-Z\s]', ' ', text.lower())
        words = text_lower.split()
        stopwords = self.processor.stopwords
        
        phrases = []
        current_phrase = []
        for word in words:
            if word in stopwords or len(word) <= 2:
                if current_phrase:
                    phrases.append(' '.join(current_phrase))
                    current_phrase = []
            else:
                current_phrase.append(word)  # type: ignore
        
        if current_phrase:
            phrases.append(' '.join(current_phrase))
        
        return [p for p in phrases if len(p.split()) >= 1]

    def extract(self, text: str, top_n: Optional[int] = 10) -> List[Tuple[str, float]]:
        """Extract keywords using RAKE scores."""
        try:
            phrases = self.get_phrases(text)
            metrics = self.calculate_metrics(text)
            
            seen = set()
            result = []
            for phrase in phrases:
                words = phrase.split()
                # Ensure phrases are not excessively long (max 4-5 words)
                if len(words) > 5:
                    continue
                if phrase in seen:
                    continue
                seen.add(phrase)
                phrase_score = sum(metrics['scores'].get(word, 0.0) for word in words)
                result.append((phrase, float(phrase_score)))

            result.sort(key=lambda x: x[1], reverse=True)
            return result[:top_n]
        except Exception as e:
            print(f"RAKE extraction error: {e}")
            return []
