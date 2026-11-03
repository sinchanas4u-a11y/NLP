"""
RAKE (Rapid Automatic Keyword Extraction) method.
Splits phrases using both stopwords and punctuation marks as delimiters.
"""
import re
from collections import Counter
from typing import List, Tuple, Optional, Set, Any, Dict
from preprocessing import TextProcessor  # type: ignore

class RAKEExtractor:
    """Extracts keywords using the RAKE algorithm with punctuation-aware phrase splitting."""
    
    def __init__(self, processor: Optional[TextProcessor] = None):
        self.processor = processor or TextProcessor()
    
    def get_phrases(self, text: str) -> List[str]:
        """Split text into candidate phrases using both stopwords and punctuation as delimiters."""
        if not text or not text.strip():
            return []
        
        stopwords = self.processor.stopwords
        tokens = re.findall(r'[a-zA-Z]+|[^a-zA-Z\s]', text)
        phrases = []
        current_phrase = []

        for token in tokens:
            if re.match(r'^[a-zA-Z]+$', token):
                word = token.lower()
                if word in stopwords or len(word) <= 2:
                    if current_phrase:
                        phrases.append(' '.join(current_phrase))
                        current_phrase = []
                else:
                    current_phrase.append(word)
            else:
                if current_phrase:
                    phrases.append(' '.join(current_phrase))
                    current_phrase = []

        if current_phrase:
            phrases.append(' '.join(current_phrase))

        return [p.strip() for p in phrases if p.strip()]

    def calculate_metrics(self, text: str) -> dict:
        """Calculate word frequencies, degrees, and scores for RAKE."""
        if not text or not text.strip():
            return {'frequencies': {}, 'degrees': {}, 'scores': {}, 'content_words': []}

        phrases = self.get_phrases(text)
        phrase_words_list = [p.split() for p in phrases]
        content_words_list = [w for p in phrase_words_list for w in p]
        unique_content_words = set(content_words_list)
        freq = Counter(content_words_list)

        degree = Counter()
        for p_words in phrase_words_list:
            unique_in_phrase = list(set(p_words))
            for word in unique_in_phrase:
                if word in unique_content_words:
                    co_occurring = [w for w in unique_in_phrase if w != word and w in unique_content_words]
                    degree[word] += len(co_occurring)

        scores: Dict[str, float] = {}
        for word in freq:
            scores[word] = float(degree[word]) / float(freq[word]) if freq[word] > 0 else 0.0

        return {
            'frequencies': dict(freq),
            'degrees': dict(degree),
            'scores': scores,
            'content_words': content_words_list
        }

    def extract(self, text: str, top_n: Optional[int] = 10) -> List[Tuple[str, float]]:
        """Extract keywords using RAKE scores."""
        try:
            phrases = self.get_phrases(text)
            if not phrases:
                return []
            metrics = self.calculate_metrics(text)
            
            seen = set()
            result = []
            for phrase in phrases:
                if phrase in seen:
                    continue
                seen.add(phrase)
                words = phrase.split()
                if len(words) > 5:
                    continue
                phrase_score = sum(metrics['scores'].get(word, 0.0) for word in words)
                result.append((phrase, float(phrase_score)))

            result.sort(key=lambda x: (x[1], len(x[0].split())), reverse=True)
            return result[:top_n] if top_n else result
        except Exception as e:
            print(f"RAKE extraction error: {e}")
            return []
