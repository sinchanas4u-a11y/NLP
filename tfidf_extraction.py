"""
TF-IDF (Term Frequency-Inverse Document Frequency) method with dynamic reference corpus.
"""
import re
import math
from typing import List, Tuple, Optional, Any, Dict
from preprocessing import TextProcessor  # type: ignore

# Fixed reference background corpus (10 domain-diverse reference documents)
REFERENCE_CORPUS = [
    "Machine learning algorithms and artificial intelligence systems are transforming healthcare data and clinical research.",
    "Deep learning models assist medical specialists in medical image analysis and automated diagnostic detection.",
    "Predictive diagnosis tools evaluate patient health records to forecast disease progression and outcomes.",
    "Artificial intelligence applications are transforming clinical workflows and robotic surgery across modern hospitals.",
    "Computer vision and digital image processing enhance medical image evaluation and diagnosis accuracy.",
    "Healthcare organizations leverage machine learning and predictive intelligence for preventative patient care.",
    "Machine learning and artificial intelligence technologies are driving modern scientific discovery.",
    "Intelligent systems utilize machine learning frameworks to solve complex pattern recognition tasks.",
    "Climate change and environmental degradation are increasing global temperatures and altering ecological systems.",
    "Extreme weather events including severe droughts, recurrent floods, and destructive wildfires impact communities worldwide."
]

class TFIDFExtractor:
    """Extracts keywords using manual TF-IDF with a dynamic background corpus."""
    
    def __init__(self, processor: Optional[TextProcessor] = None):
        self.processor = processor or TextProcessor()
        self.reference_corpus = list(REFERENCE_CORPUS)
        
    def _extract_content_tokens(self, text: str) -> List[str]:
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        tokens: List[str] = []
        for w in words:
            if w not in self.processor.stopwords:
                tokens.append(self.processor.lemmatize(w))
        return tokens

    def extract(self, text: str, top_n: Optional[int] = None) -> List[Tuple[str, float]]:
        """Extract keywords using dynamic corpus TF-IDF."""
        try:
            user_tokens = self._extract_content_tokens(text)
            if not user_tokens:
                return []
                
            dynamic_corpus = list(self.reference_corpus) + [text]
            N = len(dynamic_corpus)
            
            # Preprocess all documents in the dynamic corpus
            processed_corpus = [self._extract_content_tokens(doc) for doc in dynamic_corpus]
            
            # Calculate DF for each word in user text
            total_content_tokens = len(user_tokens)
            user_word_counts: Dict[str, int] = {}
            for w in user_tokens:
                user_word_counts[w] = user_word_counts.get(w, 0) + 1
                
            scored_keywords: List[Tuple[str, float]] = []
            for word, count in user_word_counts.items():
                # TF = count in user text / total content tokens in user text
                tf = float(count) / float(total_content_tokens) if total_content_tokens > 0 else 0.0
                
                # DF = number of docs in dynamic corpus containing word
                df = sum(1 for doc in processed_corpus if word in doc)
                
                # Smoothed IDF: ln((1 + N) / (1 + DF)) + 1
                idf = math.log((1.0 + N) / (1.0 + df)) + 1.0
                
                # TF-IDF
                tfidf = tf * idf
                scored_keywords.append((word, round(float(tfidf), 4)))
                
            scored_keywords.sort(key=lambda x: (x[1], x[0]), reverse=True)
            return scored_keywords[:top_n] if top_n else scored_keywords
        except Exception as e:
            print(f"TF-IDF extraction error: {e}")
            return []

    def get_detailed_results(self, text: str) -> Dict[str, Any]:
        """Provides detailed step-by-step calculations with dynamic corpus."""
        user_tokens = self._extract_content_tokens(text)
        
        dynamic_corpus = list(self.reference_corpus) + [text]
        N = len(dynamic_corpus)
        processed_corpus = [self._extract_content_tokens(doc) for doc in dynamic_corpus]
        
        # User document index is N - 1
        user_index = N - 1
        user_doc_tokens = processed_corpus[user_index]
        total_content_tokens = len(user_doc_tokens)
        
        # User word counts
        user_counts: Dict[str, int] = {}
        for w in user_doc_tokens:
            user_counts[w] = user_counts.get(w, 0) + 1
            
        unique_user_words = sorted(list(user_counts.keys()))
        
        # Step 4: IDF calculation across dynamic corpus
        idf_data: List[Dict[str, Any]] = []
        idf_map: Dict[str, float] = {}
        for word in unique_user_words:
            df = sum(1 for doc in processed_corpus if word in doc)
            idf = math.log((1.0 + N) / (1.0 + df)) + 1.0
            idf_map[word] = idf
            idf_data.append({
                'word': word,
                'df': df,
                'calculation': f"log((1+{N})/(1+{df})) + 1",
                'idf': round(idf, 4)
            })
            
        # Step 3: TF for user input
        words_tf_list: List[Dict[str, Any]] = []
        for word in unique_user_words:
            count = user_counts[word]
            tf_val = float(count) / float(total_content_tokens) if total_content_tokens > 0 else 0.0
            words_tf_list.append({
                'word': word,
                'count': count,
                'calculation': f"{count}/{total_content_tokens}",
                'tf': round(tf_val, 4)
            })
            
        sentence_tf = [{
            'sentence': text,
            'words': words_tf_list
        }]
        
        # Step 5: TF-IDF scores
        final_scores: List[Dict[str, Any]] = []
        for word in unique_user_words:
            count = user_counts[word]
            tf_val = float(count) / float(total_content_tokens) if total_content_tokens > 0 else 0.0
            idf = idf_map[word]
            score = tf_val * idf
            uniqueness = "Unique/High IDF" if idf >= 2.0 else "Common in Corpus"
            final_scores.append({
                'word': word,
                'tf': round(float(tf_val), 4),
                'idf': round(float(idf), 4),
                'score': round(float(score), 4),
                'uniqueness': uniqueness
            })
            
        final_scores.sort(key=lambda x: (x['score'], x['word']), reverse=True)
        
        # Return format expected by app.py and script.js
        return {
            'strategy': f"Dynamic Corpus (10 Reference Docs + 1 User Input Doc, N={N})",
            'sentences': [f"User Input: \"{text}\""],
            'processed_sentences': [", ".join(user_doc_tokens)],
            'per_sentence_tf': sentence_tf,
            'corpus_size': N,
            'idf_data': idf_data,
            'final_scores': final_scores
        }
