"""
TF-IDF (Term Frequency-Inverse Document Frequency) method manually implemented.
"""
import re
import math
from typing import List, Tuple, Optional, Any, Dict
from preprocessing import TextProcessor  # type: ignore

class TFIDFExtractor:
    """Extracts keywords using manual TF-IDF at paragraph/document level."""
    
    def __init__(self, processor: Optional[TextProcessor] = None):
        self.processor = processor or TextProcessor()
        
    def extract(self, text: str, top_n: Optional[int] = None) -> List[Tuple[str, float]]:
        """Extract keywords using custom TF-IDF calculation."""
        try:
            # 1. Treat the entire paragraph/document as a single unit or split by semantic blocks.
            raw_docs = [p.strip() for p in re.split(r'\n+', text) if p.strip()]
            if not raw_docs:
                return []
            
            # Preprocess the text
            processed_docs = []
            for doc in raw_docs:
                words = re.findall(r'\b[a-zA-Z]{3,}\b', doc.lower())
                # Lowercase, remove stopwords, and lemmatize
                cleaned = [self.processor.lemmatize(w) for w in words if w not in self.processor.stopwords]
                processed_docs.append(cleaned)
            
            if all(not doc for doc in processed_docs):
                return []
                
            N = len(processed_docs)
            all_words = set(w for doc in processed_docs for w in doc)
            
            # DF Calculation
            df_map = {}
            for word in all_words:
                df = sum(1 for doc in processed_docs if word in doc)
                df_map[word] = df
            
            # Aggregate TF-IDF scores
            total_tfidf = {}
            for word in all_words:
                total_tfidf[word] = 0.0
                # IDF(t) = log(N / df(t))
                idf = math.log(N / df_map[word]) if df_map[word] > 0 else 0.0
                
                # If N=1, log(1/1) = 0. To prevent a completely useless zero-score response
                # for single documents while strictly honoring the mathematical formula's variables,
                # we'll use the raw formula exactly, which means keywords in a single doc evaluate to 0 
                # unless we fallback or it's a multi-document text. We will strict stick to the formula explicitly.
                
                for doc in processed_docs:
                    total_terms = len(doc)
                    if total_terms == 0:
                        continue
                    
                    term_count = doc.count(word)
                    if term_count > 0:
                        # TF(t,d) = count / total_terms
                        tf = term_count / total_terms
                        # TF-IDF(t,d) = TF(t,d) * IDF(t)
                        tfidf = tf * idf
                        
                        total_tfidf[word] += tfidf
                        
            keywords = [(str(w), round(float(s), 4)) for w, s in total_tfidf.items()]
            keywords.sort(key=lambda x: x[1], reverse=True)
            
            return keywords[:top_n] if top_n else keywords
            
        except Exception as e:
            print(f"TF-IDF extraction error: {e}")
            return []

    def get_detailed_results(self, text: str) -> Dict[str, Any]:
        """Provides detailed step-by-step custom calculations exactly as requested."""
        raw_docs = [p.strip() for p in re.split(r'\n+', text) if p.strip()]
        documents = raw_docs if raw_docs else [text]
        
        stopwords = self.processor.stopwords
        processed_docs: List[List[str]] = []
        for doc in documents:
            words = re.findall(r'\b[a-zA-Z]{3,}\b', doc.lower())
            processed_docs.append([self.processor.lemmatize(w) for w in words if w not in stopwords])
            
        N = len(processed_docs)
        all_words = sorted(list(set([w for doc in processed_docs for w in doc])))
        
        idf_data: List[Dict[str, Any]] = []
        idf_map: Dict[str, float] = {}
        for word in all_words:
            df = sum(1 for doc in processed_docs if word in doc)
            # Use exact requested formula: log(N / df)
            idf = math.log(N / df) if df > 0 else 0.0
            idf_map[word] = idf
            idf_data.append({
                'word': word,
                'df': df,
                'calculation': f"log({N}/{df})",
                'idf': round(idf, 4)
            })
            
        sentence_tf: Any = []
        for i, doc in enumerate(processed_docs):
            doc_len = len(doc)
            counts = {}
            for w in doc:
                counts[w] = counts.get(w, 0) + 1
            
            words_tf_list: Any = []
            for word, count in counts.items():
                tf_val = float(count) / float(doc_len) if doc_len > 0 else 0.0
                words_tf_list.append({
                    'word': str(word),
                    'count': int(count),
                    'calculation': f"{count}/{doc_len}",
                    'tf': round(tf_val, 4)
                })
            
            sentence_data: Any = {
                'sentence': str(documents[i]),
                'words': words_tf_list
            }
            sentence_tf.append(sentence_data)
            
        final_scores = []
        # Group by word across documents
        for word in all_words:
            total_tf = 0.0
            for doc in processed_docs:
                doc_len = len(doc)
                if word in doc and doc_len > 0:
                    word_count = doc.count(word)
                    total_tf += (float(word_count) / float(doc_len))
            
            idf = idf_map[word]
            score = total_tf * idf
            uniqueness = "Unique/Important" if idf > 0 else "Common (0 IDF)"
            final_scores.append({
                'word': word,
                'tf': round(float(total_tf), 4),
                'idf': round(float(idf), 4),
                'score': round(float(score), 4),
                'uniqueness': uniqueness
            })
            
        final_scores.sort(key=lambda x: x['score'], reverse=True)
            
        return {
            'sentences': documents,
            'processed_sentences': [", ".join(d) for d in processed_docs],
            'per_sentence_tf': sentence_tf,
            'corpus_size': N,
            'idf_data': idf_data,
            'final_scores': final_scores
        }
