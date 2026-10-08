"""
KeyBERT (Transformer-based) keyword extraction with robust fallback logic.
"""
from typing import List, Tuple, Optional, Any, Dict
from preprocessing import TextProcessor  # type: ignore

class KeyBERTExtractor:
    """KeyBERT extraction with robust fallback logic."""
    
    def __init__(self, processor: Optional[TextProcessor] = None):
        self.processor = processor or TextProcessor()
        self.model = None
        
    def _load_model(self):
        """Lazy load KeyBERT with dependency checks."""
        if self.model is None:
            print("Loading KeyBERT model (all-MiniLM-L6-v2)...")
            try:
                import transformers  # type: ignore
                import sentence_transformers  # type: ignore
                from keybert import KeyBERT  # type: ignore
                self.model = KeyBERT("all-MiniLM-L6-v2")
            except ImportError:
                print("Missing KeyBERT dependencies: transformers or sentence-transformers")
                return None
            except Exception as e:
                print(f"Error loading KeyBERT: {e}")
                return None
        return self.model

    def _format_output(self, keywords: List[Tuple[str, float]], top_n: int) -> List[Tuple[str, float]]:
        """Log the output with rank, keyword, and score."""
        if not keywords:
            return []
            
        keywords = sorted(keywords, key=lambda x: x[1], reverse=True)[:top_n]
        
        # Display the ranked output via print log/debugging mechanism
        print("\n[KeyBERT Extracted Keywords]")
        print(f"{'Rank':<6} | {'Keyword':<30} | {'Score'}")
        print("-" * 50)
        for i, (kw, score) in enumerate(keywords, 1):
            print(f"{i:<6} | {kw:<30} | {score:.4f}")
            
        return [(str(kw), float(score)) for kw, score in keywords]

    def extract(self, text: str, top_n: Optional[int] = 10) -> List[Tuple[str, float]]:
        """Extract keywords using robust parameters and fallbacks."""
        if not text or not text.strip():
            print("Validation Error: Input text is empty. Please provide meaningful text.")
            return [("No valid text provided", 0.0)]
            
        model = self._load_model()
        if model is None:
            return [("KeyBERT Model failed to load", 0.0)]
            
        extract_n = top_n if top_n and top_n > 0 else 10
        try:
            # 1st attempt: robust parameters with up to trigrams
            keywords = model.extract_keywords(
                text,
                keyphrase_ngram_range=(1, 3),
                stop_words='english',
                top_n=max(10, extract_n * 2) 
            )
            
            # If nothing returned, fallback 1: remove english stop_words requirement (helps short text)
            if not keywords:
                print("Retrieval yielded empty results. Attempting Fallback 1: No Stopwords filter.")
                keywords = model.extract_keywords(
                    text,
                    keyphrase_ngram_range=(1, 3),
                    stop_words=None,
                    top_n=extract_n * 2
                )
                
            # If still nothing returned, fallback 2: single-grams only
            if not keywords:
                print("Retrieval yielded empty results. Attempting Fallback 2: Unigrams only.")
                keywords = model.extract_keywords(
                    text,
                    keyphrase_ngram_range=(1, 1),
                    stop_words=None,
                    top_n=extract_n * 2
                )

            # Final validation check
            if not keywords:
                print("Warning: KeyBERT could not extract any keywords from this text.")
                return [("No meaningful keywords found", 0.0)]
            
            # Format and return the required table mapping (ensuring list tuple structure)
            return self._format_output(keywords, extract_n)
                
        except Exception as e:
            print(f"KeyBERT extraction error: {e}")
            return [("Error during extraction", 0.0)]
            
    def get_detailed_report(self) -> Dict[str, Any]:
        """Returns placeholders for UI step display."""
        return {
            'step1': "Convert entire text to a vector representation using all-MiniLM-L6-v2",
            'step2': "Extract candidate n-grams (unigrams, bigrams, trigrams) from text",
            'step3': "Convert each candidate keyword into vector space representation",
            'step4': "Rank tokens by comparing keyword vectors vs document vector utilizing Cosine Similarity"
        }
