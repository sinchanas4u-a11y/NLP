"""
Interactive Keyword Extractor
A command-line program that accepts user text input and extracts keywords using NLP.
"""

import re
import string
from typing import List, Tuple, Optional, Set, Any, Dict
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
    
    def __init__(self, custom_stopwords: Optional[Set[str]] = None, domain: str = 'general'):
        """
        Initialize text processor.
        
        Args:
            custom_stopwords: Additional stopwords to include
            domain: Domain context ('general', 'technical', 'product', etc.)
        """
        self.domain = domain
        self.stopwords = self._load_stopwords(custom_stopwords)
    
    def _load_stopwords(self, custom_stopwords: Optional[Set[str]] = None) -> Set[str]:
        """Load English stopwords with optional custom additions."""
        try:
            from nltk.corpus import stopwords  # type: ignore
            base_stopwords = set(stopwords.words('english'))
        except:
            # Fallback stopwords
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

        # Always treat these as separators in RAKE phrase splitting.
        base_stopwords.update({'used', 'applied'})
        
        # Domain-specific stopwords
        domain_stopwords = {
            'technical': {'using', 'use', 'method', 'methods', 'approach', 'approaches'},
            'product': set(),
            'general': set()
        }
        
        # Add domain-specific stopwords
        if self.domain in domain_stopwords:
            base_stopwords.update(domain_stopwords[self.domain])
        
        # Add custom stopwords
        if custom_stopwords:
            base_stopwords.update(custom_stopwords)
        
        return base_stopwords
    
    def add_stopwords(self, words: set):
        """Add custom stopwords dynamically."""
        self.stopwords.update(words)
    
    def remove_stopwords(self, words: set):
        """Remove words from stopwords (treat as keywords)."""
        self.stopwords.difference_update(words)
    
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
        Ensures 'models' remains 'models'.
        """
        try:
            from nltk.stem import WordNetLemmatizer  # type: ignore
            try:
                nltk.data.find('corpora/wordnet')
            except LookupError:
                nltk.download('wordnet', quiet=True)
            
            lemmatizer = WordNetLemmatizer()
            words = text.split()
            lemmatized = []
            for word in words:
                word_lower = word.lower()
                if word_lower == 'models':
                    lemmatized.append(word)
                else:
                    lemmatized.append(lemmatizer.lemmatize(word))
            return ' '.join(lemmatized)
        except Exception as e:
            print(f"Lemmatization error: {e}")
            return text
    
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
    
    def extract_tfidf(self, text: str, top_n: Optional[int] = None) -> List[Tuple[str, float]]:
        """
        Extract keywords using TF-IDF with proper per-document TF calculation.
        
        Args:
            text: Input text (single document or multiple documents separated by newlines)
            top_n: Number of top keywords to return (None = return all)
            
        Returns:
            List of (keyword, score) tuples
        """
        from collections import Counter
        import math
        import re

        try:
            # Parse documents - treat paragraphs as documents instead of sentences
            raw_docs = [p.strip() for p in re.split(r'\n+', text) if p.strip()]
            documents = raw_docs if raw_docs else [text]
            
            if len(documents) == 0:
                return []
            
            # Remove common stopwords and apply lemmatization
            stopwords = self.processor.stopwords
            
            processed_docs: List[List[str]] = []
            for doc in documents:
                words: List[str] = re.findall(r'\b[a-zA-Z]{3,}\b', doc.lower())
                content_words: List[str] = []
                for w in words:
                    if w not in stopwords:
                        lemmatized_w = self.processor.lemmatize(w)
                        content_words.append(lemmatized_w)
                processed_docs.append(content_words)
            
            N = len(processed_docs)
            all_words = set()
            for doc in processed_docs:
                all_words.update(doc)
            
            def calculate_idf(term):
                """IDF(t, D) = log(Total documents / Number of documents containing term t)"""
                df = sum(1 for doc in processed_docs if term in doc)
                if df == 0:
                    return 0.0
                return math.log(float(N) / float(df))
            
            word_scores = {}
            for word in all_words:
                total_tfidf = 0.0
                idf = float(calculate_idf(word))
                
                for doc in processed_docs:
                    total_terms = len(doc)
                    if word in doc and total_terms > 0:
                        word_count = doc.count(word)
                        # TF(t, d) = Number of times term t appears in document d / Total number of terms in document d
                        tf = float(word_count) / float(total_terms)
                        # TF-IDF(t, d, D) = TF(t, d) × IDF(t, D)
                        tfidf = tf * idf
                        total_tfidf += tfidf
                
                word_scores[word] = float(total_tfidf)
            
            keywords = sorted(word_scores.items(), key=lambda x: x[1], reverse=True)
            result: List[Tuple[str, float]] = [(str(word), float(round(float(score), 4))) for word, score in keywords]
            
            if top_n is not None:
                return result[:int(top_n)]
            return result
            
        except Exception as e:
            print(f"TF-IDF extraction error: {e}")
            return []
    
    def extract_tfidf_detailed(self, documents: List[str]) -> Dict[str, Any]:
        """
        Extract keywords using TF-IDF with detailed calculation steps for multiple documents.
        
        Args:
            documents: List of document strings
            
        Returns:
            Dictionary with detailed calculation steps and results per document
        """
        from collections import Counter
        import math
        import re
        
        try:
            # Parse documents safely
            stopwords = self.processor.stopwords
            
            processed_docs: List[List[str]] = []
            doc_tokens: List[List[str]] = []
            for doc in documents:
                words: List[str] = re.findall(r'\b[a-zA-Z]{3,}\b', doc.lower())
                doc_tokens.append(words)
                content_words: List[str] = []
                for w in words:
                    if w not in stopwords:
                        lemmatized_w = self.processor.lemmatize(w)
                        content_words.append(lemmatized_w)
                processed_docs.append(content_words)
            
            N = len(processed_docs)
            all_words = set()
            for doc in processed_docs:
                all_words.update(doc)
            
            def calculate_idf(term):
                """IDF(t, D) = log(Total documents / Number of documents containing term t)"""
                df = sum(1 for doc in processed_docs if term in doc)
                if df == 0:
                    return 0.0
                return math.log(float(N) / float(df))
            
            doc_results = []
            for i, (doc_content, tokens) in enumerate(zip(processed_docs, doc_tokens)):
                word_counts = Counter(doc_content)
                total_words_in_doc = len(doc_content)
                
                tf_data = {}
                for word, count in word_counts.items():
                    # TF(t, d) = Number of times term t appears in document d / Total number of terms in document d
                    tf = float(count) / float(total_words_in_doc) if total_words_in_doc > 0 else 0.0
                    tf_data[word] = {'count': int(count), 'tf': float(tf), 'total': int(total_words_in_doc)}
                
                tfidf_data = []
                for word in word_counts.keys():
                    tf = tf_data[word]['tf']
                    idf = calculate_idf(word)
                    # TF-IDF(t, d, D) = TF(t, d) × IDF(t, D)
                    tfidf = tf * idf
                    tfidf_data.append({
                        'word': str(word),
                        'tf': float(tf),
                        'idf': float(idf),
                        'tfidf': float(tfidf)
                    })
                
                tfidf_data.sort(key=lambda x: x['tfidf'], reverse=True)
                
                doc_results.append({
                    'doc_id': i + 1,
                    'original': documents[i],
                    'tokens': tokens,
                    'content_words': doc_content,
                    'total_words': total_words_in_doc,
                    'tf_data': tf_data,
                    'tfidf_data': tfidf_data
                })
            
            idf_data = []
            for word in sorted(all_words):
                df = sum(1 for doc in processed_docs if word in doc)
                idf = calculate_idf(word)
                idf_data.append({
                    'word': str(word),
                    'df': int(df),
                    'idf': float(idf)
                })
            
            return {
                'total_documents': int(N),
                'documents': doc_results,
                'idf_data': idf_data
            }
            
        except Exception as e:
            print(f"TF-IDF detailed extraction error: {e}")
            return {}
    
    def extract_rake(self, text: str, top_n: Optional[int] = None) -> List[Tuple[str, float]]:
        """
        Extract keywords using RAKE.
        
        Args:
            text: Original text (RAKE works better with original text)
            top_n: Ignored for RAKE. All phrase keywords are returned.
            
        Returns:
            List of (keyword, score) tuples
        """
        try:
            # Build phrases using current stopword configuration.
            phrases = get_phrases_split_by_stopwords(text, self.processor.stopwords)
            metrics = calculate_rake_metrics(text, self.processor.stopwords)

            # Method 2 phrase score: sum of per-word (degree / frequency) scores.
            result = []
            for phrase in phrases:
                words = phrase.split()
                phrase_score = sum(metrics['scores'].get(word, 0.0) for word in words)
                result.append((phrase, float(phrase_score)))

            # Rank phrases by score but keep all phrases (no top_n limit).
            result.sort(key=lambda x: x[1], reverse=True)
            return result
        except Exception as e:
            print(f"RAKE extraction error: {e}")
            return []
    
    def extract_keybert(self, text: str, top_n: Optional[int] = None) -> List[Tuple[str, float]]:
        """
        Extract keywords using KeyBERT.
        
        Args:
            text: Original text
            top_n: Number of top keywords to return (None = auto-detect all meaningful)
            
        Returns:
            List of (keyword, score) tuples
        """
        try:
            if self.keybert_model is None:
                print("Loading KeyBERT model (first time only)...")
                try:
                    # Explicitly check for dependencies that often cause 'PreTrainedModel' error
                    try:
                        import transformers
                        import sentence_transformers
                    except ImportError:
                        print("Missing KeyBERT dependencies: transformers or sentence-transformers")
                        return []
                        
                    from keybert import KeyBERT  # type: ignore
                    self.keybert_model = KeyBERT()
                except Exception as e:
                    print(f"Error loading KeyBERT: {e}")
                    print("Hint: Try 'pip install transformers sentence-transformers --upgrade'")
                    return []
            
            if self.keybert_model is None:
                return []
            
            # If no limit specified, extract more and filter by score
            extract_n = top_n if top_n else 50
            
            keywords = self.keybert_model.extract_keywords(  # type: ignore
                text,
                keyphrase_ngram_range=(1, 2),
                stop_words='english',
                top_n=extract_n,
                use_mmr=True,
                diversity=0.7
            )
            
            result = [(kw, float(score)) for kw, score in keywords if score > 0.1]
            
            if top_n:
                return result[:top_n]  # type: ignore
            return result
        except Exception as e:
            print(f"KeyBERT extraction error: {e}")
            return []
    
    def extract(self, text: str, top_n: Optional[int] = None) -> List[Tuple[str, float]]:
        """
        Extract keywords using the selected method.
        
        Args:
            text: Input text
            top_n: Number of top keywords to return (None = extract all meaningful)
            
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


def find_stopwords_in_text(text: str, stopwords: set) -> Tuple[List[str], List[str], int]:
    """
    Find all stopwords in the text.
    
    Args:
        text: Input text
        stopwords: Set of stopwords
        
    Returns:
        Tuple of (all_stopwords, unique_stopwords, total_count)
    """
    # Clean and tokenize
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ' ', text)
    text = re.sub(r'\S+@\S+', ' ', text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    words = text.split()
    
    # Find stopwords
    all_stopwords = []
    for word in words:
        word_lower = word.lower()
        if word_lower in stopwords:
            all_stopwords.append(word_lower) # type: ignore
    
    unique_stopwords = sorted(list(set(all_stopwords)))
    
    return all_stopwords, unique_stopwords, len(all_stopwords)


def calculate_rake_metrics(text: str, stopwords: set) -> dict:
    """
    Calculate RAKE metrics using co-occurrence excluding itself for degree.
    
    Args:
        text: Input text
        stopwords: Set of stopwords
        
    Returns:
        Dictionary with word frequencies, degrees, and scores
    """
    # Clean and split into words
    text = re.sub(r'[^a-zA-Z\s]', ' ', text.lower())
    words = text.split()
    
    # Filter out stopwords to get content words
    content_words_list = [w for w in words if w not in stopwords and len(w) > 2]
    unique_content_words = list(set(content_words_list))
    
    # Calculate word frequency
    freq = Counter(content_words_list)
    
    # Build co-occurrence matrix (words co-occur if they appear in the same phrase/sentence)
    # Split text into phrases by stopwords
    phrases = []
    current_phrase = []
    for word in words:
        if word in stopwords or len(word) <= 2:
            if current_phrase:
                phrases.append(current_phrase)
                current_phrase = []
        else:
            current_phrase.append(word)  # type: ignore
    if current_phrase:
        phrases.append(current_phrase)
    
    # Calculate word degree using co-occurrence (excluding itself)
    # Degree = sum of co-occurrences with other words (not including itself)
    degree = Counter()
    
    for phrase in phrases:
        unique_words_in_phrase = list(set(phrase))
        for word in unique_words_in_phrase:
            if word in unique_content_words:
                # Co-occurrence with other words in the same phrase (excluding itself)
                co_occurring = [w for w in unique_words_in_phrase if w != word and w in unique_content_words]
                degree[word] += len(co_occurring)  # type: ignore
    
    # Calculate word score (degree / frequency)
    scores = {}
    for word in freq:
        scores[word] = degree[word] / freq[word] if freq[word] > 0 else 0
    
    return {
        'frequencies': freq,
        'degrees': degree,
        'scores': scores,
        'content_words': content_words_list
    }


def get_phrases_split_by_stopwords(text: str, stopwords: set) -> List[str]:
    """
    Split text into phrases using stopwords as delimiters.
    
    Args:
        text: Input text
        stopwords: Set of stopwords
        
    Returns:
        List of phrases
    """
    # Clean text
    text = re.sub(r'[^a-zA-Z\s]', ' ', text.lower())
    words = text.split()
    
    phrases = []
    current_phrase = []
    
    for word in words:
        if word in stopwords or len(word) <= 2:
            # End current phrase if we have words
            if current_phrase:
                phrases.append(' '.join(current_phrase))
                current_phrase = []
        else:
            current_phrase.append(word)  # type: ignore
    
    # Don't forget the last phrase
    if current_phrase:
        phrases.append(' '.join(current_phrase))
    
    # Filter out empty phrases and single words (keep only meaningful phrases)
    phrases = [p for p in phrases if len(p.split()) >= 1]
    
    return phrases


def display_combined_output(keywords: List[Tuple[str, float]], method: str, text: str, stopwords: set):
    """Display extracted keywords, stopwords, and RAKE calculation in expected format."""
    print("\n" + "=" * 50)
    print("         KEYWORD EXTRACTION RESULT")
    print("=" * 50)
    
    # Find stopwords in text
    all_stopwords, unique_stopwords, total_stopwords = find_stopwords_in_text(text, stopwords)
    
    # Section 1: Stopwords Found
    print("\n\nStopwords Found in Text:")
    print(", ".join(unique_stopwords))
    print(f"\nTotal Stopwords : {len(unique_stopwords)}")
    
    # Section 2: Phrases Identified (only for RAKE)
    if method.lower() == 'rake':
        phrases = get_phrases_split_by_stopwords(text, stopwords)
        
        print("\n" + "-" * 50)
        print("\nPhrases Identified (after splitting by stopwords):")
        for i, phrase in enumerate(phrases, 1):
            if phrase.strip():  # Only show non-empty phrases
                print(f"{i}. {phrase}")
    
    # Section 3: RAKE Score Calculation (only for RAKE)
    if method.lower() == 'rake':
        metrics = calculate_rake_metrics(text, stopwords)
        
        print("\n" + "-" * 50)
        print("\nRAKE Score Calculation:")
        
        print("\nWord Scores (Degree / Frequency):")
        for word in sorted(metrics['scores'].keys()):
            score = metrics['scores'][word]
            freq = metrics['frequencies'][word]
            deg = metrics['degrees'][word]
            print(f"{word:<12} = {deg} / {freq} = {score:.1f}")
        
        print("\nPhrase Score Calculation (Sum of Word Scores):")
        for phrase, score in keywords:
            words = phrase.split()
            word_scores = [metrics['scores'].get(w, 0) for w in words]
            score_sum = sum(word_scores)
            calculation = " + ".join([f"{s:.1f}" for s in word_scores])
            print(f'"{phrase}"  = {calculation} = {score_sum:.1f}')
    
    # Section 4: Final Extracted Keywords
    print("\n" + "-" * 50)
    print("\nFinal Extracted Keywords:")
    for i, (keyword, score) in enumerate(keywords, 1):
        print(f"{i}. {keyword:<25} score: {score:.1f}")
    
    print("\n" + "=" * 50)


def display_all_methods_calculations(text: str, documents: Optional[List[str]] = None):
    """
    Display detailed step-by-step calculations for all three methods.
    
    Args:
        text: Input text (single document or multiple documents joined)
        documents: Optional list of documents for multi-document TF-IDF
    """
    print("\n" + "=" * 70)
    print("         DETAILED STEP-BY-STEP CALCULATIONS")
    print("=" * 70)
    
    # Initialize extractors
    rake_extractor = KeywordExtractor(method='rake')
    tfidf_extractor = KeywordExtractor(method='tfidf')
    keybert_extractor = KeywordExtractor(method='keybert')
    
    stopwords = rake_extractor.processor.stopwords
    
    # Parse documents for TF-IDF - split by sentences ('.' or '\n')
    if documents is None:
        raw_docs = re.split(r'[.\n]+', text)
        documents = [doc.strip() for doc in raw_docs if doc.strip()]
        if len(documents) == 0:
            documents = [text]
    
    # Use first document for RAKE and KeyBERT
    first_doc = documents[0] if documents else text
    
    # Extract TF-IDF keywords for comparison at the end
    tfidf_keywords = tfidf_extractor.extract(text)
    
    # ============================================
    # RAKE CALCULATIONS
    # ============================================
    print("\n" + "=" * 70)
    print("RAKE (Rapid Automatic Keyword Extraction)")
    print("=" * 70)
    
    # Step 1: Identify stopwords
    all_stopwords, unique_stopwords, total_stopwords = find_stopwords_in_text(first_doc, stopwords)
    print("\nStep 1: Identify Stopwords")
    print("-" * 50)
    print(f"Stopwords found: {', '.join(unique_stopwords)}")
    print(f"Total unique stopwords: {len(unique_stopwords)}")
    
    # Step 2: Split into phrases
    phrases = get_phrases_split_by_stopwords(first_doc, stopwords)
    print("\nStep 2: Split Text into Phrases (by stopwords)")
    print("-" * 50)
    for i, phrase in enumerate(phrases, 1):
        if phrase.strip():
            print(f"  Phrase {i}: \"{phrase}\"")
    
    # Step 3: Calculate metrics
    rake_metrics = calculate_rake_metrics(first_doc, stopwords)
    print("\nStep 3: Calculate Word Frequency")
    print("-" * 50)
    print(f"{'Word':<15} {'Frequency'}")
    for word, freq in sorted(rake_metrics['frequencies'].items()):
        print(f"  {word:<15} {freq}")
    
    print("\nStep 4: Calculate Word Degree (Co-occurrence excluding itself)")
    print("-" * 50)
    print(f"{'Word':<15} {'Degree'}  {'Explanation'}")
    for word, deg in sorted(rake_metrics['degrees'].items()):
        # Find which phrases contain this word
        containing_phrases = [p for p in phrases if word in p.split()]
        co_occurring = set()
        for phrase in containing_phrases:
            words_in_phrase = phrase.split()
            co_occurring.update([w for w in words_in_phrase if w != word and w in rake_metrics['frequencies']])
        explanation = f"co-occurs with: {', '.join(co_occurring)}" if co_occurring else "no co-occurrence"
        print(f"  {word:<15} {deg:<7} {explanation}")
    
    print("\nStep 5: Calculate Word Score (Degree / Frequency)")
    print("-" * 50)
    print(f"{'Word':<15} {'Calculation':<20} {'Score'}")
    for word in sorted(rake_metrics['scores'].keys()):
        score = rake_metrics['scores'][word]
        freq = rake_metrics['frequencies'][word]
        deg = rake_metrics['degrees'][word]
        print(f"  {word:<15} {deg} / {freq} = {score:.2f}")
    
    # Step 6: Calculate phrase scores
    rake_keywords = rake_extractor.extract(first_doc)
    print("\nStep 6: Calculate Phrase Score (Sum of Word Scores)")
    print("-" * 50)
    for phrase, score in rake_keywords:
        words = phrase.split()
        word_scores = [rake_metrics['scores'].get(w, 0) for w in words]
        calculation = " + ".join([f"{s:.1f}" for s in word_scores])
        total = sum(word_scores)
        print(f'  "{phrase}"')
        print(f'    = {calculation} = {total:.1f}')
    
    print("\nRAKE Final Results:")
    print("-" * 50)
    for i, (keyword, score) in enumerate(rake_keywords, 1):
        print(f"  {i}. {keyword:<25} score: {score:.1f}")
    
    # ============================================
    # TF-IDF CALCULATIONS (Multi-Document)
    # ============================================
    print("\n" + "=" * 70)
    print("TF-IDF (Term Frequency-Inverse Document Frequency)")
    print("=" * 70)
    
    print("\nINPUT DOCUMENTS:")
    for i, doc in enumerate(documents, 1):
        print(f"Document {i}: \"{doc}\"")
    
    from collections import Counter
    import math
    
    # Process all documents using the new logic
    processed_docs = []
    doc_tokens = []
    for doc in documents:
        # Lowercase and find words (min 3 chars)
        words = re.findall(r'\b[a-zA-Z]{3,}\b', doc.lower())
        doc_tokens.append(words)
        # Apply lemmatization with "models" exception
        content_words = []
        for w in words:
            if w not in stopwords:
                lemmatized_w = tfidf_extractor.processor.lemmatize(w)
                content_words.append(lemmatized_w)
        processed_docs.append(content_words)
    
    print("\n" + "-" * 60)
    print("STEP 1 - STOPWORDS REMOVED & LEMMATIZED:")
    print("-" * 60)
    for i, doc in enumerate(processed_docs, 1):
        print(f"Document {i}: {', '.join(doc)}")
    
    # Calculate TF for each document
    print("\n" + "-" * 60)
    print("STEP 2 - TF CALCULATION (Corrected):")
    print("(TF = Word Count / Total Words in that Document)")
    print("-" * 60)
    
    doc_tf_data = []
    for i, doc in enumerate(processed_docs, 1):
        word_counts = Counter(doc)
        total_words = len(doc)  # type: ignore
        print(f"\nDocument {i} (Total Words = {total_words}):")
        tf_data = {}
        for word, count in word_counts.items():  # type: ignore
            tf = float(count) / float(total_words) if total_words > 0 else 0.0
            tf_data[word] = tf
            print(f"  {word:<12} = {count}/{total_words} = {tf:.3f}")
        doc_tf_data.append(tf_data)
    
    # Calculate IDF
    N = len(processed_docs)
    all_words = set()
    for doc in processed_docs:
        all_words.update(doc)
    
    print("\n" + "-" * 60)
    print("STEP 3 - IDF CALCULATION (Corrected):")
    print(f"(IDF = log(Total Documents / Documents containing the word))")
    print(f"Total Documents = {N}")
    print("-" * 60)
    print(f"{'Word':<12} {'In Docs':<10} {'IDF Calculation':<25} {'IDF Score'}")
    
    idf_values = {}
    for word in sorted(all_words):
        df = sum(1 for doc in processed_docs if word in doc)
        idf = math.log(float(N) / float(df))
        idf_values[word] = idf
        idx_calc = f"log({N}/{df})"
        print(f"  {word:<12} {df:<10} {idx_calc:<25} {idf:.3f}")
    
    # Calculate TF-IDF for each document
    print("\n" + "-" * 60)
    print("STEP 4 - TF-IDF SCORE CALCULATION (Corrected):")
    print("(TF-IDF = TF x IDF)")
    print("-" * 60)
    
    doc_tfidf_results = []
    for i, (doc, tf_data) in enumerate(zip(processed_docs, doc_tf_data), 1):
        print(f"\nDocument {i}:")
        results = []
        for word in tf_data.keys():
            tf = float(tf_data[word])
            idf = float(idf_values[word])
            tfidf = tf * idf
            results.append((word, tfidf, tf, idf))
            print(f"  {word:<12} = {tf:.3f} x {idf:.3f} = {tfidf:.3f}")
        
        # Sort by TF-IDF score
        results.sort(key=lambda x: x[1], reverse=True)
        doc_tfidf_results.append(results)
    
    # Final results
    print("\n" + "-" * 60)
    print("STEP 5 - FINAL KEYWORDS PER DOCUMENT:")
    print("(Sorted by TF-IDF Score - Highest First)")
    print("-" * 60)
    
    for i, results in enumerate(doc_tfidf_results, 1):
        print(f"\nDocument {i} Keywords:")
        for j, (word, score, tf, idf) in enumerate(results, 1):
            uniqueness = "unique to this doc - high score" if idf > 0.4 else "common across docs - low score"
            print(f"  {j}. {word:<12} score: {score:.3f}  ({uniqueness})")
    
    print("\n" + "=" * 70)
    print("KEY OBSERVATION:")
    print("- Words unique to one document  -> HIGH TF-IDF score")
    print("- Words common across documents -> LOW TF-IDF score")
    print("=" * 70)
    
    # ============================================
    # KeyBERT CALCULATIONS
    # ============================================
    print("\n" + "=" * 70)
    print("KeyBERT (Transformer-based Semantic Extraction)")
    print("=" * 70)
    
    print("\nStep 1: Generate Document Embedding")
    print("-" * 50)
    print("  Using pre-trained BERT model (all-MiniLM-L6-v2)")
    print("  Convert entire text to a vector representation")
    
    print("\nStep 2: Generate Candidate Keywords")
    print("-" * 50)
    print("  Extract n-grams (unigrams and bigrams) from text")
    print("  Filter out stopwords")
    
    print("\nStep 3: Generate Keyword Embeddings")
    print("-" * 50)
    print("  Convert each candidate keyword to vector representation")
    
    print("\nStep 4: Calculate Cosine Similarity")
    print("-" * 50)
    print("  Compare each keyword vector with document vector")
    print("  Score = cosine_similarity(keyword, document)")
    
    keybert_keywords = keybert_extractor.extract(first_doc)
    print("\nKeyBERT Similarity Scores:")
    print("-" * 50)
    print(f"{'Keyword':<25} {'Similarity Score'}")
    if keybert_keywords:
        for keyword, score in keybert_keywords[:5]:  # type: ignore
            print(f"  {keyword:<25} {score:.4f}")
    else:
        print("  No KeyBERT keywords extracted.")
    
    print("\nKeyBERT Final Results:")
    print("-" * 50)
    if keybert_keywords:
        for i, (keyword, score) in enumerate(keybert_keywords[:5], 1):  # type: ignore
            print(f"  {i}. {keyword:<25} score: {score:.4f}")
    else:
        print("  No KeyBERT keywords found.")
    
    # ============================================
    # COMPARISON SUMMARY
    # ============================================
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    print("\nTop 5 Keywords by Method:")
    print("-" * 50)
    print(f"{'RAKE':<25} {'TF-IDF':<25} {'KeyBERT'}")
    print("-" * 75)
    
    # Ensure lists exist and have items
    rake_keywords = rake_keywords if rake_keywords else []
    tfidf_keywords = tfidf_keywords if 'tfidf_keywords' in locals() and tfidf_keywords else []
    keybert_keywords = keybert_keywords if keybert_keywords else []

    max_len = max(len(rake_keywords), len(tfidf_keywords), len(keybert_keywords))
    for i in range(min(5, max_len)):
        rake_kw = str(rake_keywords[i][0]) if i < len(rake_keywords) else ""
        tfidf_kw = str(tfidf_keywords[i][0]) if i < len(tfidf_keywords) else ""
        keybert_kw = str(keybert_keywords[i][0]) if i < len(keybert_keywords) else ""
        print(f"{rake_kw:<25} {tfidf_kw:<25} {keybert_kw}")
    
    print("\n" + "=" * 70)


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
        
        # Extract all keywords automatically
        print("\nProcessing text and extracting all keywords...")
        keywords: List[Tuple[str, float]] = extractor.extract(text)
        
        # Display combined results (keywords + stopwords)
        display_combined_output(keywords, method_choice, text, extractor.processor.stopwords)
        
        # Ask to continue
        print("\n" + "-" * 60)
        choice = input("Extract keywords from another text? (y/n): ").strip().lower()
        if choice != 'y':
            print("\nThank you for using the Keyword Extractor!")
            break


if __name__ == "__main__":
    main()
