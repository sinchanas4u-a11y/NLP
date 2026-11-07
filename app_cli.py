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
    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer  # type: ignore
    import numpy as np  # type: ignore
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Fixed reference background corpus for TF-IDF training
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
    
    def __init__(self, method: str = 'tfidf', reference_corpus: Optional[List[str]] = None):
        """
        Initialize the keyword extractor.
        
        Args:
            method: Extraction method ('tfidf', 'rake', or 'keybert')
            reference_corpus: Background corpus to fit TF-IDF vectorizer
        """
        self.method = method.lower()
        self.processor = TextProcessor()
        self.keybert_model = None
        self.reference_corpus = reference_corpus or REFERENCE_CORPUS
        
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
        Extract keywords using TF-IDF fitted on a fixed reference corpus with smooth_idf=True.
        
        Args:
            text: Input text (single document)
            top_n: Number of top keywords to return (None = return all)
            
        Returns:
            List of (keyword, score) tuples ranked in descending order
        """
        if not text or not text.strip():
            return []
        try:
            dynamic_corpus = list(self.reference_corpus) + [text]
            vectorizer = TfidfVectorizer(
                smooth_idf=True,
                norm=None,
                stop_words='english',
                token_pattern=r'(?u)\b[a-zA-Z]+\b'
            )
            tfidf_matrix: Any = vectorizer.fit_transform(dynamic_corpus)
            feature_names = vectorizer.get_feature_names_out()
            scores = np.asarray(tfidf_matrix[-1].toarray()).flatten()
            
            count_vec = CountVectorizer(
                vocabulary=vectorizer.vocabulary_,
                stop_words='english',
                token_pattern=r'(?u)\b[a-zA-Z]+\b'
            )
            user_tf_mat: Any = count_vec.transform([text])
            user_counts = np.asarray(user_tf_mat.toarray()).flatten()
            
            keyword_scores = []
            for idx in range(len(feature_names)):
                if user_counts[idx] > 0 and scores[idx] > 0:
                    keyword_scores.append((str(feature_names[idx]), float(round(float(scores[idx]), 4))))
            
            # Rank in descending order of TF-IDF score
            keyword_scores.sort(key=lambda x: (x[1], x[0]), reverse=True)
            
            if top_n is not None:
                return keyword_scores[:top_n]
            return keyword_scores
        except Exception as e:
            print(f"TF-IDF extraction error: {e}")
            return []
    
    def extract_tfidf_detailed(self, text: str, reference_corpus: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Extract keywords with detailed breakdown using dynamic corpus:
        keyword, term frequency, document frequency, IDF, TF-IDF score, and final rank.
        """
        ref_corpus = reference_corpus or self.reference_corpus
        if not text or not text.strip():
            return []
            
        dynamic_corpus = list(ref_corpus) + [text]
        vectorizer = TfidfVectorizer(
            smooth_idf=True,
            norm=None,
            stop_words='english',
            token_pattern=r'(?u)\b[a-zA-Z]+\b'
        )
        try:
            tfidf_matrix: Any = vectorizer.fit_transform(dynamic_corpus)
        except ValueError:
            return []
            
        feature_names = vectorizer.get_feature_names_out()
        scores = np.asarray(tfidf_matrix[-1].toarray()).flatten()
        idf_values = vectorizer.idf_
        
        count_vec = CountVectorizer(
            vocabulary=vectorizer.vocabulary_,
            stop_words='english',
            token_pattern=r'(?u)\b[a-zA-Z]+\b'
        )
        user_tf_mat: Any = count_vec.transform([text])
        user_counts = np.asarray(user_tf_mat.toarray()).flatten()
        total_content_tokens = int(user_counts.sum())
        if total_content_tokens == 0:
            return []

        ref_counts: Any = count_vec.transform(dynamic_corpus)
        ref_df = np.asarray((ref_counts > 0).sum(axis=0)).flatten()
        
        results = []
        for idx in range(len(feature_names)):
            count = int(user_counts[idx])
            if count > 0:
                df = int(ref_df[idx])
                tf = count / total_content_tokens
                idf = float(idf_values[idx])
                tfidf = tf * idf
                results.append({
                    'keyword': str(feature_names[idx]),
                    'count': count,
                    'tf': float(round(tf, 4)),
                    'df': df,
                    'idf': float(round(idf, 4)),
                    'tfidf': float(round(tfidf, 4))
                })
        
        # Rank keywords by TF-IDF score in descending order
        results.sort(key=lambda x: (x['tfidf'], x['idf'], x['keyword']), reverse=True)
        for rank, item in enumerate(results, 1):
            item['rank'] = rank
            
        return results

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
            
            raw_keywords: Any = self.keybert_model.extract_keywords(  # type: ignore
                text,
                keyphrase_ngram_range=(1, 2),
                stop_words='english',
                top_n=extract_n,
                use_mmr=True,
                diversity=0.7
            )
            
            result: List[Tuple[str, float]] = []
            if isinstance(raw_keywords, list):
                for item in raw_keywords:
                    if isinstance(item, tuple) and len(item) == 2:
                        score_val = float(item[1])
                        if score_val > 0.1:
                            result.append((str(item[0]), score_val))
            
            if top_n:
                return result[:top_n]
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


def get_phrases_split_by_stopwords(text: str, stopwords: set) -> List[str]:
    """
    Split text into candidate phrases using both stopwords and punctuation as delimiters.
    Commas, periods, semicolons, hyphens, colons, brackets, quotes, numbers,
    special characters, and stopwords all act as phrase boundaries.
    """
    if not text or not text.strip():
        return []
    
    # Tokenize contiguous alphabetic words and any non-letter/non-space character
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
            # Punctuation (commas, periods, semicolons, hyphens, etc.), numbers, symbols
            if current_phrase:
                phrases.append(' '.join(current_phrase))
                current_phrase = []

    if current_phrase:
        phrases.append(' '.join(current_phrase))

    # Keep all non-empty phrases, including single-word phrases
    return [p.strip() for p in phrases if p.strip()]


def calculate_rake_metrics(text: str, stopwords: set) -> dict:
    """
    Calculate RAKE metrics using candidate phrases split by stopwords and punctuation.
    Degree is calculated as co-occurrence count excluding itself.
    """
    if not text or not text.strip():
        return {
            'frequencies': {},
            'degrees': {},
            'scores': {},
            'content_words': []
        }

    phrases = get_phrases_split_by_stopwords(text, stopwords)
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

    scores = {}
    for word in freq:
        scores[word] = float(degree[word]) / float(freq[word]) if freq[word] > 0 else 0.0

    return {
        'frequencies': dict(freq),
        'degrees': dict(degree),
        'scores': scores,
        'content_words': content_words_list
    }


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
    print("\nStep 2: Split Text into Phrases (by stopwords and punctuation)")
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
    # TF-IDF (Term Frequency-Inverse Document Frequency)
    # ============================================
    print("\n" + "=" * 70)
    print("TF-IDF (Term Frequency-Inverse Document Frequency)")
    print("=" * 70)
    
    ref_corpus = tfidf_extractor.reference_corpus
    
    # Step 1: Preprocess Text
    print("\nStep 1: Preprocess Text")
    print("-" * 50)
    raw_tokens = re.findall(r'[a-zA-Z]+', first_doc.lower())
    ignored_stopwords = [w for w in raw_tokens if w in stopwords]
    content_tokens = [w for w in raw_tokens if w not in stopwords]
    print(f"User Input: \"{first_doc}\"")
    print(f"Extracted Content Tokens: {', '.join(content_tokens) if content_tokens else '(None)'}")
    print(f"Filtered Stopwords: {', '.join(sorted(set(ignored_stopwords))) if ignored_stopwords else '(None)'}")

    # Step 2: Build Dynamic Corpus
    print("\nStep 2: Build Dynamic Corpus")
    print("-" * 50)
    dynamic_corpus = list(ref_corpus) + [first_doc]
    N = len(dynamic_corpus)
    N_ref = len(ref_corpus)
    print(f"Reference Corpus Size: {N_ref} documents")
    print(f"Dynamic Corpus: {N_ref} reference docs + 1 user document = {N} total documents")
    print(f"User Document Location: Index {N-1} (Document {N}, last document in dynamic corpus)")

    tfidf_detailed = tfidf_extractor.extract_tfidf_detailed(first_doc, ref_corpus)
    
    if not tfidf_detailed:
        print("\nNo keywords extracted (input is empty, whitespace-only, or contains only stopwords).")
        tfidf_keywords = []
    else:
        # Step 3: Calculate Term Frequency
        total_tokens = sum(item['count'] for item in tfidf_detailed)
        print("\nStep 3: Calculate Term Frequency (TF)")
        print("-" * 65)
        print("Formula: TF = word count in user input / total content-word tokens in user input")
        print(f"Total content-word tokens in user input = {total_tokens}")
        print(f"{'Keyword':<16} {'Word Count':<12} {'Formula Calculation':<22} {'TF Score'}")
        for item in sorted(tfidf_detailed, key=lambda x: (-x['count'], x['keyword'])):
            calc_str = f"{item['count']} / {total_tokens}"
            print(f"  {item['keyword']:<16} {item['count']:<12} {calc_str:<22} {item['tf']:.4f}")

        # Step 4: Calculate Document Frequency
        print("\nStep 4: Calculate Document Frequency (DF)")
        print("-" * 65)
        print(f"Count of documents containing each word across all N={N} dynamic corpus documents.")
        print(f"{'Keyword':<16} {'DF':<6} {'Explanation'}")
        for item in sorted(tfidf_detailed, key=lambda x: (x['df'], x['keyword'])):
            in_ref = item['df'] - 1
            expl = f"1 (user doc) + {in_ref} (ref docs) = {item['df']} doc(s)"
            print(f"  {item['keyword']:<16} {item['df']:<6} {expl}")

        # Step 5: Calculate Inverse Document Frequency
        print("\nStep 5: Calculate Inverse Document Frequency (IDF)")
        print("-" * 65)
        print(f"Formula: IDF = log((1 + N) / (1 + DF)) + 1  [smooth_idf=True, N={N}]")
        print(f"{'Keyword':<16} {'DF':<6} {'Formula Calculation':<26} {'IDF Score'}")
        for item in sorted(tfidf_detailed, key=lambda x: (-x['idf'], x['keyword'])):
            calc_str = f"ln({1+N}/{1+item['df']}) + 1"
            print(f"  {item['keyword']:<16} {item['df']:<6} {calc_str:<26} {item['idf']:.4f}")

        # Step 6: Calculate TF-IDF Score
        print("\nStep 6: Calculate TF-IDF Score")
        print("-" * 65)
        print("Formula: TF-IDF = TF × IDF")
        print(f"{'Keyword':<16} {'TF':<8} {'IDF':<8} {'TF × IDF Calculation':<24} {'TF-IDF Score'}")
        for item in sorted(tfidf_detailed, key=lambda x: (-x['tfidf'], -x['count'], x['keyword'])):
            calc_str = f"{item['tf']:.4f} × {item['idf']:.4f}"
            print(f"  {item['keyword']:<16} {item['tf']:<8.4f} {item['idf']:<8.4f} {calc_str:<24} {item['tfidf']:.4f}")

        # Final Results
        print("\nTF-IDF Final Results:")
        print("-" * 88)
        print("TF-IDF KEYWORD EXTRACTION RESULTS (Ranked by TF-IDF Score in Descending Order):")
        print("-" * 88)
        print(f"{'Rank':<6} | {'Keyword':<16} | {'Word Count':<11} | {'TF':<8} | {'DF':<6} | {'IDF':<8} | {'TF-IDF Score':<12}")
        print("-" * 88)
        for item in tfidf_detailed:
            print(f"{item['rank']:<6} | {item['keyword']:<16} | {item['count']:<11} | {item['tf']:<8.4f} | {item['df']:<6} | {item['idf']:<8.4f} | {item['tfidf']:<12.4f}")
        print("-" * 88)
        
        tfidf_keywords = [(item['keyword'], item['tfidf']) for item in tfidf_detailed]

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
        rake_kw = rake_keywords[i][0] if i < len(rake_keywords) else ""
        tfidf_kw = str(tfidf_keywords[i][0]) if i < len(tfidf_keywords) else ""
        keybert_kw = keybert_keywords[i][0] if i < len(keybert_keywords) else ""
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
