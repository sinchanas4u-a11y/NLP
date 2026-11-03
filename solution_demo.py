"""
Keyword Extraction Demonstration (RAKE, TF-IDF, KeyBERT)
Features:
- RAKE with punctuation- and stopword-aware phrase splitting
- TF-IDF fitted on fixed reference corpus with smooth_idf=True
- KeyBERT semantic keyword extraction with cosine similarity
"""

import re
from typing import List, Tuple, Optional, Dict, Any
from collections import Counter

# Safe third-party NLP imports with fallback guards and type ignores for LSP/type checkers
try:
    import numpy as np  # type: ignore
except ImportError:
    np = None  # type: ignore

try:
    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer  # type: ignore
except ImportError:
    TfidfVectorizer = None  # type: ignore
    CountVectorizer = None  # type: ignore

try:
    from rake_nltk import Rake  # type: ignore
except ImportError:
    Rake = None  # type: ignore

try:
    from keybert import KeyBERT  # type: ignore
except ImportError:
    KeyBERT = None  # type: ignore

try:
    import nltk  # type: ignore
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
except (ImportError, Exception):
    nltk = None  # type: ignore

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


class TextProcessor:
    """Handles text cleaning and stopword management with offline fallback."""
    def __init__(self):
        try:
            from nltk.corpus import stopwords  # type: ignore
            self.stopwords = set(stopwords.words('english'))
        except (ImportError, LookupError, Exception):
            # Comprehensive fallback stopwords if nltk corpus is unavailable
            self.stopwords = {
                'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours',
                'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers',
                'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
                'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are',
                'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',
                'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
                'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into',
                'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down',
                'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'such'
            }
        self.stopwords.update({'used', 'applied'})


# ===========================================================================
# 1. RAKE (RAPID AUTOMATIC KEYWORD EXTRACTION) - PUNCTUATION & STOPWORD AWARE
# ===========================================================================
def find_stopwords_in_text(text: str, stopwords: set) -> Tuple[List[str], List[str], int]:
    words = re.findall(r'[a-zA-Z]+', text.lower())
    found = [w for w in words if w in stopwords]
    unique = sorted(list(set(found)))
    return found, unique, len(found)


def get_phrases_split_by_stopwords(text: str, stopwords: set) -> List[str]:
    """
    Split text into candidate phrases using BOTH stopwords and punctuation as delimiters.
    Commas, periods, semicolons, colons, hyphens, brackets, quotes, numbers,
    special characters, and stopwords all act as phrase boundaries.
    """
    if not text or not text.strip():
        return []

    # Tokenize into contiguous alphabetic words and non-letter/non-space symbols
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

    # Keep all valid non-empty phrases, including single-word phrases
    return [p.strip() for p in phrases if p.strip()]


def calculate_rake_metrics(text: str, stopwords: set) -> dict:
    """
    Calculate RAKE metrics using candidate phrases split by stopwords and punctuation.
    Degree is calculated as co-occurrence count excluding itself.
    """
    if not text or not text.strip():
        return {'frequencies': {}, 'degrees': {}, 'scores': {}, 'content_words': []}

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


def extract_rake(text: str, top_n: Optional[int] = None) -> List[Tuple[str, float]]:
    """Extract and rank candidate phrases using RAKE."""
    processor = TextProcessor()
    phrases = get_phrases_split_by_stopwords(text, processor.stopwords)
    if not phrases:
        return []
    metrics = calculate_rake_metrics(text, processor.stopwords)

    seen = set()
    result = []
    for phrase in phrases:
        if phrase in seen:
            continue
        seen.add(phrase)
        words = phrase.split()
        phrase_score = sum(metrics['scores'].get(word, 0.0) for word in words)
        result.append((phrase, float(phrase_score)))

    # Sort descending by score, breaking ties by phrase length
    result.sort(key=lambda x: (x[1], len(x[0].split())), reverse=True)
    return result[:top_n] if top_n else result


# ===========================================================================
# 2. KeyBERT EXTRACTOR (UNCHANGED)
# ===========================================================================
def extract_keybert(text: str, top_n: int = 5) -> List[Tuple[str, float]]:
    """Extract keywords using KeyBERT with embeddings and cosine similarity."""
    try:
        from keybert import KeyBERT  # type: ignore
        kw_model = KeyBERT(model="all-MiniLM-L6-v2")
        keywords = kw_model.extract_keywords(
            text,
            keyphrase_ngram_range=(1, 2),
            stop_words='english',
            top_n=top_n,
            use_mmr=True,
            diversity=0.7
        )
        return [(str(kw), float(score)) for kw, score in keywords]
    except Exception as e:
        print(f"KeyBERT unavailable or error: {e}")
        return []


# ===========================================================================
# 3. TF-IDF EXTRACTOR (UNCHANGED)
# ===========================================================================
def extract_tfidf_with_details(user_text: str, reference_corpus: List[str]) -> List[Dict[str, Any]]:
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer  # type: ignore
        import numpy as np  # type: ignore
    except ImportError:
        print("Scikit-learn or NumPy is not installed.")
        return []

    vectorizer = TfidfVectorizer(smooth_idf=True, stop_words='english')
    vectorizer.fit(reference_corpus)

    tfidf_matrix = vectorizer.transform([user_text])
    feature_names = vectorizer.get_feature_names_out()
    tfidf_scores = tfidf_matrix.toarray()[0]
    idf_values = vectorizer.idf_

    count_vec = CountVectorizer(vocabulary=vectorizer.vocabulary_, stop_words='english')
    user_tf = count_vec.transform([user_text]).toarray()[0]
    ref_counts = count_vec.transform(reference_corpus)
    ref_df = np.asarray((ref_counts > 0).sum(axis=0)).flatten()

    keyword_details = []
    for idx in range(len(feature_names)):
        if user_tf[idx] > 0:
            keyword_details.append({
                'keyword': str(feature_names[idx]),
                'tf': int(user_tf[idx]),
                'df': int(ref_df[idx]),
                'idf': float(round(float(idf_values[idx]), 4)),
                'tfidf': float(round(float(tfidf_scores[idx]), 4))
            })

    keyword_details.sort(key=lambda x: (x['tfidf'], x['idf']), reverse=True)
    for rank, item in enumerate(keyword_details, 1):
        item['rank'] = rank

    return keyword_details


# ===========================================================================
# DISPLAY ALL METHODS (PRESERVING EXISTING OUTPUT STEPS)
# ===========================================================================
def display_all_methods_calculations(user_text: str, reference_corpus: Optional[List[str]] = None):
    if reference_corpus is None:
        reference_corpus = REFERENCE_CORPUS

    processor = TextProcessor()
    stopwords = processor.stopwords

    print("\n" + "=" * 70)
    print("         DETAILED STEP-BY-STEP CALCULATIONS")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # RAKE CALCULATIONS
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("RAKE (Rapid Automatic Keyword Extraction)")
    print("=" * 70)

    # Step 1: Identify stopwords
    all_stopwords, unique_stopwords, total_stopwords = find_stopwords_in_text(user_text, stopwords)
    print("\nStep 1: Identify Stopwords")
    print("-" * 50)
    print(f"Stopwords found: {', '.join(unique_stopwords)}")
    print(f"Total unique stopwords: {len(unique_stopwords)}")

    # Step 2: Split into phrases
    phrases = get_phrases_split_by_stopwords(user_text, stopwords)
    print("\nStep 2: Split Text into Phrases (by stopwords and punctuation)")
    print("-" * 50)
    for i, phrase in enumerate(phrases, 1):
        if phrase.strip():
            print(f"  Phrase {i}: \"{phrase}\"")

    # Step 3: Calculate word frequency
    rake_metrics = calculate_rake_metrics(user_text, stopwords)
    print("\nStep 3: Calculate Word Frequency")
    print("-" * 50)
    print(f"{'Word':<15} {'Frequency'}")
    for word, freq in sorted(rake_metrics['frequencies'].items()):
        print(f"  {word:<15} {freq}")

    # Step 4: Calculate word degree
    print("\nStep 4: Calculate Word Degree (Co-occurrence excluding itself)")
    print("-" * 50)
    print(f"{'Word':<15} {'Degree'}  {'Explanation'}")
    for word, deg in sorted(rake_metrics['degrees'].items()):
        containing_phrases = [p for p in phrases if word in p.split()]
        co_occurring = set()
        for phrase in containing_phrases:
            words_in_phrase = phrase.split()
            co_occurring.update([w for w in words_in_phrase if w != word and w in rake_metrics['frequencies']])
        explanation = f"co-occurs with: {', '.join(sorted(co_occurring))}" if co_occurring else "no co-occurrence"
        print(f"  {word:<15} {deg:<7} {explanation}")

    # Step 5: Calculate word score
    print("\nStep 5: Calculate Word Score (Degree / Frequency)")
    print("-" * 50)
    print(f"{'Word':<15} {'Calculation':<20} {'Score'}")
    for word in sorted(rake_metrics['scores'].keys()):
        score = rake_metrics['scores'][word]
        freq = rake_metrics['frequencies'][word]
        deg = rake_metrics['degrees'][word]
        print(f"  {word:<15} {deg} / {freq} = {score:.2f}")

    # Step 6: Calculate phrase scores
    rake_keywords = extract_rake(user_text)
    print("\nStep 6: Calculate Phrase Score (Sum of Word Scores)")
    print("-" * 50)
    for phrase, score in rake_keywords:
        words = phrase.split()
        word_scores = [rake_metrics['scores'].get(w, 0.0) for w in words]
        calculation = " + ".join([f"{s:.1f}" for s in word_scores])
        total = sum(word_scores)
        print(f'  "{phrase}"')
        print(f'    = {calculation} = {total:.1f}')

    print("\nRAKE Final Results:")
    print("-" * 50)
    for i, (keyword, score) in enumerate(rake_keywords, 1):
        print(f"  {i}. {keyword:<32} score: {score:.1f}")

    # -------------------------------------------------------------------------
    # TF-IDF CALCULATIONS (UNCHANGED)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("TF-IDF (Term Frequency-Inverse Document Frequency)")
    print("=" * 70)

    N = len(reference_corpus)
    print(f"\nREFERENCE BACKGROUND CORPUS ({N} documents):")
    for i, doc in enumerate(reference_corpus, 1):
        print(f"  Doc {i}: \"{doc}\"")

    print(f"\nUSER INPUT DOCUMENT:\n\"{user_text}\"")

    print("\n" + "-" * 70)
    print("TF-IDF CALCULATION FORMULAS:")
    print("  • Term Frequency (TF): Word count in user's input text")
    print(f"  • Document Frequency (DF): Document occurrences across {N} reference documents")
    print("  • IDF Score: ln((1 + N) / (1 + DF)) + 1  [smooth_idf=True]")
    print("  • TF-IDF Score: L2-normalized TF-IDF from vectorizer.transform([user_text])")
    print("-" * 70)

    tfidf_detailed = extract_tfidf_with_details(user_text, reference_corpus)

    print("\n" + "-" * 80)
    print("TF-IDF KEYWORD EXTRACTION RESULTS:")
    print("(Ranked by TF-IDF Score in Descending Order)")
    print("-" * 80)
    print(f"{'Rank':<6} | {'Keyword':<16} | {'Term Freq':<10} | {'Doc Freq':<10} | {'IDF Score':<12} | {'TF-IDF Score':<12}")
    print("-" * 80)
    for item in tfidf_detailed:
        print(f"{item['rank']:<6} | {item['keyword']:<16} | {item['tf']:<10} | {item['df']:<10} | {item['idf']:<12.4f} | {item['tfidf']:<12.4f}")
    print("-" * 80)

    tfidf_keywords = [(item['keyword'], item['tfidf']) for item in tfidf_detailed]

    # -------------------------------------------------------------------------
    # KeyBERT CALCULATIONS (UNCHANGED)
    # -------------------------------------------------------------------------
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

    keybert_keywords = extract_keybert(user_text, top_n=5)
    print("\nKeyBERT Similarity Scores:")
    print("-" * 50)
    print(f"{'Keyword':<25} {'Similarity Score'}")
    for keyword, score in keybert_keywords:
        print(f"  {keyword:<25} {score:.4f}")

    print("\nKeyBERT Final Results:")
    print("-" * 50)
    for i, (keyword, score) in enumerate(keybert_keywords, 1):
        print(f"  {i}. {keyword:<25} score: {score:.4f}")

    # -------------------------------------------------------------------------
    # COMPARISON SUMMARY
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    print("\nTop 5 Keywords by Method:")
    print("-" * 50)
    print(f"{'RAKE':<32} {'TF-IDF':<20} {'KeyBERT'}")
    print("-" * 75)

    max_len = max(len(rake_keywords), len(tfidf_keywords), len(keybert_keywords))
    for i in range(min(5, max_len)):
        rake_kw = str(rake_keywords[i][0]) if i < len(rake_keywords) else ""
        tfidf_kw = str(tfidf_keywords[i][0]) if i < len(tfidf_keywords) else ""
        keybert_kw = str(keybert_keywords[i][0]) if i < len(keybert_keywords) else ""
        print(f"{rake_kw:<32} {tfidf_kw:<20} {keybert_kw}")
    print("=" * 70)


if __name__ == "__main__":
    test_input = "Climate change is increasing global temperatures and causing extreme weather events such as floods, droughts, and wildfires."
    display_all_methods_calculations(test_input)
