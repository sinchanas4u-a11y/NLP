"""
Web-based Keyword Extraction Application
Flask backend with REST API for keyword extraction using RAKE, TF-IDF, and KeyBERT
"""

from flask import Flask, request, jsonify, render_template  # type: ignore
from flask_cors import CORS  # type: ignore
import os
import sys

# Import the keyword extraction modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from app import TextProcessor, KeywordExtractor, find_stopwords_in_text, calculate_rake_metrics, get_phrases_split_by_stopwords  # type: ignore

app = Flask(__name__)
CORS(app)

# Initialize extractors for each method
extractors = {
    'rake': KeywordExtractor(method='rake'),
    'tfidf': KeywordExtractor(method='tfidf'),
    'keybert': KeywordExtractor(method='keybert')
}


def format_rake_details(text, keywords, stopwords):
    """Format RAKE calculation details with step-by-step breakdown."""
    from collections import Counter
    import re
    
    metrics = calculate_rake_metrics(text, stopwords)
    phrases = get_phrases_split_by_stopwords(text, stopwords)
    
    # Step 1: Stopwords
    all_stopwords, unique_stopwords, total_stopwords = find_stopwords_in_text(text, stopwords)
    
    # Step 3: Word frequencies
    word_freq_data = []
    for word, freq in sorted(metrics['frequencies'].items()):
        word_freq_data.append({'word': word, 'frequency': freq})
    
    # Step 4: Word degrees with explanations
    word_degree_data = []
    for word, deg in sorted(metrics['degrees'].items()):
        containing_phrases = [p for p in phrases if word in p.split()]
        co_occurring = set()
        for phrase in containing_phrases:
            words_in_phrase = phrase.split()
            co_occurring.update([w for w in words_in_phrase if w != word and w in metrics['frequencies']])
        explanation = f"co-occurs with: {', '.join(co_occurring)}" if co_occurring else "no co-occurrence"
        word_degree_data.append({
            'word': word,
            'degree': deg,
            'explanation': explanation
        })
    
    # Step 5: Word scores
    word_score_data = []
    for word in sorted(metrics['scores'].keys()):
        score = metrics['scores'][word]
        freq = metrics['frequencies'][word]
        deg = metrics['degrees'][word]
        word_score_data.append({
            'word': word,
            'degree': deg,
            'frequency': freq,
            'score': round(score, 2),
            'calculation': f"{deg} / {freq} = {score:.2f}"
        })
    
    # Step 6: Phrase calculations
    phrase_calculations = []
    for phrase, score in keywords:
        words = phrase.split()
        word_scores_list = [metrics['scores'].get(w, 0) for w in words]
        score_sum = sum(word_scores_list)
        calculation = " + ".join([f"{s:.1f}" for s in word_scores_list])
        phrase_calculations.append({
            'phrase': phrase,
            'calculation': calculation,
            'total': round(score_sum, 1)
        })
    
    return {
        'steps': {
            'step1_stopwords': {
                'stopwords': unique_stopwords,
                'count': len(unique_stopwords)
            },
            'step2_phrases': phrases,
            'step3_word_frequencies': word_freq_data,
            'step4_word_degrees': word_degree_data,
            'step5_word_scores': word_score_data,
            'step6_phrase_scores': phrase_calculations
        },
        'phrases': phrases,
        'word_scores': word_score_data,
        'phrase_calculations': phrase_calculations
    }


def format_tfidf_details(text, keywords, stopwords):
    """Format TF-IDF calculation details with step-by-step breakdown."""
    from collections import Counter
    import re
    import math
    import nltk  # type: ignore
    try:
        nltk.tokenizers.punkt.PunktSentenceTokenizer
    except (AttributeError, LookupError):
        nltk.download('punkt', quiet=True)
    from nltk.tokenize import sent_tokenize  # type: ignore
    
    # Reference corpus for IDF
    reference_corpus = [
        "Machine learning is a subset of artificial intelligence",
        "Deep learning uses neural networks with multiple layers",
        "Natural language processing helps computers understand text",
        "Data science combines statistics and computer science",
        "Neural networks are inspired by biological neurons",
        "Computer vision enables machines to interpret images",
        "Reinforcement learning trains agents through rewards",
        "Supervised learning uses labeled training data",
        "Unsupervised learning finds patterns in unlabeled data",
        "Big data analytics processes large datasets"
    ]
    
    # Step 1: Split into Sentences
    sentences = sent_tokenize(text)
    
    # Step 2: Tokenize and remove stopwords for each sentence
    processed_sentences = []
    sentence_tokens = [] # for display
    for sent in sentences:
        words = re.findall(r'\b[a-zA-Z]{3,}\b', sent.lower())
        content_words = [w for w in words if w not in stopwords]
        processed_sentences.append(content_words)
        sentence_tokens.append(words)
    
    # Step 3: Calculate TF for EACH sentence
    all_tf_data = []
    all_content_words = set()
    for i, content_words in enumerate(processed_sentences):
        word_counts = Counter(content_words)
        total_words = len(content_words)
        if total_words == 0: continue
        
        sent_tf = []
        for word, count in word_counts.items():
            tf = count / total_words
            all_content_words.add(word)
            sent_tf.append({
                'word': word,
                'count': count,
                'total_words': total_words,
                'calculation': f"{count}/{total_words}",
                'tf': round(tf, 4)  # type: ignore
            })
        all_tf_data.append({
            'sentence_index': i + 1,
            'sentence_text': sentences[i],
            'tf_values': sent_tf
        })
    
    # Step 4: Calculate IDF across all sentences
    N = len(processed_sentences)
    idf_data = []
    idf_values = {}
    
    for word in sorted(all_content_words):
        # Calculate DF (Documents containing the word)
        df = sum(1 for sent in processed_sentences if word in sent)  # type: ignore
        # Use simple IDF log(N/df)
        # If N=df, log(N/df) = 0. We'll use log(N/df) + 1 to avoid zero
        idf = math.log(N / df) + 1
        idf_values[word] = idf  # type: ignore
        
        uniqueness = "high (unique to few sentences)" if df == 1 else "low (common across sentences)"
        if N > 1 and df > N/2: uniqueness = "low (very common)"
        
        idf_data.append({
            'word': word,
            'df': df,
            'corpus_size': N,
            'calculation': f"log({N}/{df}) + 1",
            'idf': round(idf, 4),  # type: ignore
            'uniqueness': uniqueness
        })
    
    # Step 5: Calculate Final TF-IDF (Average or Max across sentences for keyword list)
    # The 'keywords' passed to this function are already calculated by the extractor
    # We just need to format the top ones for the final step view
    top_words = [kw for kw, _ in keywords[:20]]
    tfidf_summary = []
    for word in top_words:
        # Find which sentence(s) it appeared in
        sents_found = [i+1 for i, sent in enumerate(processed_sentences) if word in sent]
        idf = idf_values.get(word, 0)
        
        tfidf_summary.append({
            'word': word,
            'sentences': sents_found,
            'idf': round(idf, 4),  # type: ignore
            'uniqueness': idf_data[next(i for i, x in enumerate(idf_data) if x['word'] == word)]['uniqueness']
        })
    
    return {
        'steps': {
            'step1_sentences': sentences,
            'step2_processed_sentences': [", ".join(s) for s in processed_sentences],
            'step3_per_sentence_tf': all_tf_data,
            'step4_idf_calculation': {
                'total_sentences': N,
                'idf_values': idf_data
            },
            'step5_final_scores': tfidf_summary
        }
    }


def format_keybert_details(text, keywords):
    """Format KeyBERT calculation details with step-by-step breakdown."""
    return {
        'steps': {
            'step1_document_embedding': 'Using pre-trained BERT model (all-MiniLM-L6-v2)',
            'step2_candidate_keywords': 'Extracting n-grams (unigrams and bigrams)',
            'step3_keyword_embeddings': 'Converting keywords to vector representations',
            'step4_cosine_similarity': 'Comparing keyword vectors with document vector',
            'similarity_scores': [{'keyword': kw, 'score': round(score, 4)} for kw, score in keywords[:10]]  # type: ignore
        }
    }


@app.route('/')
def index():
    """Serve the main HTML page."""
    return render_template('index.html')


@app.route('/extract', methods=['POST'])
def extract_keywords():
    """
    Extract keywords using all three methods.
    
    Request JSON: {"text": "input text"}
    Response JSON: {"rake": {...}, "tfidf": {...}, "keybert": {...}}
    """
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400
        
        text = data['text'].strip()
        if not text:
            return jsonify({'error': 'Empty text provided'}), 400
        
        if len(text) < 10:
            return jsonify({'error': 'Text too short (minimum 10 characters)'}), 400
        
        # Get stopwords
        stopwords = extractors['rake'].processor.stopwords
        all_stopwords, unique_stopwords, total_stopwords = find_stopwords_in_text(text, stopwords)
        
        results = {}
        
        # RAKE extraction
        rake_keywords = extractors['rake'].extract(text)
        results['rake'] = {
            'keywords': [{'keyword': kw, 'score': round(score, 2)} for kw, score in rake_keywords],  # type: ignore
            'stopwords': {
                'all': all_stopwords,
                'unique': unique_stopwords,
                'total': total_stopwords,
                'unique_count': len(unique_stopwords)
            },
            'calculation_details': format_rake_details(text, rake_keywords, stopwords)
        }
        
        # TF-IDF extraction
        tfidf_keywords = extractors['tfidf'].extract(text)
        results['tfidf'] = {  # type: ignore
            'keywords': [{'keyword': kw, 'score': round(score, 4)} for kw, score in tfidf_keywords],  # type: ignore
            'stopwords': {
                'all': all_stopwords,
                'unique': unique_stopwords,
                'total': total_stopwords,
                'unique_count': len(unique_stopwords)
            },
            'calculation_details': format_tfidf_details(text, tfidf_keywords, stopwords)
        }
        
        # KeyBERT extraction
        keybert_keywords = extractors['keybert'].extract(text)
        results['keybert'] = {  # type: ignore
            'keywords': [{'keyword': kw, 'score': round(score, 4)} for kw, score in keybert_keywords],  # type: ignore
            'stopwords': {
                'all': all_stopwords,
                'unique': unique_stopwords,
                'total': total_stopwords,
                'unique_count': len(unique_stopwords)
            },
            'calculation_details': format_keybert_details(text, keybert_keywords)
        }
        
        return jsonify({
            'success': True,
            'text': text,
            'results': results
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({'status': 'healthy'})


if __name__ == '__main__':
    print("Starting Keyword Extraction Web App...")
    print("Open http://localhost:5000 in your browser")
    app.run(debug=True, host='0.0.0.0', port=5000)
