"""
Keyword Extraction Web Application
Main Flask entry point.
"""
from flask import Flask, render_template, request, jsonify  # type: ignore
from flask_cors import CORS  # type: ignore
from preprocessing import TextProcessor  # type: ignore
from rake_extraction import RAKEExtractor  # type: ignore
from tfidf_extraction import TFIDFExtractor  # type: ignore
from keybert_extraction import KeyBERTExtractor  # type: ignore
import re
import os

# Fix for potential OpenMP crashes on Windows with PyTorch/Transformers
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

app = Flask(__name__)
CORS(app)

# Initialize Extractors
processor = TextProcessor()
extractors = {
    'rake': RAKEExtractor(processor),
    'tfidf': TFIDFExtractor(processor),
    'keybert': KeyBERTExtractor(processor)
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/extract', methods=['POST'])
def extract():
    data = request.json
    if not data or 'text' not in data:
        return jsonify({'error': 'No text provided'}), 400
    
    text = data['text']
    if len(text.strip()) < 10:
        return jsonify({'error': 'Text too short'}), 400

    results = {}
    
    # 1. RAKE Results
    rake_keywords = extractors['rake'].extract(text)
    rake_metrics = extractors['rake'].calculate_metrics(text)
    
    # Find active stopwords in text
    words = re.findall(r'[a-zA-Z]+', text.lower())
    found_stopwords = [w for w in words if w in processor.stopwords]
    
    results['rake'] = {
        'keywords': [{'keyword': str(kw), 'score': float(s)} for kw, s in rake_keywords[:10]],  # type: ignore
        'calculation_details': {
            'steps': {
                'step1_stopwords': {'stopwords': sorted(list(set(found_stopwords))), 'count': int(len(set(found_stopwords)))},
                'step2_phrases': [str(p) for p in extractors['rake'].get_phrases(text)],
                'step3_word_frequencies': [{'word': str(w), 'frequency': int(f)} for w, f in sorted(rake_metrics['frequencies'].items())],
                'step4_word_degrees': [{'word': str(w), 'degree': int(d), 'explanation': f"Co-occurs with {d} words"} for w, d in sorted(rake_metrics['degrees'].items())],
                'step5_word_scores': [{'word': str(w), 'calculation': f"{rake_metrics['degrees'][w]}/{rake_metrics['frequencies'][w]}", 'score': float(rake_metrics['scores'][w])} for w in sorted(rake_metrics['scores'].keys())],  # type: ignore
                'step6_phrase_scores': [{'phrase': str(p), 'calculation': " + ".join([f"{float(rake_metrics['scores'].get(w, 0.0)):.4f}" for w in p.split()]), 'total': float(s)} for p, s in rake_keywords[:10]]  # type: ignore
            }
        }
    }
    
    # 2. TF-IDF Results
    tfidf_keywords = extractors['tfidf'].extract(text)
    tfidf_detailed = extractors['tfidf'].get_detailed_results(text)

    results['tfidf'] = {
        'keywords': [{'keyword': kw, 'score': float(s)} for kw, s in tfidf_keywords[:10]],
        'calculation_details': {
            'strategy': tfidf_detailed.get('strategy', ''),
            'total_documents': int(tfidf_detailed['corpus_size']),
            'steps': {
                'step1_sentences': tfidf_detailed['sentences'],
                'step2_processed_sentences': tfidf_detailed['processed_sentences'],
                'step3_per_sentence_tf': [{'words': s['words']} for s in tfidf_detailed['per_sentence_tf']],
                'step4_idf_calculation': {
                    'total_sentences': int(tfidf_detailed['corpus_size']),
                    'idf_values': tfidf_detailed['idf_data']
                },
                'step5_tfidf_docs': tfidf_detailed.get('tfidf_docs', []),
                'step5_final_scores': tfidf_detailed['final_scores']
            }
        }
    }
    
    # 3. KeyBERT Results
    try:
        kb_keywords = extractors['keybert'].extract(text)
        kb_report = extractors['keybert'].get_detailed_report()
        
        results['keybert'] = {  # type: ignore
            'keywords': [{'keyword': str(kw), 'score': float(s)} for kw, s in kb_keywords[:10]],  # type: ignore
            'calculation_details': {
                'steps': {
                    'step1_document_embedding': str(kb_report['step1']),
                    'step2_candidate_keywords': str(kb_report['step2']),
                    'step3_keyword_embeddings': str(kb_report['step3']),
                    'step4_cosine_similarity': str(kb_report['step4']),
                    'similarity_scores': [{'keyword': str(kw), 'score': float(s)} for kw, s in kb_keywords[:10]]  # type: ignore
                }
            }
        }
    except Exception as e:
        print(f"KeyBERT error in app.py: {e}")
        results['keybert'] = {
            'keywords': [],
            'calculation_details': {'error': str(e), 'steps': {}}
        }

    return jsonify({'results': results})



if __name__ == '__main__':
    # Disable reloader for stability with heavy transformer models on Windows
    app.run(debug=True, port=5000, use_reloader=False)
