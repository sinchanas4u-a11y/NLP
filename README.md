# Keyword Extraction System

A comprehensive NLP-based keyword extraction system that supports multiple extraction methods including TF-IDF, RAKE, KeyBERT, and spaCy.

## Features

- **Multiple Extraction Methods**: Choose from TF-IDF, RAKE, KeyBERT, or spaCy
- **Text Preprocessing**: Automatic lowercase conversion, stopword removal, punctuation filtering
- **Multi-word Keyphrases**: Support for unigrams, bigrams, and trigrams
- **Edge Case Handling**: Graceful handling of empty input, short text, and non-English text
- **Word Cloud Visualization**: Generate visual representations of keyword frequency
- **Unified Interface**: Simple API for all extraction methods

## Installation

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Quick Start

```python
from keyword_extractor import extract_keywords

text = """
Machine learning is a subset of artificial intelligence that enables 
systems to learn and improve from experience without being explicitly 
programmed.
"""

# Extract keywords using KeyBERT (default)
keywords = extract_keywords(text, method='keybert', top_n=5)
print(keywords)
# Output: [('machine learning', 0.5523), ('deep learning', 0.4891), ...]
```

## Usage Examples

### Using Different Methods

```python
from keyword_extractor import KeywordExtractor

extractor = KeywordExtractor()

# TF-IDF
keywords = extractor.extract(text, method='tfidf', top_n=5)

# RAKE
keywords = extractor.extract(text, method='rake', top_n=5)

# KeyBERT (transformer-based)
keywords = extractor.extract(text, method='keybert', top_n=5)

# spaCy (noun phrases)
keywords = extractor.extract(text, method='spacy', top_n=5)
```

### Multi-word Keyphrases

```python
# Extract only single words
keywords = extractor.extract(text, method='keybert', top_n=5, ngram_range=(1, 1))

# Extract bigrams and trigrams
keywords = extractor.extract(text, method='keybert', top_n=5, ngram_range=(2, 3))
```

### Compare All Methods

```python
# Extract with all methods at once
all_results = extractor.extract_all(text, top_n=5)

for method, keywords in all_results.items():
    print(f"{method}: {keywords}")
```

### Generate Word Cloud

```python
from keyword_extractor import create_wordcloud

keywords = extractor.extract(text, method='keybert', top_n=20)
create_wordcloud(keywords, output_path='wordcloud.png')
```

## Extraction Methods

| Method | Description | Best For |
|--------|-------------|----------|
| **TF-IDF** | Statistical frequency-based | Single document analysis |
| **RAKE** | Rapid Automatic Keyword Extraction | Domain-specific texts |
| **KeyBERT** | Transformer-based semantic extraction | Semantic understanding |
| **spaCy** | POS tagging and noun phrases | Linguistic structure analysis |

## API Reference

### `KeywordExtractor`

Main class for keyword extraction.

#### Methods

- `extract(text, method='keybert', top_n=10, ngram_range=(1, 2))` - Extract keywords using specified method
- `extract_all(text, top_n=10)` - Extract keywords using all methods
- `validate_input(text)` - Validate input text

### `extract_keywords(text, method='keybert', top_n=10)`

Convenience function for quick extraction.

### `create_wordcloud(keywords, output_path='wordcloud.png', width=800, height=400)`

Generate a word cloud from extracted keywords.

## Output Format

All methods return a list of tuples:
```python
[("keyword1", 0.92), ("keyword2", 0.87), ("keyword3", 0.81)]
```

## Edge Cases

The system handles:
- Empty input strings
- Very short text (less than 10 characters)
- Non-English text detection
- Invalid method names

## Dependencies

- scikit-learn >= 1.3.0
- spacy >= 3.7.0
- rake-nltk >= 1.0.6
- keybert >= 0.8.3
- wordcloud >= 1.9.3
- matplotlib >= 3.7.0
- numpy >= 1.24.0
- nltk >= 3.8.1

## Running the Example

```bash
python example.py
```

This will demonstrate all features including:
- Single method extraction
- Comparison of all methods
- Multi-word keyphrase extraction
- Word cloud generation
- Edge case handling
# NLP
