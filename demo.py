"""Demo of the interactive keyword extractor processing"""

from interactive_keyword_extractor import TextProcessor, KeywordExtractor  # type: ignore

text = 'Artificial intelligence is transforming the healthcare industry through machine learning and data analytics.'

print('='*60)
print('INTERACTIVE KEYWORD EXTRACTOR')
print('='*60)
print()
print('Enter or paste your text (press Enter to process):')
print('-'*60)
print(text)
print()
print('='*60)
print('PROCESSING STEPS')
print('='*60)

processor = TextProcessor()

print()
print('Step 1: Clean text (remove punctuation, special chars, extra spaces)')
cleaned = processor.clean_text(text)
print('Result:', cleaned)

print()
print('Step 2: Lowercase all words')
lower = processor.lowercase(cleaned)
print('Result:', lower)

print()
print('Step 3: Remove stopwords')
no_stop = processor.remove_stopwords(lower)
print('Result:', no_stop)

print()
print('Step 4: Apply lemmatization')
lemmatized = processor.lemmatize(no_stop)
print('Result:', lemmatized)

print()
print('='*60)
print('EXTRACTED KEYWORDS (TF-IDF)')
print('='*60)

extractor = KeywordExtractor(method='tfidf')
keywords = extractor.extract(text, top_n=5)

print()
print('Rank   Keyword                   Relevance Score')
print('-'*60)
for i, (kw, score) in enumerate(keywords, 1):
    print(f'{i:<6} {kw:<25} {score:.4f}')
