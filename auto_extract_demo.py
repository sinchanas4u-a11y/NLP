"""Demo of automatic keyword extraction - no user count needed"""

from app import TextProcessor, KeywordExtractor  # type: ignore

text = '''Artificial intelligence and machine learning are transforming the healthcare 
industry through data analytics. Deep learning neural networks can process massive 
amounts of data to recognize complex patterns. Natural language processing enables 
computers to understand human language.'''

print('='*70)
print('AUTOMATIC KEYWORD EXTRACTION SYSTEM')
print('Extracts ALL meaningful keywords without asking for count')
print('='*70)
print()
print('Input Text:')
print('-'*70)
print(text)
print()

processor = TextProcessor()

print('='*70)
print('NLP PROCESSING PIPELINE')
print('='*70)

print()
print('Step 1: Clean text (remove punctuation, special chars, extra spaces)')
cleaned = processor.clean_text(text)
print(f'  Result: {cleaned}')

print()
print('Step 2: Lowercase all words')
lower = processor.lowercase(cleaned)
print(f'  Result: {lower}')

print()
print('Step 3: Remove stopwords (e.g., "the", "is", "and", "a")')
no_stop = processor.remove_stopwords(lower)
print(f'  Result: {no_stop}')

print()
print('Step 4: Apply lemmatization (normalize words to base form)')
lemmatized = processor.lemmatize(no_stop)
print(f'  Result: {lemmatized}')

print()
print('='*70)
print('EXTRACTED KEYWORDS - TF-IDF Method')
print('='*70)

extractor = KeywordExtractor(method='tfidf')
keywords = extractor.extract(text)

print(f'\nTotal keywords extracted: {len(keywords)}')
print()
print(f'{"Rank":<6} {"Keyword":<35} {"Relevance Score"}')
print('-'*70)
for i, (kw, score) in enumerate(keywords, 1):
    print(f'{i:<6} {kw:<35} {score:.4f}')

print()
print('='*70)
print('SINGLE WORDS vs MULTI-WORD PHRASES')
print('='*70)

single_words = [(kw, score) for kw, score in keywords if ' ' not in kw]
phrases = [(kw, score) for kw, score in keywords if ' ' in kw]

print(f'\nSingle Words ({len(single_words)}):')
print(', '.join([kw for kw, _ in single_words[:10]]) + ('...' if len(single_words) > 10 else ''))  # type: ignore

print(f'\nMulti-word Phrases ({len(phrases)}):')
for kw, score in phrases[:10]:  # type: ignore
    print(f'  - {kw} ({score:.4f})')
if len(phrases) > 10:
    print(f'  ... and {len(phrases) - 10} more')
