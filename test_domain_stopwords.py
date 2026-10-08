"""Test domain-specific stopword handling"""

from app import TextProcessor, KeywordExtractor  # type: ignore

text = 'Deep learning models are used for disease prediction and drug discovery.'

print('='*60)
print('DOMAIN-SPECIFIC STOPWORD HANDLING')
print('='*60)
print()
print(f'Text: {text}')
print()

# Test 1: General domain (treats 'used' as stopword)
print('1. GENERAL DOMAIN (used = stopword)')
print('-'*60)
processor1 = TextProcessor(domain='general')
extractor1 = KeywordExtractor(method='rake')
extractor1.processor = processor1
keywords1 = extractor1.extract(text)
check_words = {'used', 'using', 'use'}
found_stopwords = processor1.stopwords & check_words
print(f'Stopwords from check list: {sorted(found_stopwords)}')
print(f'Keywords found: {[kw for kw, _ in keywords1]}')
print()

# Test 2: Product domain (treats 'used' as keyword)
print('2. PRODUCT DOMAIN (used = keyword)')
print('-'*60)
processor2 = TextProcessor(domain='product')
extractor2 = KeywordExtractor(method='rake')
extractor2.processor = processor2
keywords2 = extractor2.extract(text)
found_stopwords2 = processor2.stopwords & check_words
print(f'Stopwords from check list: {sorted(found_stopwords2)}')
print(f'Keywords found: {[kw for kw, _ in keywords2]}')
print()

# Test 3: Technical domain (treats 'used' as stopword)
print('3. TECHNICAL DOMAIN (used = stopword)')
print('-'*60)
processor3 = TextProcessor(domain='technical')
extractor3 = KeywordExtractor(method='rake')
extractor3.processor = processor3
keywords3 = extractor3.extract(text)
check_words3 = {'used', 'using', 'use', 'method', 'methods'}
found_stopwords3 = processor3.stopwords & check_words3
print(f'Stopwords from check list: {sorted(found_stopwords3)}')
print(f'Keywords found: {[kw for kw, _ in keywords3]}')
print()

# Summary
print('='*60)
print('SUMMARY')
print('='*60)
print()
print('Word "used" treatment:')
print(f'  - General domain:   {"STOPWORD" if "used" in processor1.stopwords else "KEYWORD"}')
print(f'  - Product domain:   {"STOPWORD" if "used" in processor2.stopwords else "KEYWORD"}')
print(f'  - Technical domain: {"STOPWORD" if "used" in processor3.stopwords else "KEYWORD"}')
