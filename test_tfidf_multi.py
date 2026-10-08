"""Test multi-document TF-IDF calculation"""

from app import KeywordExtractor  # type: ignore
from collections import Counter
import math

# Test documents from expected output
documents = [
    "Deep learning models are used for disease prediction",
    "Machine learning is used for drug discovery",
    "Artificial intelligence enables disease detection and prediction"
]

print("="*60)
print("TF-IDF KEYWORD EXTRACTION - MULTIPLE DOCUMENTS")
print("="*60)
print()

print("INPUT DOCUMENTS:")
for i, doc in enumerate(documents, 1):
    print(f"Document {i}: \"{doc}\"")
print()

# Stopwords
stopwords = {'the', 'and', 'for', 'are', 'used', 'with', 'that', 'this', 
             'from', 'they', 'have', 'has', 'had', 'what', 'when', 'where',
             'who', 'which', 'why', 'how', 'all', 'any', 'both', 'each',
             'more', 'most', 'other', 'some', 'such', 'only', 'own', 'same',
             'than', 'too', 'very', 'can', 'will', 'just', 'should', 'now',
             'is', 'a', 'of', 'in', 'to', 'it', 'be', 'or', 'as', 'by'}

import re

# Process documents
processed_docs = []
for doc in documents:
    words = re.findall(r'\b[a-zA-Z]{3,}\b', doc.lower())
    content_words = [w for w in words if w not in stopwords]
    processed_docs.append(content_words)

print("-"*60)
print("STEP 1 - STOPWORDS REMOVED:")
print("-"*60)
for i, doc in enumerate(processed_docs, 1):
    print(f"Document {i}: {', '.join(doc)}")
print()

# Calculate TF for each document
print("-"*60)
print("STEP 2 - TF CALCULATION:")
print("(TF = Word Count / Total Words in that Document)")
print("-"*60)

doc_tf_data = []
for i, doc in enumerate(processed_docs, 1):
    word_counts = Counter(doc)
    total_words = len(doc)
    print(f"\nDocument {i} (Total Words = {total_words}):")
    tf_data = {}
    for word, count in word_counts.items():
        tf = count / total_words
        tf_data[word] = tf
        print(f"  {word:<12} = {count}/{total_words} = {tf:.3f}")
    doc_tf_data.append(tf_data)

# Calculate IDF
N = len(processed_docs)
all_words = set()
for doc in processed_docs:
    all_words.update(doc)

print()
print("-"*60)
print("STEP 3 - IDF CALCULATION:")
print(f"(IDF = log(Total Documents / Documents containing the word))")
print(f"Total Documents = {N}")
print("-"*60)
print(f"{'Word':<12} {'In Docs':<10} {'IDF Calculation':<25} {'IDF Score'}")

idf_values = {}
for word in sorted(all_words):
    df = sum(1 for doc in processed_docs if word in doc)
    idf = math.log(N / df)
    idf_values[word] = idf
    print(f"  {word:<12} {df:<10} log({N}/{df}) = log({N/df:.1f}){'':<10} {idf:.3f}")

# Calculate TF-IDF for each document
print()
print("-"*60)
print("STEP 4 - TF-IDF SCORE CALCULATION:")
print("(TF-IDF = TF x IDF)")
print("-"*60)

doc_tfidf_results = []
for i, (doc, tf_data) in enumerate(zip(processed_docs, doc_tf_data), 1):
    print(f"\nDocument {i}:")
    results = []
    for word in tf_data.keys():
        tf = tf_data[word]
        idf = idf_values[word]
        tfidf = tf * idf
        results.append((word, tfidf, tf, idf))
        print(f"  {word:<12} = {tf:.3f} x {idf:.3f} = {tfidf:.3f}")
    
    # Sort by TF-IDF score
    results.sort(key=lambda x: x[1], reverse=True)
    doc_tfidf_results.append(results)

# Final results
print()
print("-"*60)
print("STEP 5 - FINAL KEYWORDS PER DOCUMENT:")
print("(Sorted by TF-IDF Score - Highest First)")
print("-"*60)

for i, results in enumerate(doc_tfidf_results, 1):
    print(f"\nDocument {i} Keywords:")
    for j, (word, score, tf, idf) in enumerate(results, 1):
        uniqueness = "unique to this doc - high score" if idf > 0.4 else "common across docs - low score"
        print(f"  {j}. {word:<12} score: {score:.3f}  ({uniqueness})")

print()
print("="*60)
print("KEY OBSERVATION:")
print("- Words unique to one document  → HIGH TF-IDF score")
print("- Words common across documents → LOW TF-IDF score")
print("="*60)
