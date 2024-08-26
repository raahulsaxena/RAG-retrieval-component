
import numpy as np
import sys
from collections import Counter, defaultdict
import math

def compute_idf(corpus, n_docs):
    """Compute the IDF (Inverse Document Frequency) for each term in the corpus."""
    idf = {}
    for doc in corpus:
        for word in set(doc.split()):
            idf[word] = idf.get(word, 0) + 1

    for word in idf:
        idf[word] = math.log((n_docs - idf[word] + 0.5) / (idf[word] + 0.5) + 1)
    
    return idf

def bm25_retrieve(query, chunk_texts, top_k=20):
    """Retrieve the top_k most relevant chunks using BM25."""
    
    # Precompute IDF for all terms in the corpus
    n_docs = len(chunk_texts)
    idf = compute_idf(chunk_texts, n_docs)
    
    # Define BM25 parameters
    k1 = 1.5  # Controls term frequency scaling
    b = 0.75  # Controls document length normalization

    # Compute average document length
    avg_doc_len = np.mean([len(doc.split()) for doc in chunk_texts])
    
    bm25_scores = []

    for i, text in enumerate(chunk_texts):
        score = 0
        doc_len = len(text.split())
        term_freq = Counter(text.split())
        
        for term in query.split():
            if term in term_freq:
                tf = term_freq[term]
                term_score = idf.get(term, 0) * ((tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (doc_len / avg_doc_len))))
                score += term_score
        
        bm25_scores.append((i, score))
    
    # Sort by BM25 scores
    bm25_scores.sort(key=lambda x: x[1], reverse=True)
    top_k_indices = [index for index, score in bm25_scores[:top_k]]
    scores = [score for index, score in bm25_scores[:top_k]]
    
    return top_k_indices, scores

# TD_IDF Retrieval
def compute_tf_idf(corpus):
    """Compute the TF-IDF scores for the corpus."""
    num_docs = len(corpus)
    term_freqs = []
    doc_freqs = defaultdict(int)
    
    # Calculate term frequencies and document frequencies
    for doc in corpus:
        tf = Counter(doc.split())
        term_freqs.append(tf)
        for term in tf.keys():
            doc_freqs[term] += 1
    
    # Calculate IDF
    idf = {term: math.log(num_docs / (1 + doc_freq)) for term, doc_freq in doc_freqs.items()}
    
    # Calculate TF-IDF for each document
    tf_idf = []
    for tf in term_freqs:
        tf_idf_doc = {term: (freq * idf[term]) for term, freq in tf.items()}
        tf_idf.append(tf_idf_doc)
    
    return tf_idf, idf

def transform_query(query, idf):
    """Transform the query into a TF-IDF vector based on the IDF of the corpus."""
    tf = Counter(query.split())
    query_tf_idf = {term: (freq * idf.get(term, 0)) for term, freq in tf.items()}
    return query_tf_idf

def cosine_similarity(tf_idf_doc, query_tf_idf):
    """Calculate the cosine similarity between a document and the query."""
    dot_product = sum(query_tf_idf.get(term, 0) * tf_idf_doc.get(term, 0) for term in query_tf_idf)
    doc_norm = math.sqrt(sum(val**2 for val in tf_idf_doc.values()))
    query_norm = math.sqrt(sum(val**2 for val in query_tf_idf.values()))
    
    if doc_norm * query_norm == 0:
        return 0.0
    else:
        return dot_product / (doc_norm * query_norm)

def load_chunks():
    """Load the chunks from the compressed .npz file."""
    # Load the numpy archive
    db = np.load('./chunks_array.npz')
    filenames = db['filename']  # Array of filenames corresponding to chunks
    chunk_idxs = db['chunk_idx']  # Array of chunk indices
    chunk_texts = db['chunk_text']  # Array of chunk texts
    embeddings = db['embedding']  # Array of embeddings for each chunk
    
    return filenames, chunk_idxs, chunk_texts, embeddings

def process_input():
    """Process the input query embedding and query string."""
    # Read binary data from standard input
    input_data = sys.stdin.buffer.read()
    
    # Split the input data into embedding and query parts
    embedding_size = 4096 * 4  # 4096 floats * 4 bytes per float (32-bit)
    query_embedding = np.frombuffer(input_data[:embedding_size], dtype=np.float32)
    query = input_data[embedding_size:].decode('utf-8')
    
    return query_embedding, query

def keyword_based_retrieve(query, chunk_texts, top_k=20):
    keyword_scores = []
    for i, text in enumerate(chunk_texts):
        score = sum([1 for word in query.split() if word.lower() in text.lower()])
        keyword_scores.append((i, score))
    
    # Sort by keyword match scores
    keyword_scores.sort(key=lambda x: x[1], reverse=True)
    top_k_indices = [index for index, score in keyword_scores[:top_k]]
    scores = [score for index, score in keyword_scores[:top_k]]
    return top_k_indices, scores

def cosine_similarity_doc_query(tf_idf_doc, query_tf_idf):
    """Calculate the cosine similarity between a document and the query."""
    # Calculate dot product
    dot_product = sum(query_tf_idf.get(term, 0) * tf_idf_doc.get(term, 0) for term in query_tf_idf)
    
    # Calculate the norm (magnitude) of the document and the query
    doc_norm = math.sqrt(sum(val**2 for val in tf_idf_doc.values()))
    query_norm = math.sqrt(sum(val**2 for val in query_tf_idf.values()))
    
    if doc_norm * query_norm == 0:
        return 0.0
    else:
        return dot_product / (doc_norm * query_norm)
    
def tfidf_based_retrieve(query, chunk_texts, top_k=20):
    tf_idf, idf = compute_tf_idf(chunk_texts)
    query_tf_idf = transform_query(query, idf)
    
    # Calculate similarity scores
    similarities = [cosine_similarity_doc_query(doc, query_tf_idf) for doc in tf_idf]
    
    # Sort by similarity scores
    top_k_indices = np.argsort(similarities)[-top_k:][::-1]
    scores = [similarities[i] for i in top_k_indices]
    
    return top_k_indices, scores

# Levenshetein Retrieval

def levenshtein_distance(s1, s2):
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    # Initialize the distance matrix
    previous_row = np.arange(len(s2) + 1)
    
    for i, c1 in enumerate(s1):
        current_row = previous_row.copy()
        current_row[0] = i + 1
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row[j + 1] = min(insertions, deletions, substitutions)
        previous_row = current_row
    
    return previous_row[-1]

def levenshtein_based_retrieve(query, chunk_texts, top_k=20):
    levenshtein_scores = []
    for i, text in enumerate(chunk_texts):
        score = levenshtein_distance(query, text)
        levenshtein_scores.append((i, -score))  # Use negative since lower distance is better
    
    # Sort by Levenshtein distance scores
    levenshtein_scores.sort(key=lambda x: x[1], reverse=True)
    top_k_indices = [index for index, score in levenshtein_scores[:top_k]]
    scores = [score for index, score in levenshtein_scores[:top_k]]
    
    return top_k_indices, scores

def bag_of_words_retrieve(query, chunk_texts, top_k=20):
    """Retrieve the top_k most relevant chunks using Bag of Words."""
    query_words = query.split()
    bow_scores = []

    for i, text in enumerate(chunk_texts):
        text_words = text.split()
        score = sum([1 for word in query_words if word in text_words])
        bow_scores.append((i, score))
    
    # Sort by BoW scores
    bow_scores.sort(key=lambda x: x[1], reverse=True)
    top_k_indices = [index for index, score in bow_scores[:top_k]]
    scores = [score for index, score in bow_scores[:top_k]]

    return top_k_indices, scores

def generate_ngrams(text, n=2):
    """Generate n-grams from a given text."""
    words = text.split()
    ngrams = zip(*[words[i:] for i in range(n)])
    return [' '.join(ngram) for ngram in ngrams]

def ngram_based_retrieve(query, chunk_texts, top_k=20, n=2):
    """Retrieve the top_k most relevant chunks using n-grams."""
    query_ngrams = generate_ngrams(query, n=n)
    ngram_scores = []

    for i, text in enumerate(chunk_texts):
        text_ngrams = generate_ngrams(text, n=n)
        score = sum([1 for ngram in query_ngrams if ngram in text_ngrams])
        ngram_scores.append((i, score))
    
    # Sort by n-gram match scores
    ngram_scores.sort(key=lambda x: x[1], reverse=True)
    top_k_indices = [index for index, score in ngram_scores[:top_k]]
    scores = [score for index, score in ngram_scores[:top_k]]

    return top_k_indices, scores


# Normalizer function
def normalize(scores):
    """Normalize scores to a 0-1 range."""
    min_score = min(scores)
    max_score = max(scores)
    return [(score - min_score) / (max_score - min_score) if max_score != min_score else 0 for score in scores]



def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))



def ensemble_retrieve(query_embedding, query, chunk_texts, embeddings, top_k=20):
    """Returns the normalized emsembled scores from a number of different retrieval methods"""
    # Embedding-based retrieval
    similarities = [cosine_similarity(query_embedding, emb) for emb in embeddings]
    embedding_indices = np.argsort(similarities)[-top_k:][::-1]
    embedding_scores = [similarities[i] for i in embedding_indices]
    embedding_scores = normalize(embedding_scores)

    # Keyword-based retrieval
    keyword_indices, keyword_scores = keyword_based_retrieve(query, chunk_texts, top_k)
    keyword_scores = normalize(keyword_scores)


    # Levenshtein distance retrieval
    levenshtein_indices, levenshtein_scores = levenshtein_based_retrieve(query, chunk_texts, top_k)
    levenshtein_scores = normalize(levenshtein_scores)

    # TF-IDF-based retrieval
    tfidf_indices, tfidf_scores = tfidf_based_retrieve(query, chunk_texts, top_k)
    tfidf_scores = normalize(tfidf_scores)

    # BM25-based retrieval
    bm25_indices, bm25_scores = bm25_retrieve(query, chunk_texts, top_k)
    bm25_scores = normalize(bm25_scores)

    
    # Bag of words based
    bow_indices, bow_scores = bag_of_words_retrieve(query, chunk_texts, top_k)
    bow_scores = normalize(bow_scores)

    # N-gram based

    ngram_indices, ngram_scores = ngram_based_retrieve(query, chunk_texts, top_k)
    ngram_scores = normalize(ngram_scores)


    # Combine the results using a weighted sum
    combined_scores = defaultdict(float)

    # Assign weights to each method (these can be adjusted based on performance)
    # After trying different weights, I noticed that keeping embedding weight = 1.0 
    # gave me better results
    embedding_weight = 0.2 #0.8
    keyword_weight = 0.0
    levenshtein_weight = 0.2
    tfidf_weight = 0.2 # 0.2
    bm25_weight = 0.0
    bow_weight = 0.2
    ngram_weight = 0.4

    for i, index in enumerate(embedding_indices):
        combined_scores[index] += embedding_scores[i] * embedding_weight

    for i, index in enumerate(keyword_indices):
        combined_scores[index] += keyword_scores[i] * keyword_weight

    for i, index in enumerate(levenshtein_indices):
        combined_scores[index] += levenshtein_scores[i] * levenshtein_weight

    for i, index in enumerate(tfidf_indices):
        combined_scores[index] += tfidf_scores[i] * tfidf_weight

    for i, index in enumerate(bm25_indices):
        combined_scores[index] += bm25_scores[i] * bm25_weight

    for i, index in enumerate(bow_indices):
        combined_scores[index] += bow_scores[i] * bow_weight

    for i, index in enumerate(ngram_indices):
        combined_scores[index] += ngram_scores[i] * ngram_weight

    # Sort by combined scores and return top_k indices
    sorted_combined_scores = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
    top_k_indices = [index for index, score in sorted_combined_scores[:top_k]]
    top_k_scores = sorted_combined_scores[:top_k]
    return top_k_indices, top_k_scores




def jaccard_similarity(text1, text2):
    """Jaccard similarity between text1 and text2."""

    # Tokenize the texts into words
    set1 = set(text1.lower().split())
    set2 = set(text2.lower().split())

    # Calculate the intersection and union of the two sets
    intersection = set1.intersection(set2)
    union = set1.union(set2)

    # Calculate the Jaccard similarity
    jaccard_sim = len(intersection) / len(union)
    
    return jaccard_sim


def jaccard_utility(chunk_texts, top_k_indices):
    """Calculating the maximum jaccard similarity of the response chunk test 
    against the ground truth from the queries.json """

    max_jaccard = 0
    for idx in top_k_indices:
        jaccard = jaccard_similarity(chunk_texts[347], chunk_texts[idx])
        max_jaccard = max(jaccard, max_jaccard)
    
    print(max_jaccard)

def main():
    # Load the chunks data
    filenames, chunk_idxs, chunk_texts, embeddings = load_chunks()

    # filename: source file of the chunk (row)
    # chunk_idxs : position of chunk within its respective doc
    # chunk_texts: actual chunk text
    # embeddings : numeric representations of each chunk text

    # Process the input query
    query_embedding, query = process_input()

    # Retrieve the most relevant chunks using ensemble method
    top_k_indices, top_k_scores = ensemble_retrieve(query_embedding, query, chunk_texts, embeddings, top_k=20)

    # Output the row indices of the selected chunks
    output_indices = ','.join(map(str, top_k_indices))

    print(output_indices)


if __name__ == "__main__":
    main()


