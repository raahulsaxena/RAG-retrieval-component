
import numpy as np
import sys
from collections import Counter, defaultdict
import math




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
    # input_data = sys.stdin.buffer.read()

    # Example: Create a random 4096-dimensional embedding vector
    query_embedding = np.random.rand(4096).astype(np.float32)

    # Convert the embedding to bytes
    embedding_bytes = query_embedding.tobytes()

    query_string = "Which paragraph in the description supports Karafin, claim 11?" #3
    query_bytes = query_string.encode('utf-8')

    input_data = embedding_bytes + query_bytes
    
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

    # bag of words based
    bow_indices, bow_scores = bag_of_words_retrieve(query, chunk_texts, top_k)
    bow_scores = normalize(bow_scores)

    ngram_indices, ngram_scores = ngram_based_retrieve(query, chunk_texts, top_k)
    ngram_scores = normalize(ngram_scores)


    # Combine the results using a weighted sum
    combined_scores = defaultdict(float)

    # Assign weights to each method (these can be adjusted based on performance)
    embedding_weight = 0.5 #0.8
    keyword_weight = 0.0
    bow_weight = 0.5
    ngram_weight = 0.0


    for i, index in enumerate(embedding_indices):
        combined_scores[index] += embedding_scores[i] * embedding_weight

    for i, index in enumerate(keyword_indices):
        combined_scores[index] += keyword_scores[i] * keyword_weight

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


def remove_stop_words(text):

    # Example stop words list
    stop_words_set = set(["the", "is", "does", "by", "of", "in", "a", "an", "on"])

    words = text.split()
    filtered_words = [word for word in words if word.lower() not in stop_words_set]
    return ' '.join(filtered_words)

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

    jaccard_utility(chunk_texts, top_k_indices)
    

if __name__ == "__main__":
    main()


