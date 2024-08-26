import numpy as np
import sys
# NpzFile './chunks_array.npz'with keys:
# filename, chunk_idx, chunk_text, embedding
db = np.load('./chunks_array.npz')
embedding = db['embedding'][0].tobytes()
query = b'What does Williams, claim 3 say?'

sys.stdout.buffer.write(
embedding + query
)