from time import time
from sentence_transformers import SentenceTransformer

t1 = time()
model = SentenceTransformer('all-MiniLM-L6-v2')
dt = time() - t1
print(f"Loading model took {dt:.2f} seconds")

t1 = time()
sentence = "That is a happy person"
embedding = model.encode(sentence)
dt = time() - t1
print(f"Embedding took {dt:.2f} seconds")

# print("Sentence:", sentence)
# print("Embedding:", embedding[:5])
