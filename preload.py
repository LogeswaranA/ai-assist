from sentence_transformers import SentenceTransformer

# Download and cache the model
model = SentenceTransformer('all-MiniLM-L6-v2')
print("Model cached successfully!")