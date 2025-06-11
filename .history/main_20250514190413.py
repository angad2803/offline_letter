import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Load model (runs offline after first download)
model = SentenceTransformer('all-MiniLM-L6-v2')  # small, fast

# Load dataset
df = pd.read_csv("dataset.csv")

# Create or load vector index
def build_index(data):
    embeddings = model.encode(data['input_letter'].tolist())
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    return index, embeddings

index, embeddings = build_index(df)

# Function to find best match
def find_response(user_input):
    input_vec = model.encode([user_input])
    D, I = index.search(np.array(input_vec), k=1)
    match_idx = I[0][0]
    similarity = D[0][0]
    matched_input = df.iloc[match_idx]['input_letter']
    response = df.iloc[match_idx]['response']
    return matched_input, response, similarity

# Main loop
print("=== Offline Letter Response App ===")
while True:
    query = input("\nEnter your letter (or 'exit'): ")
    if query.lower() == 'exit':
        break

    matched_input, response, score = find_response(query)
    print("\nBest match found:")
    print(f"→ Similar letter: {matched_input}")
    print(f"→ Response: {response}")
