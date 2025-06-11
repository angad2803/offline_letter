import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import tkinter as tk
from tkinter import messagebox

# Load and preprocess the dataset
def load_dataset(file_path):
    df = pd.read_csv(file_path)
    if 'Letter' not in df.columns or 'Response' not in df.columns:
        raise ValueError("Dataset must contain 'Letter' and 'Response' columns.")
    return df

# Generate embeddings using SentenceTransformer
def generate_embeddings(texts, model):
    return model.encode(texts, convert_to_tensor=True)

# Find the closest match using cosine similarity
def find_closest_match(input_embedding, dataset_embeddings):
    similarities = cosine_similarity([input_embedding], dataset_embeddings)
    closest_idx = similarities.argmax()
    return closest_idx, similarities[0][closest_idx]

# Main function to integrate the steps
def main():
    # Load the dataset
    dataset_path = 'dataset.csv'
    try:
        dataset = load_dataset(dataset_path)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # Load the SentenceTransformer model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Generate embeddings for the dataset
    dataset_embeddings = generate_embeddings(dataset['Letter'].tolist(), model)

    # Tkinter GUI setup
    def generate_response():
        user_input = input_text.get("1.0", tk.END).strip()
        if not user_input:
            messagebox.showerror("Error", "Please enter a letter.")
            return

        # Generate embedding for user input
        input_embedding = model.encode(user_input, convert_to_tensor=True)

        # Find the closest match
        closest_idx, similarity = find_closest_match(input_embedding, dataset_embeddings)

        # Display the response
        response = dataset['Response'].iloc[closest_idx]
        response_text.delete("1.0", tk.END)
        response_text.insert(tk.END, response)

    root = tk.Tk()
    root.title("Offline Letter Response Generator")

    tk.Label(root, text="Enter your letter:").pack()
    input_text = tk.Text(root, height=10, width=50)
    input_text.pack()

    tk.Button(root, text="Generate Response", command=generate_response).pack()

    tk.Label(root, text="Generated Response:").pack()
    response_text = tk.Text(root, height=10, width=50)
    response_text.pack()

    root.mainloop()

if __name__ == "__main__":
    main()