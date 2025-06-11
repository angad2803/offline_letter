import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import tkinter as tk
from tkinter import messagebox, filedialog
import logging
import os
import pickle
from docx import Document
from PyPDF2 import PdfReader
from gpt4all import GPT4All

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load and preprocess the dataset
def load_dataset(file_path):
    df = pd.read_csv(file_path)
    if 'Letter' not in df.columns or 'Response' not in df.columns:
        raise ValueError("Dataset must contain 'Letter' and 'Response' columns.")
    return df

# Generate embeddings using SentenceTransformer
def generate_embeddings(texts, model):
    return model.encode(texts, convert_to_tensor=True)

# Save precomputed embeddings to a file
def save_embeddings(embeddings, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(embeddings, f)

# Load precomputed embeddings from a file
def load_embeddings(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    return None

# Find the closest match using cosine similarity
def find_closest_match(input_embedding, dataset_embeddings):
    similarities = cosine_similarity([input_embedding], dataset_embeddings)
    closest_idx = similarities.argmax()
    return closest_idx, similarities[0][closest_idx]

# Function to extract text from a Word file
def extract_text_from_word(file_path):
    doc = Document(file_path)
    return '\n'.join([paragraph.text for paragraph in doc.paragraphs])

# Function to extract text from a PDF file
def extract_text_from_pdf(file_path):
    reader = PdfReader(file_path)
    return '\n'.join([page.extract_text() for page in reader.pages])

# Update the GPT-4All model initialization to use the alternative model file
try:
    model_path = os.path.join(os.getcwd(), "resources", "nomic-embed-text-v1.5.f16.gguf")  # Ensure the model file is in the correct directory
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}. Please ensure it exists.")
    gpt_model = GPT4All(model_path)
except (ValueError, FileNotFoundError) as e:
    logging.error(f"Failed to load GPT-4All model: {e}")
    logging.info("Please ensure the model file is available locally or check the model name.")
    exit()

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

    # Check for precomputed embeddings
    embeddings_file = 'dataset_embeddings.pkl'
    dataset_embeddings = load_embeddings(embeddings_file)
    if dataset_embeddings is None:
        logging.info("Generating embeddings for the dataset...")
        dataset_embeddings = generate_embeddings(dataset['Letter'].tolist(), model)
        save_embeddings(dataset_embeddings, embeddings_file)
        logging.info("Embeddings saved to file.")
    else:
        logging.info("Loaded precomputed embeddings from file.")

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

        # Get the response content from the dataset
        response_content = dataset['Response'].iloc[closest_idx]

        # Use GPT-4All to expand the response
        try:
            from gpt4all import GPT4All
            gpt = GPT4All("gpt4all-lora-quantized")
            gpt.open()

            # Generate a detailed response
            prompt = f"Expand the following response into a detailed and professional draft letter:\n\n{response_content}\n\nInclude appropriate greetings, body, and closing."
            detailed_response = gpt.prompt(prompt)
            gpt.close()

            # Format the response as a draft letter
            draft_letter = f"Dear [Recipient],\n\n{detailed_response}\n\nSincerely,\n[Your Name]"
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate response: {e}")
            return

        # Display the formatted response
        response_text.delete("1.0", tk.END)
        response_text.insert(tk.END, draft_letter)

    def save_response():
        user_input = input_text.get("1.0", tk.END).strip()  # Ensure user_input is defined
        response = response_text.get("1.0", tk.END).strip()
        if not response:
            messagebox.showerror("Error", "No response to save.")
            return
        with open("saved_responses.txt", "a", encoding="utf-8") as f:
            f.write(f"Input: {user_input}\nResponse: {response}\n{'-'*50}\n")
        messagebox.showinfo("Success", "Response saved successfully.")

    def upload_file():
        file_path = filedialog.askopenfilename(filetypes=[("Word Files", "*.docx"), ("PDF Files", "*.pdf")])
        if not file_path:
            return
        try:
            if file_path.endswith('.docx'):
                extracted_text = extract_text_from_word(file_path)
            elif file_path.endswith('.pdf'):
                extracted_text = extract_text_from_pdf(file_path)
            else:
                messagebox.showerror("Error", "Unsupported file format.")
                return
            input_text.delete("1.0", tk.END)
            input_text.insert(tk.END, extracted_text)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to extract text: {e}")

    root = tk.Tk()
    root.title("Offline Letter Response Generator")

    tk.Label(root, text="Enter your letter:").pack()
    input_text = tk.Text(root, height=10, width=50)
    input_text.pack()

    tk.Button(root, text="Generate Response", command=generate_response).pack()
    tk.Button(root, text="Save Response", command=save_response).pack()
    tk.Button(root, text="Upload File", command=upload_file).pack()

    tk.Label(root, text="Generated Response:").pack()
    response_text = tk.Text(root, height=10, width=50)
    response_text.pack()

    root.mainloop()

if __name__ == "__main__":
    main()