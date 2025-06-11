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

# Load and preprocess the dataset from a folder
def load_dataset_from_folder(folder_path):
    # Find all .docx files in the folder
    files = [f for f in os.listdir(folder_path) if f.lower().endswith('.docx')]
    # Try to pair files as letter/response by name pattern
    # Example: letter001.docx and response001.docx, or any two files as pairs
    # We'll use a simple approach: sort files, pair every two
    files.sort()
    letters = []
    responses = []
    for i in range(0, len(files)-1, 2):
        letter_path = os.path.join(folder_path, files[i])
        response_path = os.path.join(folder_path, files[i+1])
        # Extract text
        letter_text = extract_text_from_word(letter_path)
        response_text = extract_text_from_word(response_path)
        letters.append(letter_text)
        responses.append(response_text)
    # Return as a DataFrame for compatibility
    return pd.DataFrame({'Letter': letters, 'Response': responses})

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

# Update the GPT-4All model initialization to use a Mistral 7B model (smaller/faster)
try:
    model_path = os.path.join(os.getcwd(), "resources", "mistral-7b-instruct-v0.1.Q4_K_M.gguf")  # Use Mistral 7B model for text generation
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}. Please download a Mistral 7B GGUF model and place it in the resources folder.")
    gpt_model = GPT4All(model_path)
except (ValueError, FileNotFoundError) as e:
    logging.error(f"Failed to load GPT-4All model: {e}")
    logging.info("Please ensure the Mistral 7B model file is available locally or check the model name.")
    exit()

# Main function to integrate the steps
def main():
    # Load the dataset from the folder
    dataset_folder = 'letters_dataset'
    try:
        dataset = load_dataset_from_folder(dataset_folder)
    except Exception as e:
        print(f"Error loading dataset from folder: {e}")
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
    def prompt_for_blanks(text):
        import re
        blanks = list(re.finditer(r"-{3,}", text))
        filled_text = text
        for i, match in enumerate(blanks):
            answer = tk.simpledialog.askstring("Fill in the blank", f"Please provide a value for blank #{i+1}:")
            if answer is None:
                answer = "[Not Provided]"
            filled_text = filled_text.replace(match.group(), answer, 1)
        return filled_text

    def generate_response():
        user_input = input_text.get("1.0", tk.END).strip()
        if not user_input:
            messagebox.showerror("Error", "Please enter a letter.")
            return

        # Fill in blanks in the input letter if present
        if '---' in user_input:
            user_input = prompt_for_blanks(user_input)
            input_text.delete("1.0", tk.END)
            input_text.insert(tk.END, user_input)

        # Generate embedding for user input
        input_embedding = model.encode(user_input, convert_to_tensor=True)

        # Find the closest match
        closest_idx, similarity = find_closest_match(input_embedding, dataset_embeddings)

        # Get the response content from the dataset
        response_content = dataset['Response'].iloc[closest_idx]

        # Fill in blanks in the response template if present
        if '---' in response_content:
            response_content = prompt_for_blanks(response_content)

        # Use GPT-4All to expand the response
        try:
            from gpt4all import GPT4All
            gpt = GPT4All(model_path)
            # Generate a detailed response with enhanced prompt
            def build_prompt(resp_content):
                return ("Expand the following response into a detailed and professional draft letter:\n\n"
                        f"{resp_content}\n\n"
                        "Ensure the letter includes:\n"
                        "1. A polite and professional greeting.\n"
                        "2. A clear and concise introduction.\n"
                        "3. A well-structured body with detailed explanations or justifications.\n"
                        "4. A courteous closing statement.\n"
                        "5. Proper formatting and tone suitable for formal communication.")
            max_prompt_length = 2000
            prompt = build_prompt(response_content)
            # If too long, truncate response_content to fit
            if len(prompt) > max_prompt_length:
                allowed_resp_len = max_prompt_length - len(build_prompt(""))
                response_content = response_content[:allowed_resp_len]
                prompt = build_prompt(response_content)
            # Increase token limit and log raw output
            detailed_response = gpt.generate(prompt, max_tokens=500)
            logging.info(f"Raw model output: {detailed_response}")
            # Fallback for empty or trivial model output
            if not detailed_response or detailed_response.strip() == '' or detailed_response.strip().lower() in ['dear [recipient],', 'sincerely,', '[your name]']:
                messagebox.showerror("Error", "The model did not generate a meaningful response. Please check your dataset, use a different prompt, or ensure your template is not too short.")
                return
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