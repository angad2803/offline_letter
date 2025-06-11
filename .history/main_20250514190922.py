import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from gpt4all import GPT4All
import tkinter as tk
from tkinter import scrolledtext

# --- Load sentence embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# --- Load dataset
df = pd.read_csv("dataset.csv")

# --- GPT4All LLM Setup
llm_path = "models/mistral-7b-instruct.gguf"
gpt_model = GPT4All(model_name=llm_path)

# --- Function: Prepare Training Prompt from Dataset
def prepare_training_prompt(data):
    prompt = "You are an assistant that helps with letter writing. Below are some examples of letters and responses.\n"
    for _, row in data.iterrows():
        prompt += f"\nLetter: {row['input_letter']}\nResponse: {row['response']}\n"
    return prompt

# --- Function: Generate response using GPT-4 (LLM)
def generate_response_llm(input_text, training_prompt):
    prompt = f"{training_prompt}\n\nNow, based on the examples above, write a letter for the following:\nLetter: {input_text}\nResponse:"

    with gpt_model.chat_session():
        output = gpt_model.generate(prompt, max_tokens=300)
    return output.strip()

# --- GUI Setup with Tkinter
def on_submit():
    user_input = entry.get()
    if user_input.strip().lower() == 'exit':
        root.quit()

    # Prepare training prompt based on dataset
    training_prompt = prepare_training_prompt(df)

    # Generate a new letter based on the input and learned examples
    response = generate_response_llm(user_input, training_prompt)

    # Display results in the text area
    output_text.config(state=tk.NORMAL)  # Enable text box for editing
    output_text.delete(1.0, tk.END)  # Clear previous output
    output_text.insert(tk.END, f"Input Letter:\n{user_input}\n\nGenerated Response:\n{response}\n")
    output_text.config(state=tk.DISABLED)  # Disable text box for editing

# --- Setting up Tkinter window
root = tk.Tk()
root.title("Offline Letter Response App")

# Input text
entry_label = tk.Label(root, text="Enter your letter:")
entry_label.pack(pady=5)

entry = tk.Entry(root, width=50)
entry.pack(pady=10)

# Submit button
submit_button = tk.Button(root, text="Generate Response", command=on_submit)
submit_button.pack(pady=10)

# Output text box
output_text = scrolledtext.ScrolledText(root, width=60, height=15, wrap=tk.WORD, state=tk.DISABLED)
output_text.pack(pady=10)

root.mainloop()
