# FAQ Assistant

A simple Python-based FAQ Assistant leveraging DistilBERT for semantic understanding of questions.
This assistant matches user queries to a set of pre-defined questions and returns the most relevant answer with confidence.

---

## Features

- Semantic search of FAQ questions using **DistilBERT** embeddings.
- Configurable similarity threshold for determining matches.
- Handles multiple FAQ entries with ease.

---

## Requirements

Ensure you have the following Python packages installed:

- [Transformers](https://huggingface.co/docs/transformers): For using the DistilBERT model.
- [Torch](https://pytorch.org/): For processing model inputs and outputs.
- [Scikit-learn](https://scikit-learn.org/): For cosine similarity calculations.
- [NumPy](https://numpy.org/): For handling embeddings.
- [Pandas](https://pandas.pydata.org/): (Optional) If you plan to use data frames.

Install dependencies using pip:

```bash
pip install transformers torch scikit-learn numpy pandas
```

---

## Setup

1. Clone or download the repository containing the code.
2. Save the script as `main.py`.

---

## Usage

### Step 1: Initialize the Assistant

```python
assistant = Assistant()
```

### Step 2: Add FAQ Data

Provide a list of question-answer pairs as a tuple data type:

```python
faqs = [
    ("How do I reset my password?", "Click the 'Forgot Password' link on the login page and follow the email instructions."),
    ("Where can I find my account settings?", "Account settings are in the top-right menu under your profile icon."),
    ("What payment methods do you accept?", "We accept credit cards, PayPal, and bank transfers."),
]
assistant.add_data(faqs)
```

### Step 3: Ask Questions

Input a user query and get a response:

```python
answer, confidence = assistant.answer_question("Do you take PayPal?")
print(f"Answer: {answer} | Confidence: {confidence}")
```

If the question doesn't meet the similarity threshold, currently set to `0.7 by default`:

```plaintext
"I am as confused as you are right now!"
```

---

## How It Works

1. **Data Loading**: FAQ data (questions and answers) is added and converted into embeddings.
2. **Embeddings**: Questions are processed into numeric embeddings using DistilBERT.
3. **Similarity Search**: The cosine similarity between user query embeddings and FAQ embeddings determines the most relevant question.

---

## Code Comments Explained

### `torch.no_grad()`

This is a PyTorch context manager used to disable gradient calculations.
It reduces memory usage and speeds up inference, as gradients are unnecessary for model evaluation.

### `outputs.last_hidden_state[:, 0, :]`

- **`last_hidden_state`**: Outputs the embeddings for all tokens in the input sequence.
- **`[:, 0, :]`**: Extracts the embedding for the **[CLS] token**, which is often used as the representation of the entire input sequence.

### `cosine_similarity`

Calculates the similarity between two embeddings. The result is a value between -1 (completely dissimilar) and 1 (identical).

### Thresholding

The `threshold` parameter (default: `0.7`) ensures the assistant only returns answers with a sufficient confidence level.

---

## Sample Output

```plaintext
Answer: We accept credit cards, PayPal, and bank transfers. | Confidence: 0.87
```

---

## Future Enhancements

- Support for dynamic data loading from files (e.g., CSV, JSON).
- Integration with web frameworks for API-based queries.
- Fine-tuning of DistilBERT on custom FAQ datasets.

---

Let us know if you need further clarifications!
