"""
This script performs two main tasks:
1. It loads a markdown document, splits it into chunks, generates embeddings,
   and builds a FAISS index (which is saved locally).
2. It generates QA pairs from the document using llama.
   For each chunk (using a sliding window for context), it generates multiple question-answer pairs
   with different difficulties. The generation is performed in batch with one retry for failed prompts.
   Successfully generated QA pairs are saved to "saved_data/questions.json".

Requirements:
    pip install langchain faiss-cpu unsloth vllm
"""

import os
import re
import json
import pickle
from typing import List, Tuple, Optional, Dict

# ========= Part 1: Document Processing and Embedding Generation =========

# Load and split the markdown document using LangChain
from langchain.document_loaders import UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from embeddings import CustomHuggingFaceEmbeddings

# Load your markdown file (adjust the path as needed)
loader = UnstructuredMarkdownLoader("./data/env_sci.md")
docs = loader.load()

print(f"Number of initial documents: {len(docs)}")

print(f"Loaded content length: {len(docs[0].page_content)}")
print(f"First 500 characters: {docs[0].page_content[:500]}")

# Split the document into smaller chunks (each 1000 characters, no overlap)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
chunks = text_splitter.split_documents(docs)

total_chars = sum(len(chunk.page_content) for chunk in chunks)
print(f"Total characters in chunks: {total_chars}")

# Save chunks for later use
os.makedirs("saved_data", exist_ok=True)
with open("saved_data/chunks.pkl", "wb") as f:
    pickle.dump(chunks, f)
print(f"Saved {len(chunks)} chunks to saved_data/chunks.pkl")

embeddings = CustomHuggingFaceEmbeddings()

# Create a FAISS vector store from the document chunks and save it locally
vectorstore = FAISS.from_documents(chunks, embeddings)
vectorstore.save_local("faiss_index")
print("Saved FAISS index to 'faiss_index'")

# ========= Part 2: QA Generation using OpenAI API =========

# Setup OpenAI backend via LangChain
from langchain_mistralai import ChatMistralAI
from langchain_core.messages import HumanMessage, SystemMessage

# Load OpenAI API key from environment variable
mistral_api_key = os.getenv("MISTRAL_API_KEY")
if not mistral_api_key:
    raise ValueError("MISTRAL_API_KEY environment variable not set")

llm = ChatMistralAI(
    model_name="mistral-large-2411",  # Use the correct model name
    temperature=0.2,
    max_retries=10,
    max_tokens=8192,  # Reduced to a safer value; adjust as needed
    top_p=0.95,
)

def batch_generate(prompts: List[str]) -> List[str]:
    """
    Given a list of prompt strings, returns a list of generated outputs using OpenAI chat model.
    """
    # Format each prompt as a chat message
    batch_messages = [[HumanMessage(content=prompt)] for prompt in prompts]
    result = llm.generate(batch_messages)
    return [result.generations[i][0].text for i in range(len(prompts))]

def parse_qa_block(block: str) -> Optional[Tuple[str, str, str]]:
    """
    Parses a QA block that should contain exactly three non-empty lines:
      - A line starting with "Question:"
      - A line starting with "Answer:"
      - A line starting with "Difficulty:"
    
    If the markers are not present but the block contains exactly three lines,
    those are used in order.
    
    Returns a tuple (question, answer, difficulty) or None if parsing fails.
    """
    lines = [line.strip() for line in block.splitlines() if line.strip()]
    if not lines:
        return None

    question, answer, difficulty = None, None, None
    for line in lines:
        lower = line.lower()
        if question is None and lower.startswith("question:"):
            question = line[len("question:"):].strip()
        elif answer is None and lower.startswith("answer:"):
            answer = line[len("answer:"):].strip()
        elif difficulty is None and lower.startswith("difficulty:"):
            difficulty = line[len("difficulty:"):].strip()

    if question and answer and difficulty:
        return question, answer, difficulty
    if len(lines) == 3:
        return lines[0], lines[1], lines[2]
    return None

def parse_multiple_qa_output(output: str) -> List[Tuple[str, str, str]]:
    """
    Splits the output into blocks (separated by one or more blank lines) and
    attempts to parse each as a QA pair.
    
    Returns a list of successfully parsed QA tuples.
    """
    blocks = re.split(r'\n\s*\n', output.strip())
    qa_pairs = []
    for block in blocks:
        parsed = parse_qa_block(block)
        if parsed:
            qa_pairs.append(parsed)
    return qa_pairs

def generate_question_batch_for_chunks(chunks: List, num_questions: int = 2, difficulty: str = None) -> List[Dict]:
    """
    Generates QA pairs for multiple chunks in batch.
    
    For each chunk (except the first and last), a sliding window is used for context:
      - before: previous chunk's content
      - current: current chunk's content
      - after: next chunk's content
    
    Each prompt instructs the model to output exactly three lines per QA pair with markers.
    Failed prompts are retried once in batch; if still unsuccessful, they are skipped.
    
    Returns a list of dicts with keys: "chunk_id", "question", "answer", "difficulty".
    """
    prompts = []
    chunk_ids = []
    
    # Prepare prompts using a sliding window
    for i in range(1, len(chunks) - 1):
        before = chunks[i-1].page_content
        current = chunks[i].page_content
        after = chunks[i+1].page_content
        prompt = (
            f"From the text within ==BEGIN== and ==END==, generate {num_questions} questions with answers.\n"
            "For each QA pair, output exactly three lines with no extra commentary:\n"
            "Line 1: Question: <your question>\n"
            "Line 2: Answer: <the answer>\n"
            "Line 3: Difficulty: <easy, medium, or hard>\n"
            "Do not include any additional text.\n\n"
            "==BEGIN==\n"
            f"{before}\n{current}\n{after}\n"
            "==END==\n"
        )
        prompts.append(prompt)
        chunk_ids.append(i)
    
    # First batch generation
    outputs = batch_generate(prompts)
    results = [None] * len(outputs)
    failed_indices = []
    
    # Parse each output
    for idx, output in enumerate(outputs):
        qa_pairs = parse_multiple_qa_output(output)
        if qa_pairs is None or len(qa_pairs) < num_questions:
            failed_indices.append(idx)
        else:
            results[idx] = qa_pairs[:num_questions]
    
    # Retry failed prompts in batch
    if failed_indices:
        print(f"Retrying {len(failed_indices)} failed prompt(s)...")
        retry_prompts = [prompts[i] for i in failed_indices]
        retry_outputs = batch_generate(retry_prompts)
        for j, idx in enumerate(failed_indices):
            qa_pairs = parse_multiple_qa_output(retry_outputs[j])
            if qa_pairs is not None and len(qa_pairs) >= num_questions:
                results[idx] = qa_pairs[:num_questions]
            else:
                results[idx] = None  # Mark as failed
    
    # Build final output, skipping prompts that failed even after retry
    final_questions = []
    for i, qa_list in enumerate(results):
        if qa_list is not None:
            for qa in qa_list:
                final_questions.append({
                    "chunk_id": chunk_ids[i],
                    "question": qa[0],
                    "answer": qa[1],
                    "difficulty": qa[2]
                })
    return final_questions

# Generate QA pairs in batch (using a sliding window over the chunks)
all_questions = generate_question_batch_for_chunks(chunks, num_questions=8, difficulty="medium")
print(f"Generated {len(all_questions)} QA pairs.")

# Save the QA pairs to a JSON file
questions_path = os.path.join("saved_data", "questions.json")
with open(questions_path, "w") as f:
    json.dump(all_questions, f, indent=2)
print(f"Saved questions to {questions_path}")
