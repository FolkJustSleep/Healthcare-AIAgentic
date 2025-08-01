#contexual retrival with prompt chaning feedback and reduce latency
import chromadb
from chromadb.utils.embedding_functions import EmbeddingFunction
from openai import OpenAI
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.chat_models import ChatOpenAI
from huggingface_hub import InferenceClient
import os
import hashlib
import time
from requests.exceptions import HTTPError
# Load environment variables
load_dotenv()
hunggingface_key = os.getenv("HF_TOKEN")  # Hugging Face token for embedding
# client = openai.OpenAI(
#     api_key="sk-DmbMLFXsAWYQkCkxDTmuLHeVssgnbbVT9lxKneY5YMNi5UNu",
#     base_url="https://api.opentyphoon.ai/v1"
# )
llm = ChatOpenAI(model="typhoon-v2-70b-instruct", temperature=0.1, api_key=os.getenv("OPENAI_API_KEY_MCP"), base_url="https://api.opentyphoon.ai/v1")
   
#client = OpenAI(api_key=openai_key)
hf_client = InferenceClient(
    provider="hf-inference",
    api_key=hunggingface_key,
)
# === Define embedding function using Hugging Face ===
class E5EmbeddingFunction:
    def __init__(self, hf_client):
        self.client = hf_client

    def __call__(self, input):
        inputs = [f"passage: {text}" for text in input]
        return self.client.feature_extraction(
            inputs,
            model="intfloat/multilingual-e5-large"
        )

    def name(self):
        return "e5-huggingface-inference"

# Paths
DATA_PATH = r"data"
CHROMA_PATH = r"chroma_db"
print(f"Using data path: {DATA_PATH}")
# Load and process PDF documents
loader = PyPDFDirectoryLoader(DATA_PATH)
documents = loader.load()
print(f"Loaded {len(documents)} documents from {DATA_PATH}")
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", " ", ""]
)
print("Splitting documents into chunks...")
chunks = splitter.split_documents(documents)

# Setup ChromaDB
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
embedding_fn = E5EmbeddingFunction(
    hf_client=hf_client
)
collection = chroma_client.get_or_create_collection(name="contextual_chunks", embedding_function=embedding_fn)

DOCUMENT_CONTEXT_PROMPT = """
<document>
{doc_content}
</document>
"""
CHUNK_CONTEXT_PROMPT = """
Here is the chunk we want to situate within the whole document
<chunk>
{chunk_content}
</chunk>

Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk.
Answer only with the succinct context and nothing else.
"""

def situate_context(doc: str, chunk: str) -> str:
    prompt = DOCUMENT_CONTEXT_PROMPT.format(doc_content=doc) + "\n\n" + CHUNK_CONTEXT_PROMPT.format(chunk_content=chunk)
    response = llm.invoke(
        prompt, max_tokens=200, temperature=0.1
    )
    return response.content

full_doc_text = "\n\n".join([d.page_content for d in documents])
existing_ids = set(collection.get()["ids"])

for idx, doc_chunk in enumerate(chunks):
    chunk_text = doc_chunk.page_content
    uid = f"pdf_{idx}_" + hashlib.md5(chunk_text.encode("utf-8")).hexdigest()

    if uid in existing_ids:
        print(f"[!] Skipping chunk {idx} (already exists)")
        continue

    try:
        context = situate_context(chunk_text[:1500], chunk_text)
        full_chunk = context.strip() + "\n\n" + chunk_text
        collection.upsert(
            ids=[uid],
            documents=[full_chunk],
            metadatas=[{"chunk_index": idx, "context": context}]
        )
        print(f"[\u2713] Added chunk {idx}")
        time.sleep(5)
    except Exception as e:
        print(f"[\u2717] Failed to add chunk {idx}: {e}")
        time.sleep(3)

def ask_llm(messages) -> str:
    response = llm.invoke(messages, max_tokens=200, temperature=0.1)
    return response.content

def search_chunks(query, top_k=3):
    query_embedding =  "query: " + query
    results = collection.query(query_texts=[query_embedding], n_results=top_k)
    return results['documents'][0]

def generate_answer_with_feedback(query) -> str:
    chunks = search_chunks(query)
    context = "\n\n".join(chunks)
    messages = [
        {"role": "system", "content": "You are an expert assistant that only uses the provided documents."},
        {"role": "user", "content": f"""Answer this question using only the following context:

Context:
---------
{context}

Question:
{query}
"""}
    ]
    initial_answer = ask_llm(messages)

    feedback_prompt = [
        {"role": "system", "content": "You are a meta-reasoning assistant."},
        {"role": "user", "content": f"""
Here is the question: "{query}"

Here is the answer:
---------------------
{initial_answer}

Was the answer fully supported by the context? If not, suggest an improved search query. If yes, say: "Answer is sufficient and well-supported."
"""}
    ]
    feedback = ask_llm(feedback_prompt)

    if "suggest" in feedback.lower() or "improve" in feedback.lower():
        print(f"Feedback suggests refining query:\n{feedback}\n")
        refined_query = feedback.split("query")[-1].strip(":.\n\" ")
        new_chunks = search_chunks(refined_query)
        new_context = "\n\n".join(new_chunks)

        new_answer_prompt = [
            {"role": "system", "content": "You are an expert assistant that only uses the provided documents."},
            {"role": "user", "content": f"""Answer this question using the refined context:

Context:
---------
{new_context}

Original Question:
{query}
"""}
        ]
        final_answer = ask_llm(new_answer_prompt)
        return final_answer
    else:
        print(" Initial answer was sufficient.")
        return initial_answer
while True:
        user_input = input("You: ")
        response = generate_answer_with_feedback(user_input)
        print(f"AI: {response}")
        if user_input.lower() == "exit" or user_input.lower() == "quit":
            break