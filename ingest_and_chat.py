from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core import StorageContext
from llama_index.vector_stores.faiss import FaissVectorStore
import faiss
import os

# models 
Settings.llm = Ollama(
    model="qwen2.5:7b-instruct-q5_k_m",
    request_timeout=600.0,
    options={"num_ctx": 8192}
)

Settings.embed_model = OllamaEmbedding(
    model_name="nomic-embed-text"
)

# Load documents
# For testing: local folder
documents = SimpleDirectoryReader("test").load_data()

# For real Google Drive
# from llama_index.readers.google import GoogleDriveReader

# loader = GoogleDriveReader(
#     folder_id="your_folder_id_here",   # ← put your real folder ID
#     recursive=True
# )
# documents = loader.load_data()

# FAISS vector store
# Get embedding dimension automatically (usually 768 for nomic-embed-text)
test_embedding = Settings.embed_model.get_text_embedding("test sentence")
dimension = len(test_embedding)

faiss_index = faiss.IndexFlatL2(dimension)
# For better quality (recommended) you can use:
# faiss_index = faiss.IndexIVFFlat(faiss_index, dimension, 100)  # needs training

vector_store = FaissVectorStore(faiss_index=faiss_index)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# Build index (only once or when documents change) ─────────────
index = VectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context,
    show_progress=True
)

# Optional: Save index for reuse
faiss.write_index(faiss_index, "my_index.faiss")

# Later you can load it like this:
# faiss_index = faiss.read_index("my_index.faiss")
# vector_store = FaissVectorStore(faiss_index=faiss_index)
# storage_context = StorageContext.from_defaults(vector_store=vector_store)
# index = VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context)

query_engine = index.as_query_engine()
response = query_engine.query("Quantos Serviços teria o modelo de negócio descrito no documento Blackout.docx?")
print(response)