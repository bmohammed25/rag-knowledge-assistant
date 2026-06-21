from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores import Chroma
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
import shutil
import os

loader = PyPDFLoader("data/Transformer_Attention_paper.pdf")
documents = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=15)
chunks = splitter.split_documents(documents)
embeddings = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(chunks,embeddings)
if os.path.exists("data/chroma_db"):
    shutil.rmtree("data/chroma_db")
chroma_store = Chroma.from_documents(chunks, embeddings, persist_directory="data/chroma_db")
llm = ChatOllama(model="phi3:mini")
prompt = ChatPromptTemplate.from_template("""
Answer the question based only on the following context. If the context doesn't contain the answer, say you don't know.

Context:
{context}

Question:
{question}
""")
retriever = vectorstore.as_retriever(sear_kwargs={"k":3})
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
)

# results = vectorstore.similarity_search("What is self-attention?", k=3)
# for i, result in enumerate(results):
#     print(f"\n--- Result {i+1} ---")
#     print(result.page_content[:200])

# print("\n\n=== ChromaDB Results ===")
# chroma_results = chroma_store.similarity_search("What is self-attention?", k=3)

# for i, result in enumerate(chroma_results):
#     print(f"\n--- Result {i+1} ---")
#     print(result.page_content[:200])

print("\n\n=== RAG Answer ===")
answer = chain.invoke("What is self-attention?")
print(answer.content)