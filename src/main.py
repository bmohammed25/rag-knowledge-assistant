from fastapi import FastAPI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

embeddings = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma(persist_directory = "data/chroma_db" , embedding_function = embeddings)
print(f"Number of chunks in vectorstore: {vectorstore._collection.count()}")
llm = ChatOllama(model="phi3:mini")

prompt = ChatPromptTemplate.from_template("""
Answer the question based only on the following context. If the context doesn't contain the answer, say you don't know.
Context:
{context}

Question:
{question}
""")

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
)

app = FastAPI()
@app.get("/ask")
def ask(question: str):
    answer = chain.invoke(question)
    return {"answer": answer.content}