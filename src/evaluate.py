from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
import re

def extract_score(text):
    match = re.search(r'\d+', text)
    if match:
        return int(match.group())
    return None

eval_questions = [
    "What is the main architecture proposed in this paper?",
    "What is self-attention?",
    "Why did the authors choose attention over recurrent or convolutional layers?",
    "What are the two sub-layers in each encoder layer?",
    "What is multi-head attention?"
]

ground_truths = [
    "The Transformer, a model architecture based entirely on attention mechanisms, dispensing with recurrence and convolutions entirely.",
    "Self-attention is an attention mechanism relating different positions of a single sequence to compute a representation of that sequence.",
    "Self-attention layers connect all positions with a constant number of operations, are more parallelizable, and have shorter path lengths between long-range dependencies compared to recurrent layers.",
    "A multi-head self-attention mechanism, and a simple position-wise fully connected feed-forward network.",
    "Multi-head attention runs multiple attention operations in parallel on different learned linear projections of the queries, keys, and values, then concatenates and projects the results."
]

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma(persist_directory="data/chroma_db", embedding_function=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
llm = ChatOllama(model="phi3:mini")

prompt = ChatPromptTemplate.from_template("""
Answer the question based only on the following context. If the context doesn't contain the answer, say you don't know.

Context:
{context}

Question:
{question}
""")

faithfulness_prompt = ChatPromptTemplate.from_template("""
You are evaluating an AI-generated answer for faithfulness to its source context.

Context:
{context}

Answer:
{answer}

Does the answer contain ONLY information that is supported by the context? Respond with a single number from 0 to 10, where 10 means completely faithful (no unsupported claims) and 0 means completely unfaithful (mostly made up). Respond with ONLY the number, nothing else.
""")

relevance_prompt = ChatPromptTemplate.from_template("""
You are evaluating whether an AI-generated answer actually addresses the question asked.

Question:
{question}

Answer:
{answer}

Does the answer directly address what was asked? Respond with a single number from 0 to 10, where 10 means perfectly relevant and 0 means completely off-topic. Respond with ONLY the number, nothing else.
""")

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
)

results = {
    "question": [],
    "answer": [],
    "contexts": [],
    "ground_truth": []
}

for question, ground_truth in zip(eval_questions, ground_truths):
    retrieved_docs = retriever.invoke(question)
    contexts = [doc.page_content for doc in retrieved_docs]

    answer = chain.invoke(question)

    results["question"].append(question)
    results["answer"].append(answer.content)
    results["contexts"].append(contexts)
    results["ground_truth"].append(ground_truth)

    print(f"Processed: {question}")

faithfulness_scores = []
relevance_scores = []

for i in range(len(results["question"])):
    question = results["question"][i]
    answer = results["answer"][i]
    contexts = results["contexts"][i]
    context_text = "\n".join(contexts)

    faithfulness_chain = faithfulness_prompt | llm
    faithfulness_result = faithfulness_chain.invoke({"context": context_text, "answer": answer})
    faithfulness_score = extract_score(faithfulness_result.content)
    faithfulness_scores.append(faithfulness_score)

    relevance_chain = relevance_prompt | llm
    relevance_result = relevance_chain.invoke({"question": question, "answer": answer})
    relevance_score = extract_score(relevance_result.content)
    relevance_scores.append(relevance_score)

    print(f"Scored: {question} -> Faithfulness: {faithfulness_score}, Relevance: {relevance_score}")

avg_faithfulness = sum(faithfulness_scores) / len(faithfulness_scores)
avg_relevance = sum(relevance_scores) / len(relevance_scores)

print("\n=== Custom Evaluation Results ===")
print(f"Average Faithfulness: {avg_faithfulness:.2f} / 10")
print(f"Average Relevance: {avg_relevance:.2f} / 10")
print(f"\nPer-question scores:")
for i in range(len(results["question"])):
    print(f"  {results['question'][i]}")
    print(f"    Faithfulness: {faithfulness_scores[i]}, Relevance: {relevance_scores[i]}")
