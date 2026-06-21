import gradio as gr
import requests

def ask_question(question):
    response = requests.get("http://127.0.0.1:8000/ask", params={"question": question})
    data = response.json()
    return data["answer"]

demo = gr.Interface(
    fn=ask_question,
    inputs=gr.Textbox(label="Ask a question about the paper"),
    outputs=gr.Textbox(label="Answer"),
    title="RAG Knowledge Assistant",
    description="Ask questions about the 'Attention Is All You Need' paper."
)

demo.launch()