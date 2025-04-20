import gradio as gr
import requests
import os

API_URL = os.getenv("FACTCHECK_API_URL", "http://localhost:8000/fact_check")


def fact_check_interface(query, use_vector_db, use_tavily):
    payload = {
        "query": query,
        "use_vector_db": use_vector_db,
        "use_tavily": use_tavily,
    }
    try:
        response = requests.post(API_URL, json=payload, timeout=120)
        response.raise_for_status()
        data = response.json()
        verdict = data.get("verdict", "Tidak diketahui")
        explanation = data.get("explanation", "")
        sources = data.get("sources", [])
        sources_str = "\n".join(sources) if sources else "Tidak ada sumber."
        return verdict, explanation, sources_str
    except Exception as e:
        return "Error", f"API error: {e}", ""


with gr.Blocks() as demo:
    gr.Markdown("# Hoax News Fact Checking Demo")
    with gr.Row():
        query = gr.Textbox(label="Masukkan judul atau isi berita", lines=4)
    with gr.Row():
        use_vector_db = gr.Checkbox(label="Gunakan Database Hoax", value=True)
        use_tavily = gr.Checkbox(label="Gunakan Web Search (Tavily)", value=True)
    with gr.Row():
        btn = gr.Button("Cek Fakta")
    with gr.Row():
        verdict = gr.Textbox(label="Putusan", interactive=False)
    with gr.Row():
        explanation = gr.Textbox(label="Penjelasan", lines=4, interactive=False)
    with gr.Row():
        sources = gr.Textbox(label="Sumber", lines=4, interactive=False)

    btn.click(
        fact_check_interface,
        inputs=[query, use_vector_db, use_tavily],
        outputs=[verdict, explanation, sources],
    )

if __name__ == "__main__":
    demo.launch()
