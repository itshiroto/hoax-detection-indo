import gradio as gr
from hoax_detect.config import settings
from hoax_detect.models import FactCheckRequest
import requests

def fact_check_interface(query: str, use_vector_db: bool, use_tavily: bool) -> tuple[str, str, str]:
    """Gradio interface for fact checking."""
    try:
        response = requests.post(
            settings.FACTCHECK_API_URL,
            json=FactCheckRequest(
                query=query,
                use_vector_db=use_vector_db,
                use_tavily=use_tavily
            ).dict(),
            timeout=120
        )
        response.raise_for_status()
        data = response.json()
        sources = "\n".join(data.get("sources", [])) or "No sources"
        return data.get("verdict", "Unknown"), data.get("explanation", ""), sources
    except Exception as e:
        return "Error", f"API error: {e}", ""

with gr.Blocks(title="Hoax News Fact Checker") as demo:
    gr.Markdown("# Hoax News Fact Checking")
    with gr.Row():
        query = gr.Textbox(
            label="News to Check",
            placeholder="Paste news title or content here...",
            lines=4
        )
    with gr.Row():
        use_vector_db = gr.Checkbox(label="Use Hoax Database", value=True)
        use_tavily = gr.Checkbox(label="Use Web Search", value=True)
    with gr.Row():
        submit_btn = gr.Button("Check Fact", variant="primary")
    with gr.Row():
        verdict = gr.Textbox(label="Verdict", interactive=False)
    with gr.Row():
        explanation = gr.Textbox(label="Explanation", lines=5, interactive=False)
    with gr.Row():
        sources = gr.Textbox(label="Sources", lines=3, interactive=False)

    submit_btn.click(
        fn=fact_check_interface,
        inputs=[query, use_vector_db, use_tavily],
        outputs=[verdict, explanation, sources]
    )

if __name__ == "__main__":
    demo.launch()
