import gradio as gr

with gr.Blocks(fill_height=True) as demo:
    with gr.Sidebar():
        gr.Markdown("# Inference Provider")
        gr.Markdown("This Space showcases the distilbert/distilbert-base-uncased-finetuned-sst-2-english model, served by the hf-inference API. Sign in with your Hugging Face account to use this API.")
        button = gr.LoginButton("Sign in")
    gr.load("models/distilbert/distilbert-base-uncased-finetuned-sst-2-english", accept_token=button, provider="hf-inference")
    
demo.launch()