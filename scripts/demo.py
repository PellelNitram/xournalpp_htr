import os

import gradio as gr

# TODO: Add gradio demo here.
# - it needs to run locally
# - it needs to run in a docker container that's compatible w/ huggingface space


def greet(name):
    return f"Hello, {name}!"


with gr.Blocks() as demo:
    gr.Markdown("# Greeting App")
    with gr.Row():
        name_input = gr.Textbox(
            label="Enter your name", placeholder="Type your name here..."
        )
        output = gr.Textbox(label="Greeting")
    greet_button = gr.Button("Greet")
    greet_button.click(fn=greet, inputs=name_input, outputs=output)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))  # Use HF-provided port or fallback
    demo.launch(server_name="0.0.0.0", server_port=port)
