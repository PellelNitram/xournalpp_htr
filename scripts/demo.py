import gradio as gr
import numpy as np
import os

# This is a placeholder for your actual handwriting recognition model.
# The input 'image' is a NumPy array from the Image component.
def recognize_handwriting(image):
    """
    Placeholder function to simulate handwriting recognition.
    
    Args:
        image (np.ndarray): The image data from the Gradio Image component.
                            It's a NumPy array, where None means no image was provided.
    
    Returns:
        str: A string with the "recognized" text.
    """
    if image is None:
        return "Please load an image first!"
    
    # In a real application, you would process the image here:
    # 1. Preprocess the image (resize, normalize, binarize, etc.).
    # 2. Feed it into your HTR model.
    # 3. Get the predicted text from the model's output.
    print(f"Received image of shape: {image.shape}")
    
    # For now, we'll just return a dummy response.
    return "This is a placeholder for the recognized text."

def load_random_image():
    """
    Generates a random noisy image to simulate loading an image file.
    """
    # Create a random image (height, width, channels)
    random_image = np.random.randint(0, 256, size=(200, 600, 3), dtype=np.uint8)
    # Also return an empty string to clear the output text box upon loading a new image
    return random_image, ""


# --- Gradio Interface Definition ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # Xournal++ Handwriting Recognition
        Click "Load Random Image" to get a new image, then click "Recognize".
        """
    )
    
    with gr.Row(equal_height=True):
        # Input component: An image display
        with gr.Column(scale=2):
            image_display = gr.Image(
                label="Input Image",
                show_label=True,
                type="numpy"
            )
        
        # Output and actions column
        with gr.Column(scale=1):
            output_text = gr.Textbox(
                label="Recognized Text",
                show_label=True,
                interactive=False, # User should not edit the output
                lines=5,
                placeholder="Recognition result will appear here..."
            )
            load_button = gr.Button("Load Random Image", variant="secondary")
            recognize_button = gr.Button("Recognize", variant="primary")
            
    # Connect the buttons to the functions
    load_button.click(
        fn=load_random_image, 
        inputs=None, 
        outputs=[image_display, output_text]
    )
    
    recognize_button.click(
        fn=recognize_handwriting, 
        inputs=image_display, 
        outputs=output_text
    )

if __name__ == "__main__":
    # Use the PORT environment variable if available (for Hugging Face Spaces), otherwise default to 7860
    port = int(os.environ.get("PORT", 7860))
    
    # Launch the Gradio app
    demo.launch(server_name="0.0.0.0", server_port=port)






# import os

# import gradio as gr

# from xournalpp_htr.documents import Stroke

# # TODO: Add gradio demo here.
# # - it needs to run locally
# # - it needs to run in a docker container that's compatible w/ huggingface space


# def greet(name):
#     s = Stroke  # Just for testing import
#     print(s)
#     return f"Hello, {name}!"


# with gr.Blocks() as demo:
#     gr.Markdown("# Greeting App")
#     with gr.Row():
#         name_input = gr.Textbox(
#             label="Enter your name", placeholder="Type your name here..."
#         )
#         output = gr.Textbox(label="Greeting")
#     greet_button = gr.Button("Greet")
#     greet_button.click(fn=greet, inputs=name_input, outputs=output)

# if __name__ == "__main__":
#     port = int(os.environ.get("PORT", 7860))  # Use HF-provided port or fallback
#     demo.launch(server_name="0.0.0.0", server_port=port)
