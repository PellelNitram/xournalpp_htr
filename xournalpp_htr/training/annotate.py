"""
This script helps to annotate Xournal(++) files.
"""

# =========================
# EXPERIMENT W/ FILE UPLOAD
# =========================

# Docs: https://www.gradio.app/docs/gradio/uploadbutton

# ---------
# Attempt 1
# ---------

# import gradio as gr


# def greet(name, intensity):
#     return "Hello, " + name + "!" * int(intensity)


# demo = gr.Interface(
#     fn=greet,
#     inputs=["text", "slider"],
#     outputs=["text"],
# )

# demo.launch()

# ---------
# Attempt 2
# ---------

# import gradio as gr


# def upload_file(files):
#     file_paths = [file.name for file in files]
#     for x in file_paths:
#         print("uf", x)
#     return file_paths


# with gr.Blocks() as demo:
#     file_output = gr.File()
#     upload_button = gr.UploadButton(
#         "Click to Upload a File", file_types=["image", "video"], file_count="multiple"
#     )
#     abc = upload_button.upload(upload_file, upload_button, file_output)

#     print("demo", file_output)

#     print("abc", abc)

# demo.launch()

# ==========================================
# Experiment w/ annotation of bounding boxes
# ==========================================

# TODO: Do research if that's possible!
