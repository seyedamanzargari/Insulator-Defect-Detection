import gradio as gr
from ultralytics import YOLO

def predict(model, image):
    model = YOLO(model)
    results = model(image)
    labeled_img = results[0].plot()
    return labeled_img

iface = gr.Interface(
    fn=predict,
    inputs=[
        "text",
        "image"
    ],
    outputs="image"
)

iface.launch()