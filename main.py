import torch
from flask import Flask, render_template, request
import numpy as np
import cv2
import supervision as sv
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import base64
from io import BytesIO
from PIL import Image, ImageDraw, ImageTk
import tkinter as tk
from queue import Queue
import matplotlib.pyplot as plt

app = Flask(__name__)

roi_data = {}  # Global variable to store ROI data

# Queue for inter-thread communication
gui_queue = Queue()

@app.route('/')
def index():
    return render_template('index.html', image_data=None)

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    global roi_data, gui_queue

    if request.method == 'POST':
        if 'image' not in request.files:
            return "No file part"

        file = request.files['image']

        if file.filename == '':
            return "No selected file"

        # Load the image
        image = Image.open(file)
        draw = ImageDraw.Draw(image)
        image_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # Your Tkinter code here
        def handle_click(event):
            global roi_data

            clicks.append((event.x, event.y))

            if len(clicks) == 2:
                x1, y1 = clicks[0]
                x2, y2 = clicks[1]

                # Calculate and store ROI data
                roi_data = {
                    "x": min(x1, x2),
                    "y": min(y1, y2),
                    "width": abs(x2 - x1),
                    "height": abs(y2 - y1)
                }

                # Put the ROI data in the queue for the main thread
                gui_queue.put(roi_data)

                # Close the window after recording ROI data
                root.destroy()

        # Create a Tkinter window and canvas
        root = tk.Tk()
        canvas = tk.Canvas(root, width=image.width, height=image.height)
        canvas.pack()

        # Display the image on the canvas
        canvas.image = ImageTk.PhotoImage(image)
        canvas.create_image(0, 0, anchor="nw", image=canvas.image)

        # Bind left mouse click to the handle_click function
        canvas.bind("<Button-1>", handle_click)
        clicks = []  # List to store click coordinates

        root.mainloop()  # Start the Tkinter event loop

        # Get the ROI data from the queue
        roi_data = gui_queue.get()

        # Your Sam model and segmentation code here
        sam_checkpoint = "sam_vit_h_4b8939.pth"  # the model needs to be downloaded from Meta AI Segment Anything website before the use
        model_type = "vit_h"
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam = sam.to('cuda')  # Move the model to the GPU
        #mask_generator = SamAutomaticMaskGenerator(sam)
        print("ROI data:", roi_data)
        box = np.array([roi_data['x'], roi_data['y'],
                        roi_data['x'] + roi_data['width'],
                        roi_data['y'] + roi_data['height']])
        #cpu_tensor = torch.from_numpy(box)

        #box_tensor = cpu_tensor.to(dtype=torch.float32, device='cuda')


        # Mask creation
        mask_predictor = SamPredictor(sam)
        mask_predictor.set_image(image_bgr)

        masks, scores, logits = mask_predictor.predict(box=box, multimask_output=True)

        # Segmentation using Draw_box
        box_annotator = sv.BoxAnnotator(color=sv.Color.RED)
        mask_annotator = sv.MaskAnnotator(color=sv.Color.RED, color_lookup=sv.ColorLookup.INDEX)

        detections = sv.Detections(xyxy=sv.mask_to_xyxy(masks=masks), mask=masks)
        detections = detections[detections.area == np.max(detections.area)]

        source_image = box_annotator.annotate(scene=image_bgr.copy(), detections=detections, skip_label=True)
        segmented_image = mask_annotator.annotate(scene=image_bgr.copy(), detections=detections)

        #Image showcase
        # Ensure both images have the same height
        min_height = min(source_image.shape[0], segmented_image.shape[0])
        source_image = source_image[:min_height, :]
        segmented_image = segmented_image[:min_height, :]

        # Create a white line to separate the images
        separator_line = np.ones((min_height, 5, 3), dtype=np.uint8) * 255

        # Concatenate images with the separator line
        result_image = np.concatenate((source_image, separator_line, segmented_image), axis=1)
        
        #sv.plot_images_grid(images=[source_image, segmented_image], grid_size=(1, 2),titles=['source image', 'segmented image']).savefig(img_bytes, format='png')
        _, img_bytes = cv2.imencode('.png', result_image)
        img_bytes = img_bytes.tobytes()
        

        # Store the image data as a base64-encoded string
        image_data = base64.b64encode(img_bytes).decode('utf-8')


        return render_template('upload.html', image_data=image_data)

    return render_template('upload.html')

if __name__ == '__main__':
    app.run(debug=True)
