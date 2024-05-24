
# AI Image Annotator 🖼️

The AI Image Annotator is a Python application built with tkinter for annotating images with bounding boxes representing objects detected in the images. It also provides functionality to annotate images using an AI model for object detection.

## Features ✨

- Load and display images for annotation.
- Draw bounding boxes around objects in images.
- Annotate images with custom classes and colors.
- Annotate images manually or with the assistance of an AI model.
- Save annotations in YOLO Format.

## Requirements 🛠️
- autodistill
- autodistill_grounding_dino
- matplotlib
- Pillow
- roboflow

## Usage 🚀

1. Clone the repository.
2. Install the required packages using the following command:
```bash
pip install -r requirements.txt
```
3. Run the application using the following command:
- For CSV output:
```bash 
python AI_annotation_csv.py
```
- For YOLOV8 output:
```bash
python AI_annotation_yolov8.py
```
4. Load an image using the "Load Image" button.
5. Annotate the image by drawing bounding boxes around objects.
6. Save the annotations in YOLO format using the "Save Annotations" button.


## License 📝
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
