
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


## Screenshots 📸
![image](https://github.com/Marinto-Richee/Image-annotation/assets/65499285/2350a4c7-9ea3-4a66-9880-73035c375d5f)
![image](https://github.com/Marinto-Richee/Image-annotation/assets/65499285/938aa976-6765-42a5-a58a-ab99c3ba6665)

![image](https://github.com/Marinto-Richee/Image-annotation/assets/65499285/ddfb3221-5dfd-49f0-8e76-177874705761)
![image](https://github.com/Marinto-Richee/Image-annotation/assets/65499285/f6bd05f8-da3c-4f9e-b2f8-e32d03400061)


<iframe width="560" height="315" src="https://github.com/Marinto-Richee/AI-Image-Annotation/blob/d6815f54b709607a8d12ce3f840b5b7c97528aa9/AI%20annotation.mp4" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

## License 📝
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
