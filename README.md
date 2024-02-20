
# Image Annotator

Image Annotator is a simple tool built with Python and Tkinter for annotating images with bounding boxes. It allows users to load images, draw bounding boxes around objects of interest, assign class labels to the objects, and save the annotations in a CSV file format.

## Features

- Load image files (supports JPEG, JPG, and PNG formats).
- Draw bounding boxes around objects in the images.
- Assign class labels to the annotated objects.
- Add and delete annotation classes.
- Save annotations to a CSV file.
- Clear annotations for individual images.

## Getting Started

### Prerequisites

- Python 3.x
- Tkinter (should be included in standard Python installations)
- OpenCV (`pip install opencv-python`)
- Pillow (`pip install pillow`)

### Installation

1. Clone the repository:

```
git clone https://github.com/yourusername/image-annotator.git
```

2. Navigate to the project directory:

```
cd image-annotator
```

3. Install the required dependencies:

```
pip install -r requirements.txt
```

### Usage

1. Run the application:

```
python image_annotator.py
```

2. Load image files using the "Load Images" button.
3. Draw bounding boxes around objects by clicking and dragging on the image.
4. Assign class labels to the objects using the class listbox.
5. Save annotations to a CSV file using the "Save Annotations" button.
6. To clear annotations for a specific image, select the image from the list and click the "Clear Annotations" button.

