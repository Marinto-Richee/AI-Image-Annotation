
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog, messagebox, simpledialog
import cv2
from PIL import Image, ImageTk
import os
import random
from autodistill_grounding_dino import GroundingDINO
from autodistill.detection import CaptionOntology

class ImageAnnotator:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Image Annotator")

        # Initialize attributes
        self.images = []
        self.current_image_index = None
        self.current_image = None
        self.bbox_start = None
        self.bbox_list = {}
        self.selected_class = None
        self.classes = {}  # Define the classes attribute here

        # Create a frame for the files section
        self.files_frame = tk.Frame(root, bg="white")
        self.files_frame.pack(side=tk.LEFT, fill=tk.BOTH,
                              expand=True, padx=10, pady=10)

        # Create a label for the files section
        self.files_label = tk.Label(
            self.files_frame, text="Image Files", font=("Helvetica", 12), bg="white")
        self.files_label.pack(side=tk.TOP, pady=(0, 5))

        # Create a listbox to display the image files
        self.files_listbox = tk.Listbox(
            self.files_frame, width=40, height=20, bg="#f0f0f0")
        self.files_listbox.pack(
            side=tk.TOP, fill=tk.BOTH, padx=(0, 5), pady=(0, 5))
        self.files_listbox.bind('<<ListboxSelect>>', self.load_image)

        # Create buttons for loading and deleting image files
        self.load_button = tk.Button(
            self.files_frame, text="Load Images", command=self.load_images, relief=tk.FLAT, bg="#007bff", fg="white", borderwidth=0,)
        self.load_button.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        self.delete_button = tk.Button(
            self.files_frame, text="Delete Image", command=self.delete_image, relief=tk.FLAT, bg="#007bff", fg="white", borderwidth=0,)
        self.delete_button.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        # Button to annotate all the images at once using the AI model
        self.ai_annotate_all_button = tk.Button(
            self.files_frame, text="AI Annotate All", command=self.annotate_all_with_model, relief=tk.FLAT, bg="#007bff", fg="white", borderwidth=0,)
        self.ai_annotate_all_button.pack(
            side=tk.TOP, fill=tk.X, padx=5, pady=5)

        # Create a frame for the classes section
        self.class_frame = tk.Frame(root, bg="white")
        self.class_frame.pack(side=tk.RIGHT, fill=tk.BOTH,
                              expand=True, padx=10, pady=10)

        # Create a label for the classes section
        self.class_label = tk.Label(
            self.class_frame, text="Annotation Classes", font=("Helvetica", 12), bg="white")
        self.class_label.pack(side=tk.TOP, pady=(0, 5))

        # Create a listbox to display the available classes
        self.class_listbox = tk.Listbox(
            self.class_frame, width=20, height=10, bg="#f0f0f0")
        self.class_listbox.pack(
            side=tk.TOP, fill=tk.BOTH, padx=(0, 5), pady=(0, 5))
        self.class_listbox.bind('<<ListboxSelect>>', self.select_class)

        # Add colored rectangles behind the class labels
        for class_name, color in self.classes.items():
            self.class_listbox.insert(tk.END, class_name)
            self.class_listbox.itemconfig(tk.END, {'bg': color})

        # Create buttons for adding and deleting classes
        self.add_class_button = tk.Button(
            self.class_frame, text="Add Class", command=self.add_class, relief=tk.FLAT, bg="#007bff", fg="white", borderwidth=0,)
        self.add_class_button.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        self.delete_class_button = tk.Button(
            self.class_frame, text="Delete Class", command=self.delete_class, relief=tk.FLAT, bg="#007bff", fg="white", borderwidth=0)
        self.delete_class_button.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        # Create a button to trigger AI-assisted annotation
        self.ai_annotate_button = tk.Button(
            root, text="AI Annotate", command=self.annotate_with_model, relief=tk.FLAT, bg="#007bff", fg="white", borderwidth=0,)
        self.ai_annotate_button.pack(side=tk.BOTTOM, pady=10)

        # Create a button to clear annotations for the current image
        self.clear_button = tk.Button(
            root, text="Clear Annotations", command=self.clear_annotations_for_image, relief=tk.FLAT, bg="#007bff", fg="white", borderwidth=0,)
        self.clear_button.pack(side=tk.BOTTOM, pady=10)

        # Create a button to save annotations
        self.save_button = tk.Button(
            root, text="Save Annotations", command=self.save_annotations, relief=tk.FLAT, bg="#007bff", fg="white", borderwidth=0,)
        self.save_button.pack(side=tk.BOTTOM, pady=10)

        # Initialize the current image index
        self.current_image_index = None

        # Create buttons for navigating between images
        self.prev_button = tk.Button(
            self.files_frame, text="Previous", command=self.prev_image, relief=tk.FLAT, bg="#007bff", fg="white", borderwidth=0,)
        self.prev_button.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        self.next_button = tk.Button(
            self.files_frame, text="Next", command=self.next_image, relief=tk.FLAT, bg="#007bff", fg="white", borderwidth=0,)
        self.next_button.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        # Create a canvas to display the image and annotations
        self.canvas = tk.Canvas(
            self.root, bg="white", highlightbackground="gray", highlightthickness=1)
        self.canvas.pack(side=tk.BOTTOM, fill=tk.BOTH,
                         padx=10, pady=10, expand=True)
        self.canvas.bind("<Button-1>", self.start_bbox)
        self.canvas.bind("<B1-Motion>", self.draw_bbox)
        self.canvas.bind("<ButtonRelease-1>", self.end_bbox)
        self.annotations_dict = {}
        # Create a canvas for the zoomed view
        self.zoom_canvas = tk.Canvas(
            self.class_frame, bg="white", highlightbackground="gray", highlightthickness=1)
        self.zoom_canvas.pack(side=tk.BOTTOM, padx=10, pady=10)

        # Initialize variables for zoom functionality
        self.zoom_img_id = None
        self.zoom_img = None

        # Bind mouse motion event to update zoomed view
        self.canvas.bind("<Motion>", self.update_zoom_view)
        self.canvas.bind("<Leave>", self.clear_zoom_view)

        # label for the zoom canvas
        self.zoom_label = tk.Label(
            self.class_frame, text="Zoom View", font=("Helvetica", 12), bg="white")
        self.zoom_label.pack(side=tk.BOTTOM, pady=(0, 5))

        # create a slider for confidence threshold
        self.confidence_threshold_label = tk.Label(
            self.class_frame, text="Confidence Threshold", font=("Helvetica", 12), bg="white")
        self.confidence_threshold_label.pack(side=tk.TOP, pady=(0, 5))
        self.confidence_threshold_slider = tk.Scale(
            self.class_frame, from_=0, to=100, orient=tk.HORIZONTAL, bg="white")
        self.confidence_threshold_slider.pack(side=tk.BOTTOM, pady=(0, 5))

        # Ask the user to load images
        self.load_images()
        # Ask the user to update the classes
        self.add_class()

    def update_zoom_view(self, event):
        try:
            if self.zoom_img_id:
                self.zoom_canvas.delete(self.zoom_img_id)
            if self.current_image is not None:
                # Copy the current image
                self.zoom_img = self.current_image.copy()
                # Resize the image to fit the canvas
                img_height, img_width, _ = self.zoom_img.shape
                canvas_width = self.canvas.winfo_width()
                canvas_height = self.canvas.winfo_height()
                if canvas_width / img_width < canvas_height / img_height:
                    resize_factor = canvas_width / img_width
                else:
                    resize_factor = canvas_height / img_height
                self.zoom_img = cv2.resize(
                    self.zoom_img, (int(img_width * resize_factor), int(img_height * resize_factor)))

                # Crop a small region around the cursor position
                zoom_x0 = max(0, event.x - 20)
                zoom_y0 = max(0, event.y - 20)
                zoom_x1 = min(self.canvas.winfo_width(), event.x + 20)
                zoom_y1 = min(self.canvas.winfo_height(), event.y + 20)

                zoom_img_crop = self.zoom_img[zoom_y0:zoom_y1,
                                              zoom_x0:zoom_x1, :]
                # CHECK IF THE CURSOR POSITION IS WITHIN THE IMAGE BOUNDARIES
                if zoom_img_crop.size == 0:
                    return
                # Resize the cropped region to fit the zoom canvas
                zoom_img_crop_resized = cv2.resize(
                    zoom_img_crop, (self.zoom_canvas.winfo_width(), self.zoom_canvas.winfo_height()))
                # Convert the cropped image to PhotoImage format
                zoom_img_crop_resized = Image.fromarray(
                    cv2.cvtColor(zoom_img_crop_resized, cv2.COLOR_BGR2RGB))
                zoom_img_crop_resized = ImageTk.PhotoImage(
                    zoom_img_crop_resized)
                # Display the cropped image on the zoom canvas
                self.zoom_img_id = self.zoom_canvas.create_image(
                    0, 0, image=zoom_img_crop_resized, anchor=tk.NW)
                # Keep a reference to the image to prevent it from being garbage collected
                self.zoom_canvas.image = zoom_img_crop_resized
        except Exception as e:
            print(f"Error updating zoom view: {e}")

    def clear_zoom_view(self, event):
        # Clear the zoom canvas when the cursor leaves the image boundaries
        self.zoom_canvas.delete("all")

    def next_image(self):
        if self.current_image_index is not None and self.current_image_index < len(self.images) - 1:
            self.current_image_index += 1
            self.load_selected_image()

    def prev_image(self):
        if self.current_image_index is not None and self.current_image_index > 0:
            self.current_image_index -= 1
            self.load_selected_image()

    def load_selected_image(self):
        self.save_annotations_temp()
        self.current_image = cv2.imread(self.images[self.current_image_index])
        self.display_image()
        image_path = self.images[self.current_image_index]
        if image_path in self.annotations_dict:
            for bbox, cls in self.annotations_dict[image_path]:
                self.canvas.create_rectangle(
                    *bbox, outline=self.classes[cls], tags="bbox")

    def load_image(self, event):
        selected_index = self.files_listbox.curselection()
        if selected_index:
            # Save annotations before loading new image
            self.save_annotations_temp()

            self.current_image_index = int(
                selected_index[0])  # Convert to integer
            self.current_image = cv2.imread(
                self.images[self.current_image_index])
            self.display_image()

            # Display annotations if available for the loaded image
            image_path = self.images[self.current_image_index]
            if image_path in self.annotations_dict:
                for bbox, cls in self.annotations_dict[image_path]:
                    self.canvas.create_rectangle(
                        *bbox, outline=self.classes[cls], tags="bbox")

    def add_class(self):
        new_class = simpledialog.askstring(
            "Add Class", "Object to detect:")
        if new_class:
            # lower case the class name and check if it already exists with all the lower case in self.classes
            if new_class.lower() in [x.lower() for x in self.classes.keys()]:
                messagebox.showwarning(
                    "Duplicate Class", "This class already exists.")

            else:
                color = '#' + "%06x" % random.randint(0, 0xFFFFFF)
                self.class_listbox.insert(tk.END, new_class)
                self.class_listbox.itemconfig(
                    tk.END, {'bg': color})  # Set background color
                self.classes[new_class] = color
                messagebox.showinfo(
                    "Class Added", "New class added successfully.")

    def delete_class(self):
        selected_index = self.class_listbox.curselection()
        if selected_index:
            selected_class = self.class_listbox.get(selected_index[0])
            del self.classes[selected_class]
            self.class_listbox.delete(selected_index[0])
            messagebox.showinfo("Class Deleted", "Class deleted successfully.")
            # Ask if they want to delete the annotations for the class for all the images
            if messagebox.askyesno("Delete Annotations", "Do you want to delete the annotations for this class for all the images?"):
                # Remove the annotations
                for image_path, annotations in self.annotations_dict.items():
                    for i, (bbox, cls) in enumerate(annotations):
                        if cls == selected_class:
                            annotations.pop(i)
                    self.annotations_dict[image_path] = annotations

    def select_class(self, event):
        selected_index = self.class_listbox.curselection()
        if selected_index:
            self.selected_class = self.class_listbox.get(selected_index[0])

    def start_bbox(self, event):
        if self.selected_class and self.current_image is not None:
            self.bbox_start = (event.x, event.y)
            self.draw_bbox(event)

    def draw_bbox(self, event):
        if self.bbox_start is not None:
            x0, y0 = self.bbox_start
            x1, y1 = (event.x, event.y)
            # Delete previous bounding boxes and redraw existing ones
            self.canvas.delete("bbox")
            if self.current_image_index is not None:
                image_path = self.images[self.current_image_index]
                if image_path in self.annotations_dict:
                    for bbox, cls in self.annotations_dict[image_path]:
                        self.canvas.create_rectangle(
                            *bbox, outline=self.classes[cls], tags="bbox")
            # Draw the current bounding box
            self.canvas.create_rectangle(
                x0, y0, x1, y1, outline=self.classes[self.selected_class], tags="bbox")

    def end_bbox(self, event):
        if self.bbox_start is not None:
            x0, y0 = self.bbox_start
            x1, y1 = (event.x, event.y)
            # Save the bounding box coordinates and class label
            bbox = ((x0, y0, x1, y1), self.selected_class)
            image_path = self.images[self.current_image_index]
            if image_path not in self.bbox_list:
                self.bbox_list[image_path] = []
            self.bbox_list[image_path].append(bbox)
            self.bbox_start = None
        self.save_annotations_temp()  # Save the annotations
        self.load_selected_image()  # Reload the image with the new annotations

    def display_image(self):
        # Convert image from OpenCV BGR format to RGB format
        img_rgb = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB)
        # Resize image to fit into the canvas while maintaining aspect ratio
        img_height, img_width, _ = img_rgb.shape
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        if canvas_width / img_width < canvas_height / img_height:
            resize_factor = canvas_width / img_width
        else:
            resize_factor = canvas_height / img_height
        resized_img = cv2.resize(
            img_rgb, (int(img_width * resize_factor), int(img_height * resize_factor)))
        # Convert resized image to ImageTk format
        img_tk = ImageTk.PhotoImage(Image.fromarray(resized_img))
        # Display image on canvas
        self.canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
        # Keep a reference to the image to prevent it from being garbage collected
        self.canvas.image = img_tk

    def load_images(self):
        try:
            file_paths = filedialog.askopenfilenames(
                filetypes=[("Image files", "*.jpg; *.jpeg; *.png")])
            if file_paths:
                self.images = list(file_paths)
                self.files_listbox.delete(0, tk.END)  # Clear previous entries
                for image_path in self.images:
                    self.files_listbox.insert(
                        tk.END, os.path.basename(image_path))
                messagebox.showinfo(
                    "Images Loaded", "Image files loaded successfully.")
                self.clear_annotations()  # Clear annotations when loading new images
        except Exception as e:
            messagebox.showerror(
                "Error", f"An error occurred while loading images: {str(e)}")

    def delete_image(self):
        try:
            selected_index = self.files_listbox.curselection()
            if selected_index:
                # Save annotations before clearing
                self.save_annotations_temp()

                del self.images[selected_index[0]]
                self.files_listbox.delete(selected_index[0])
                messagebox.showinfo(
                    "Image Deleted", "Image file deleted successfully.")
        except Exception as e:
            messagebox.showerror(
                "Error", f"An error occurred while deleting image: {str(e)}")

    def save_annotations(self):
        # check if there are any annotations in the annotations dictionary
        if not self.annotations_dict:
            messagebox.showwarning(
                "No Annotations", "There are no annotations to save.")
            return

        try:
            # Save annotations before saving
            self.save_annotations_temp()
            # Create a copy 
            annotations_dict_copy = self.annotations_dict.copy()

            # resize the bounding box coordinates to the original image size
            for image_path, annotations in self.annotations_dict.items():
                img = cv2.imread(image_path)
                img_height, img_width, _ = img.shape
                canvas_width = self.canvas.winfo_width()
                canvas_height = self.canvas.winfo_height()
                if canvas_width / img_width < canvas_height / img_height:
                    resize_factor = canvas_width / img_width
                else:
                    resize_factor = canvas_height / img_height
                for i, (bbox, cls) in enumerate(annotations):
                    x_min, y_min, x_max, y_max = bbox
                    x_min /= resize_factor
                    y_min /= resize_factor
                    x_max /= resize_factor
                    y_max /= resize_factor
                    annotations[i] = ((x_min, y_min, x_max, y_max), cls)

            # Save the annotations in YOLO format
            # Create a base folder to save the annotations
            base_folder = filedialog.askdirectory()
            if base_folder:
                # Create data.yaml file with class indices
                with open(os.path.join(base_folder, "data.yaml"), "w") as file:
                    file.write("train: ../train/images\n")
                    file.write("test: ../test/images\n")
                    file.write("val: ../val/images\n")
                    file.write("nc: " + str(len(self.classes)) + "\n")
                    file.write("names: " + str(list(self.classes.keys())) + "\n")
                # Create folders for train, test and validation sets
                train_folder = os.path.join(base_folder, "train")
                val_folder = os.path.join(base_folder, "val")
                test_folder = os.path.join(base_folder, "test")
                os.makedirs(train_folder, exist_ok=True)
                os.makedirs(val_folder, exist_ok=True)
                os.makedirs(test_folder, exist_ok=True)
                # Split the images into train, test and validation sets
                random.shuffle(self.images)
                train_images = self.images[:int(0.7 * len(self.images))]
                val_images = self.images[int(0.7 * len(self.images)):int(
                    0.85 * len(self.images))]
                test_images = self.images[int(0.85 * len(self.images)):]
                # Create images and labels folders for train, test and validation sets
                train_images_folder = os.path.join(train_folder, "images")
                val_images_folder = os.path.join(val_folder, "images")
                test_images_folder = os.path.join(test_folder, "images")
                train_labels_folder = os.path.join(train_folder, "labels")
                val_labels_folder = os.path.join(val_folder, "labels")
                test_labels_folder = os.path.join(test_folder, "labels")
                os.makedirs(train_images_folder, exist_ok=True)
                os.makedirs(val_images_folder, exist_ok=True)
                os.makedirs(test_images_folder, exist_ok=True)
                os.makedirs(train_labels_folder, exist_ok=True)
                os.makedirs(val_labels_folder, exist_ok=True)
                os.makedirs(test_labels_folder, exist_ok=True)
                # Save the annotations in YOLO format
                for image_path, annotations in self.annotations_dict.items():
                    img = cv2.imread(image_path)
                    img_height, img_width, _ = img.shape
                    # Get image name without extension (.jpg,.png, etc.)
                    img_name = os.path.basename(image_path).split(".")[0]
                    # Create a text file for the annotations
                    if image_path in train_images:
                        txt_file_path = os.path.join(
                            train_labels_folder, img_name+".txt")
                    elif image_path in val_images:
                        txt_file_path = os.path.join(
                            val_labels_folder, img_name+".txt")
                    else:
                        txt_file_path = os.path.join(
                            test_labels_folder, img_name+".txt")
                        
                    with open(txt_file_path, "w") as file:
                        for bbox, cls in annotations:
                            x_min, y_min, x_max, y_max = bbox
                            x_center = (x_min + x_max) / 2
                            y_center = (y_min + y_max) / 2
                            width = x_max - x_min
                            height = y_max - y_min
                            x_center /= img_width
                            y_center /= img_height
                            width /= img_width
                            height /= img_height
                            file.write(str(list(self.classes.keys()).index(cls)) +
                                       " " + str(x_center) + " " + str(y_center) + " " + str(width) + " " + str(height) + "\n")
                    # Copy the image to the train images folder
                    if image_path in train_images:
                        img_folder = train_images_folder
                    elif image_path in val_images:
                        img_folder = val_images_folder
                    else:
                        img_folder = test_images_folder
                    img_file_path = os.path.join(
                        img_folder, os.path.basename(image_path))
                    # Resize the image to 650x650
                    img = cv2.resize(img, (650, 650))
                    cv2.imwrite(img_file_path, img)
                messagebox.showinfo(
                    "Annotations Saved", "Annotations saved successfully.")
                # Replace the annotations dictionary with the copy
                self.annotations_dict = annotations_dict_copy

        except Exception as e:
            messagebox.showerror(
                "Error", f"An error occurred while saving annotations: {str(e)}")

                

    def save_annotations_temp(self):
        """
        Save annotations temporarily before clearing the annotations dictionary.
        """
        if self.current_image_index is not None:
            image_path = self.images[self.current_image_index]
            if image_path not in self.annotations_dict:
                self.annotations_dict[image_path] = self.bbox_list.get(
                    image_path, [])
            else:
                self.annotations_dict[image_path].extend(
                    self.bbox_list.get(image_path, []))
            # Clear annotations for the current image
            self.bbox_list.pop(image_path, None)

    def clear_annotations(self):
        self.bbox_list = {}  # Clear the bounding box list
        self.canvas.delete("bbox")  # Clear annotations displayed on the canvas

    def clear_annotations_for_image(self):
        if self.current_image_index is not None:
            # Clear annotations for the current image
            image_path = self.images[self.current_image_index]
            self.bbox_list.pop(image_path, None)
            self.canvas.delete("bbox")
            # Clear the annotations dictionary for the current image
            self.annotations_dict.pop(image_path, None)

            # Optionally, you can also reset the selected class
            self.selected_class = None

            # Show a message informing the user that annotations are cleared
            messagebox.showinfo(
                "Annotations Cleared", "Annotations for the current image cleared. You can now redraw annotations.")

            # You may also want to reset any other relevant attributes or UI elements
            # For example, if you want to allow users to select a new class for annotations:
            self.class_listbox.selection_clear(0, tk.END)
            self.selected_class = None

    def annotate_with_model(self):
        ontology_dict = self.get_classes_from_user()
        # Check if the user has entered any classes
        if not ontology_dict:
            messagebox.showwarning(
                "No Classes", "Please enter the classes to annotate.")
            return

        # Check if there are any images to annotate
        if not self.images:
            messagebox.showwarning(
                "No Images", "Please load some images to annotate.")
            return

        # Check if an image is loaded
        if self.current_image is None:
            messagebox.showwarning(
                "No Image Selected", "Please select an image to annotate.")
            return

        if self.current_image is not None:
            # Create a progress bar
            progress_bar = ttk.Progressbar(
                self.root, orient='horizontal', mode='determinate')
            progress_bar.pack(side=tk.BOTTOM, fill=tk.X, padx=10)
            progress_bar['value'] = 20
            progress_bar.update_idletasks()
            # Get the classes entered by the user

            ontology = CaptionOntology(ontology_dict)
            base_model = GroundingDINO(ontology=ontology)

            result = base_model.predict(self.current_image)
            progress_bar['value'] = 50
            progress_bar.update_idletasks()
            for i, (bbox, cls, conf) in enumerate(zip(result.xyxy, result.class_id, result.confidence), start=1):
                if conf*100 < self.confidence_threshold_slider.get():
                    continue

                # Resize the bounding box coordinates to the canvas size
                img_height, img_width, _ = self.current_image.shape
                canvas_width = self.canvas.winfo_width()
                canvas_height = self.canvas.winfo_height()
                if canvas_width / img_width < canvas_height / img_height:
                    resize_factor = canvas_width / img_width
                else:
                    resize_factor = canvas_height / img_height
                bbox = (bbox[0] * resize_factor, bbox[1] * resize_factor,
                        bbox[2] * resize_factor, bbox[3] * resize_factor)

                # save the bounding box coordinates and class label
                bbox = ((bbox[0], bbox[1], bbox[2], bbox[3]),
                        list(ontology_dict.keys())[cls])
                image_path = self.images[self.current_image_index]
                if image_path not in self.bbox_list:
                    self.bbox_list[image_path] = []
                self.bbox_list[image_path].append(bbox)
            progress_bar['value'] = 100
            progress_bar.update_idletasks()

            # Hide the progress bar when prediction is complete
            progress_bar.pack_forget()

        self.save_annotations_temp()  # Save the annotations
        self.load_selected_image()  # Reload the image with the new annotations

    def get_classes_from_user(self):
        classes = {}
        for i in range(self.class_listbox.size()):
            class_name = self.class_listbox.get(i)
            classes[class_name] = class_name
        return classes

    def on_closing(self):
        # check if there are any annotations in the annotations dictionary"
        if self.annotations_dict:
            # check if the user wants to save annotations before closing
            if messagebox.askokcancel("Quit", "Do you want to save annotations before quitting?"):
                self.save_annotations()
        self.root.destroy()

    # Method to annotate all the images at once using the AI model
    def annotate_all_with_model(self):
        ontology_dict = self.get_classes_from_user()
        # Check if the user has entered any classes
        if not ontology_dict:
            messagebox.showwarning(
                "No Classes", "Please enter the classes to annotate.")
            return

            # Check if there are any images to annotate
        if not self.images:
            messagebox.showwarning(
                "No Images", "Please load some images to annotate.")
            return

        # Create a progress bar
        progress_bar = ttk.Progressbar(
            self.root, orient='horizontal', mode='determinate')
        progress_bar.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=5)
        progress_bar['value'] = 20
        progress_bar.update_idletasks()
        ontology = CaptionOntology(ontology_dict)
        base_model = GroundingDINO(ontology=ontology)
        for i, image_path in enumerate(self.images, start=1):
            img_rgb = cv2.imread(image_path)
            result = base_model.predict(img_rgb)
            for bbox, cls, conf in zip(result.xyxy, result.class_id, result.confidence):
                # Check if the confidence is above the threshold
                if conf*100 < self.confidence_threshold_slider.get():
                    continue
                # Resize the bounding box coordinates to the canvas size
                img_height, img_width, _ = img_rgb.shape
                canvas_width = self.canvas.winfo_width()
                canvas_height = self.canvas.winfo_height()
                if canvas_width / img_width < canvas_height / img_height:
                    resize_factor = canvas_width / img_width
                else:
                    resize_factor = canvas_height / img_height
                bbox = (bbox[0] * resize_factor, bbox[1] * resize_factor,
                        bbox[2] * resize_factor, bbox[3] * resize_factor)

                # save the bounding box coordinates and class label
                bbox = ((bbox[0], bbox[1], bbox[2], bbox[3]),
                        list(ontology_dict.keys())[cls])
                if image_path not in self.bbox_list:
                    self.bbox_list[image_path] = []
                self.bbox_list[image_path].append(bbox)
            progress_bar['value'] = 20 + (i/len(self.images))*80
            progress_bar.update_idletasks()
        progress_bar['value'] = 100
        progress_bar.update_idletasks()
        # Hide the progress bar when prediction is complete
        progress_bar.pack_forget()
        self.save_annotations_temp()  # Save the annotations


if __name__ == "__main__":
    root = tk.Tk()
    root.resizable(False, False)
    app = ImageAnnotator(root)
    # Bind on_closing method to close window event
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()