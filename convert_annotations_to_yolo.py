import os
from PIL import Image

# Define the relative paths
# annotation_folder = './OPIXray/train/train_annotation/'
# image_folder = './OPIXray/train/train_image/'
# output_folder = './OPIXray/train/train_labels_yolo/'

# # comment below if you want test data in yolo format
annotation_folder = './OPIXray/test/test_annotation/'
image_folder = './OPIXray/test/test_image/'
output_folder = './OPIXray/test/test_labels_yolo/'

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Define the class map (update this to match your dataset classes)
class_map = {
    "Straight_Knife": 0,
    "Scissor": 1,
    "Utility_Knife": 2,
    "Folding_Knife": 3,
    "Multi-tool_Knife": 4
}

# Function to convert annotation to YOLO format
def convert_to_yolo_format(filename, image_width, image_height):
    with open(filename, 'r') as file:
        lines = file.readlines()
    
    yolo_annotations = []
    for line in lines:
        parts = line.strip().split()
        
        # Example: 009000.jpg Straight_Knife 763 486 840 549
        image_name, label, xmin, ymin, xmax, ymax = parts
        
        # Convert to YOLO format
        if label not in class_map:
            continue  # Skip if the label is not in the class_map
        class_id = class_map[label]
        
        # Calculate center, width, and height in YOLO format
        width = float(xmax) - float(xmin)
        height = float(ymax) - float(ymin)
        x_center = (float(xmin) + width / 2) / float(image_width)
        y_center = (float(ymin) + height / 2) / float(image_height)
        width = width / float(image_width)
        height = height / float(image_height)
        
        yolo_annotations.append(f"{class_id} {x_center} {y_center} {width} {height}")
    
    return yolo_annotations

# Iterate over all images and annotations
for image_filename in os.listdir(image_folder):
    if image_filename.endswith('.jpg'):
        # Get the corresponding annotation file
        annotation_filename = os.path.join(annotation_folder, image_filename.replace('.jpg', '.txt'))
        output_filename = os.path.join(output_folder, image_filename.replace('.jpg', '.txt'))
        
        if os.path.exists(annotation_filename):
            # Open the image to get its dimensions
            image_path = os.path.join(image_folder, image_filename)
            with Image.open(image_path) as img:
                image_width, image_height = img.size
            
            # Convert and save the YOLO format annotation
            yolo_annotations = convert_to_yolo_format(annotation_filename, image_width, image_height)
            with open(output_filename, 'w') as output_file:
                for annotation in yolo_annotations:
                    output_file.write(f"{annotation}\n")

print("Annotations successfully converted to YOLO format!")
