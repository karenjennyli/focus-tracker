import os
import csv
import shutil

def convert_bbox_to_yolo(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)

def convert_to_yolo(labels_csv_path, positive_images_dir, negative_images_dir, output_images_dir, output_labels_dir):
    # Ensure output directories exist
    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(output_labels_dir, exist_ok=True)

    # Process positive images and labels
    with open(labels_csv_path, 'r') as csvfile:
        csvreader = csv.DictReader(csvfile)
        for row in csvreader:
            filename = row['filename']
            width, height = int(row['width']), int(row['height'])
            xmin, ymin, xmax, ymax = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])

            # Convert bounding box to YOLO format
            bbox_yolo = convert_bbox_to_yolo((width, height), (xmin, xmax, ymin, ymax))

            # Write YOLO format label to .txt file
            label_filename = os.path.splitext(filename)[0] + '.txt'
            label_path = os.path.join(output_labels_dir, label_filename)
            with open(label_path, 'w') as label_file:
                # Assuming mobile phone class is 0
                label_file.write(f"0 {bbox_yolo[0]:.6f} {bbox_yolo[1]:.6f} {bbox_yolo[2]:.6f} {bbox_yolo[3]:.6f}\n")

            # Copy image to output directory
            image_path = os.path.join(positive_images_dir, filename)
            output_image_path = os.path.join(output_images_dir, filename)
            shutil.copy(image_path, output_image_path)

    # Process negative images
    for filename in os.listdir(negative_images_dir):
        if filename.endswith(".png") or filename.endswith(".jpg"):
            # Copy image to output directory
            image_path = os.path.join(negative_images_dir, filename)
            output_image_path = os.path.join(output_images_dir, filename)
            shutil.copy(image_path, output_image_path)

labels_csv_path = "muid_iitr/labels.csv"
positive_images_dir = "muid_iitr/positive"
negative_images_dir = "muid_iitr/negative"
output_images_dir = "images"
output_labels_dir = "labels"
convert_to_yolo(labels_csv_path, positive_images_dir, negative_images_dir, output_images_dir, output_labels_dir)
