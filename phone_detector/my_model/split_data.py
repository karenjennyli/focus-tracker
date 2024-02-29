import os
import random
import shutil

def split_data_structure(input_dir, output_base_dir, train_ratio=0.85, valid_ratio=0.10, test_ratio=0.05):
    # Define directories
    images_dir = os.path.join(input_dir, "images")
    labels_dir = os.path.join(input_dir, "labels")

    # Create output directories
    train_images_dir = os.path.join(output_base_dir, "train/images")
    train_labels_dir = os.path.join(output_base_dir, "train/labels")
    valid_images_dir = os.path.join(output_base_dir, "valid/images")
    valid_labels_dir = os.path.join(output_base_dir, "valid/labels")
    test_images_dir = os.path.join(output_base_dir, "test/images")
    test_labels_dir = os.path.join(output_base_dir, "test/labels")

    os.makedirs(train_images_dir, exist_ok=True)
    os.makedirs(train_labels_dir, exist_ok=True)
    os.makedirs(valid_images_dir, exist_ok=True)
    os.makedirs(valid_labels_dir, exist_ok=True)
    os.makedirs(test_images_dir, exist_ok=True)
    os.makedirs(test_labels_dir, exist_ok=True)

    # Get list of all image files
    image_files = [f for f in os.listdir(images_dir) if f.endswith('.jpg') or f.endswith('.png')]
    # Assuming label files match image file names but with different extensions
    label_files = [f.replace('.jpg', '.txt').replace('.png', '.txt') for f in image_files]

    # Shuffle the image files and their corresponding label files in unison
    combined = list(zip(image_files, label_files))
    random.shuffle(combined)
    image_files[:], label_files[:] = zip(*combined)

    # Calculate split indices
    total_images = len(image_files)
    num_train = int(total_images * train_ratio)
    num_valid = int(total_images * valid_ratio)
    # The rest goes into test

    # Function to copy files
    def copy_files(files, src_dir, dst_dir):
        for f in files:
            src_path = os.path.join(src_dir, f)
            dst_path = os.path.join(dst_dir, f)
            shutil.copy(src_path, dst_path)

    # Split and copy image and label files
    copy_files(image_files[:num_train], images_dir, train_images_dir)
    copy_files(label_files[:num_train], labels_dir, train_labels_dir)

    copy_files(image_files[num_train:num_train+num_valid], images_dir, valid_images_dir)
    copy_files(label_files[num_train:num_train+num_valid], labels_dir, valid_labels_dir)

    copy_files(image_files[num_train+num_valid:], images_dir, test_images_dir)
    copy_files(label_files[num_train+num_valid:], labels_dir, test_labels_dir)

# Example usage
input_dir = "data/all_positive"
output_base_dir = "data"
split_data_structure(input_dir, output_base_dir)
