import os

def delete_negative_images(directory):
    for filename in os.listdir(directory):
        if "n" in filename.split(".")[0]:  # Check if "n" is in the filename (excluding extension)
            os.remove(os.path.join(directory, filename))

# Example usage
images_directory = "data/all_positive/images"
delete_negative_images(images_directory)
