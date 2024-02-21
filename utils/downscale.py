import os
from PIL import Image

# Scaling factor
scale_factor = 4
# Paths
original_dir = "../dataset/DIV2K/DIV2K_train_HR"
output_dir = f"../dataset/DIV2K/DIV2K_train_LR_bicubic/X{scale_factor}"

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# List all image files in the original directory
image_files = [f for f in os.listdir(original_dir) if f.endswith((".jpg", ".png"))]

# Process each image
for image_file in image_files:
    # Load the original image
    image_path = os.path.join(original_dir, image_file)
    original_image = Image.open(image_path)

    # Calculate dimensions for LR image
    lr_width = original_image.width // scale_factor
    lr_height = original_image.height // scale_factor

    # Resize using bicubic interpolation for LR image
    lr_image = original_image.resize((lr_width, lr_height), Image.BICUBIC)

    # Save LR image with scale factor in the name
    lr_filename = f"{os.path.splitext(image_file)[0]}x{scale_factor}.{original_image.format.lower()}"
    lr_image.save(os.path.join(output_dir, lr_filename))