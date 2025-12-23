import os
from PIL import Image


def slice_image_with_coordinates(image_path, output_folder, board, overlap_percent=0.0, final_size=(224, 224),
                                 zero_padding=True):
    """
    Slices an image into an 8x8 grid with overlap, resizing inputs,
    and saving with Row/Col filenames.
    """
    try:
        img = Image.open(image_path)
    except FileNotFoundError:
        print(f"Error: The file '{image_path}' was not found.")
        return

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    img_width, img_height = img.size

    # Standard 8x8 grid stride
    cols = 8
    rows = 8
    stride_w = img_width / cols
    stride_h = img_height / rows

    # Crop size with overlap
    crop_w = stride_w * (1 + overlap_percent)
    crop_h = stride_h * (1 + overlap_percent)

    for r in range(rows):
        for c in range(cols):
            # 1. Find Center
            center_x = (c * stride_w) + (stride_w / 2)
            center_y = (r * stride_h) + (stride_h / 2)

            # 2. Calculate Box
            left = center_x - (crop_w / 2)
            upper = center_y - (crop_h / 2)
            right = center_x + (crop_w / 2)
            lower = center_y + (crop_h / 2)

            # 3. Handle Edges (Clamp)
            if not zero_padding:
                left = max(0, left)
                upper = max(0, upper)
                right = min(img_width, right)
                lower = min(img_height, lower)

            # 4. Crop
            tile = img.crop((left, upper, right, lower))

            # 5. Resize
            tile = tile.resize(final_size, Image.Resampling.LANCZOS)

            # 6. Save with Row/Col in filename

            # Get the filename first
            filename = os.path.basename(image_path)
            # Split the name and the extension, and take the first part
            name_only = os.path.splitext(filename)[0]
            tile_filename = f"{name_only}_tile_row{r}_column{c}_class{board[r, c]}.png"
            tile_path = os.path.join(output_folder, tile_filename)
            tile.save(tile_path)
