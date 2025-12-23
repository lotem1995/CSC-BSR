import os
import pandas as pd
import re


def pair_images_with_fens(csv_path, images_folder):
    """
    Pairs images in a folder with FEN strings from a CSV based on timestamp/frame number.

    Args:
        csv_path (str): Path to the game2.csv file.
        images_folder (str): Directory containing the images.

    Returns:
        list of tuples: [(image_path, fen_string), ...]
    """
    # 1. Load the CSV
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: CSV file '{csv_path}' not found.")
        return []

    # Ensure columns exist
    required_cols = {'from_frame', 'to_frame', 'fen'}
    if not required_cols.issubset(df.columns):
        print(f"Error: CSV is missing columns. Found: {df.columns}")
        return []

    results = []

    # 2. List all files in the image folder
    try:
        image_files = [f for f in os.listdir(images_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    except FileNotFoundError:
        print(f"Error: Image folder '{images_folder}' not found.")
        return []

    print(f"Found {len(image_files)} images. Matching with CSV...")

    # 3. Iterate through images and find the matching FEN
    matched_count = 0

    for img_file in image_files:
        # Extract the numeric timestamp/frame from the filename
        # This regex looks for the first sequence of digits in the filename
        match = re.search(r'(\d+)', img_file)

        if match:
            frame_number = int(match.group(1))

            # Find the row where the frame number falls within the range [from_frame, to_frame]
            # We use a mask to filter the DataFrame
            matching_row = df[
                (df['from_frame'] <= frame_number) &
                (df['to_frame'] >= frame_number)
                ]

            if not matching_row.empty:
                # Get the FEN from the first matching row (usually there's only one)
                fen = matching_row.iloc[0]['fen']
                full_image_path = os.path.join(images_folder, img_file)

                results.append((full_image_path, fen))
                matched_count += 1
            else:
                # Optional: Print if no match found for this frame
                # print(f"No FEN found for frame {frame_number}")
                pass
        else:
            print(f"Warning: Could not find a number in filename '{img_file}'")

    print(f"Successfully matched {matched_count} images to FENs.")
    return results


