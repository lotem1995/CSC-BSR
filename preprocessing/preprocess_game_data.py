from tqdm import tqdm

from preprocessing.handle_fen import fen_to_board_int
from preprocessing.handle_game_CSV import pair_images_with_fens
from preprocessing.splitting_images import slice_image_with_coordinates
import os

script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
parent_dir = os.path.abspath(os.path.join(script_dir, '..'))
data_dir = os.path.join(parent_dir, 'data')


csv_file = os.path.join(data_dir, 'game2_per_frame',"game2.csv")
img_dir = os.path.join(data_dir, 'game2_per_frame',"tagged_images")

print(f"data_path is: {csv_file}")

output_dir = "preprocessed_data"

# Get the list of pairs
dataset_pairs = pair_images_with_fens(csv_file, img_dir)


# Verify the first few matches
for img_path, fen in tqdm(dataset_pairs):
    board=fen_to_board_int(fen)
    input_image = img_path

    slice_image_with_coordinates(
        input_image,
        output_dir,
        board,
        overlap_percent=0.3,
        final_size=(224, 224)
    )

