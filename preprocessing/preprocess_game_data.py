from tqdm import tqdm

from preprocessing.handle_fen import fen_to_board_int
from preprocessing.handle_game_CSV import pair_images_with_fens
from preprocessing.splitting_images import slice_image_with_coordinates
import os

script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
parent_dir = os.path.abspath(os.path.join(script_dir, '..'))
data_dir = os.path.join(parent_dir, 'data')




output_dir = os.path.join(parent_dir, "preprocessed_data")
for game_id in [2,4,5,6,7]: # a bit those games number XD
    csv_file = os.path.join(data_dir, f'game{game_id}_per_frame', f"game{game_id}.csv")
    img_dir = os.path.join(data_dir, f'game{game_id}_per_frame', "tagged_images")
    print(f"data_path is: {csv_file}")

    # Get the list of pairs
    dataset_pairs = pair_images_with_fens(csv_file, img_dir)

    # Verify the first few matches
    for img_path, fen in tqdm(dataset_pairs):
        board = fen_to_board_int(fen)
        input_image = img_path

        slice_image_with_coordinates(
            game_id,
            input_image,
            output_dir,
            board,
            overlap_percent=0.7, # 0.7 for this parameter looks good by eye
            final_size=(224, 224),
            zero_padding=True
        )
