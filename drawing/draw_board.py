import os
import requests
import torch
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# ==========================================
# 1. Automatic Asset Downloading System
# ==========================================
ASSET_DIR = "chess_assets"

def download_chess_assets():
    """
    Downloads standard chess piece images from Wikimedia Commons
    if they don't already exist locally.
    """
    if not os.path.exists(ASSET_DIR):
        os.makedirs(ASSET_DIR)
        print(f"Created directory: {ASSET_DIR}")

    # URLs for standard Wikimedia chess pieces (high quality PNGs)
    base_url = "https://upload.wikimedia.org/wikipedia/commons/thumb"

    # Mapping: Filename -> Wikimedia URL suffix
    piece_urls = {
        'bK': 'f/f0/Chess_kdt45.svg/120px-Chess_kdt45.svg.png',
        'bQ': '4/47/Chess_qdt45.svg/120px-Chess_qdt45.svg.png',
        'bR': 'f/ff/Chess_rdt45.svg/120px-Chess_rdt45.svg.png',
        'bB': '9/98/Chess_bdt45.svg/120px-Chess_bdt45.svg.png',
        'bN': 'e/ef/Chess_ndt45.svg/120px-Chess_ndt45.svg.png',
        'bP': 'c/c7/Chess_pdt45.svg/120px-Chess_pdt45.svg.png',
        'wK': '4/42/Chess_klt45.svg/120px-Chess_klt45.svg.png',
        'wQ': '1/15/Chess_qlt45.svg/120px-Chess_qlt45.svg.png',
        'wR': '7/72/Chess_rlt45.svg/120px-Chess_rlt45.svg.png',
        'wB': 'b/b1/Chess_blt45.svg/120px-Chess_blt45.svg.png',
        'wN': '7/70/Chess_nlt45.svg/120px-Chess_nlt45.svg.png',
        'wP': '4/45/Chess_plt45.svg/120px-Chess_plt45.svg.png',
    }

    print("Checking assets...")
    for filename, url_suffix in piece_urls.items():
        filepath = os.path.join(ASSET_DIR, f"{filename}.png")
        if not os.path.exists(filepath):
            full_url = f"{base_url}/{url_suffix}"
            try:
                r = requests.get(full_url, headers={'User-Agent': 'Mozilla/5.0'})
                with open(filepath, 'wb') as f:
                    f.write(r.content)
            except Exception as e:
                print(f"Error downloading {filename}: {e}")
    print("Assets ready.")

# ==========================================
# 2. Board Rendering Function
# ==========================================
def generate_ood_board(board_tensor, output_file="ood_board_result.png"):
    """
    Draws the board using Matplotlib and the downloaded assets.
    """
    # Ensure assets exist before starting
    download_chess_assets()

    # --- MAPPING BASED ON YOUR SCREENSHOT ---
    idx_to_name = {
        0: 'wP',  # White Pawn
        1: 'wR',  # White Rook
        2: 'wN',  # White Knight
        3: 'wB',  # White Bishop
        4: 'wQ',  # White Queen
        5: 'wK',  # White King
        6: 'bP',  # Black Pawn
        7: 'bR',  # Black Rook
        8: 'bN',  # Black Knight
        9: 'bB',  # Black Bishop
        10: 'bQ', # Black Queen
        11: 'bK'  # Black King
    }

    # Board Setup
    board = board_tensor.cpu().numpy()
    fig, ax = plt.subplots(figsize=(6, 6))

    # Colors (Standard nice wood colors)
    color_light = '#F0D9B5'
    color_dark = '#B58863'

    for row in range(8):
        for col in range(8):
            # 1. Draw Square
            is_light = (row + col) % 2 == 0
            square_color = color_light if is_light else color_dark

            # Matplotlib y-axis inversion (row 0 at top)
            y_pos = 7 - row

            rect = plt.Rectangle((col, y_pos), 1, 1, facecolor=square_color)
            ax.add_patch(rect)

            val = board[row, col]

            # 2. Draw Piece
            if val in idx_to_name:
                piece_name = idx_to_name[val]
                img_path = os.path.join(ASSET_DIR, f"{piece_name}.png")
                try:
                    img = mpimg.imread(img_path)
                    ax.imshow(img, extent=[col, col+1, y_pos, y_pos+1], zorder=10)
                except FileNotFoundError:
                    pass

            # 3. Draw OOD Marker (Red X)
            elif val == 13:
                # Draw thick red cross with padding to prevent merging
                padding = 0.15
                # Line 1: Top-Left to Bottom-Right
                ax.plot([col + padding, col + 1 - padding],
                        [y_pos + 1 - padding, y_pos + padding],
                        color='red', linewidth=4, zorder=20)
                # Line 2: Bottom-Left to Top-Right
                ax.plot([col + padding, col + 1 - padding],
                        [y_pos + padding, y_pos + 1 - padding],
                        color='red', linewidth=4, zorder=20)

    # Cleanup Plot
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 8)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('off')

    # Save
    plt.savefig(output_file, bbox_inches='tight', pad_inches=0, dpi=100)
    plt.close(fig)
    print(f"Success! Image saved to: {output_file}")


# ==========================================
# Main Execution
# ==========================================
if __name__ == "__main__":
    # Example Tensor updated to use the new mapping
    # 7=bR (Black Rook), 11=bK (Black King), 13=OOD, etc.
    example_board = torch.tensor([
        [7, 8, 9, 10, 11, 9, 8, 13],
        [6, 6, 6, 6, 6, 6, 6, 6],
        [12, 12, 12, 12, 12, 12, 12, 12],
        [13, 13, 12, 12, 12, 12, 12, 12],
        [13, 13, 12, 12, 12, 12, 12, 12],
        [12, 12, 12, 12, 12, 12, 12, 12],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [1, 2, 3, 4, 5, 3, 2, 1]

    ])

    generate_ood_board(example_board, output_file="board_drawing.png")