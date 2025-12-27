"""
Test script for QwenVisionEmbedding class.
Loads one chessboard image, splits it into 64 tiles, and extracts embeddings.
"""

import os
import sys
import torch
import numpy as np
from PIL import Image
import tempfile

# Add preprocessing to path
sys.path.insert(0, '/home/lotems/Documents/DL_Oren/CSC-BSR/preprocessing')
sys.path.insert(0, '/home/lotems/Documents/DL_Oren/CSC-BSR/embadding')

from splitting_images import slice_image_with_coordinates
from main import QwenVisionEmbedding


def test_embedding_pipeline():
    """
    Test the complete pipeline: load image -> split into tiles -> extract embeddings
    """
    print("=" * 80)
    print("Testing QwenVisionEmbedding with Real Chessboard Image")
    print("=" * 80)
    
    # Step 1: Select one test image
    data_dir = "/home/lotems/Documents/DL_Oren/CSC-BSR/data/game2_per_frame/tagged_images"
    test_image = os.path.join(data_dir, "frame_000200.jpg")
    
    print(f"\n[Step 1] Loading test image: {test_image}")
    if not os.path.exists(test_image):
        print(f"ERROR: Test image not found at {test_image}")
        return
    
    # Display image info
    img = Image.open(test_image)
    print(f"  - Image size: {img.size}")
    print(f"  - Image mode: {img.mode}")
    
    # Step 2: Create temporary directory for tiles
    temp_dir = tempfile.mkdtemp(prefix="chess_tiles_")
    print(f"\n[Step 2] Splitting image into 64 tiles...")
    print(f"  - Output directory: {temp_dir}")
    
    # Create a dummy board (we'll use zeros as placeholder for piece classes)
    # In real usage, you'd have the FEN string to generate this
    dummy_board = np.zeros((8, 8), dtype=int)
    
    try:
        slice_image_with_coordinates(
            image_path=test_image,
            output_folder=temp_dir,
            board=dummy_board,
            overlap_percent=0.0,
            final_size=(224, 224),
            zero_padding=True
        )
        print("  ✓ Image splitting completed")
    except Exception as e:
        print(f"ERROR during image splitting: {e}")
        return
    
    # Step 3: Load all tile images
    tile_files = sorted([f for f in os.listdir(temp_dir) if f.endswith('.png')])
    print(f"\n[Step 3] Loaded {len(tile_files)} tile images")
    
    if len(tile_files) != 64:
        print(f"  WARNING: Expected 64 tiles, got {len(tile_files)}")
    
    # Load PIL images
    tile_images = []
    for tile_file in tile_files:
        tile_path = os.path.join(temp_dir, tile_file)
        tile_img = Image.open(tile_path).convert('RGB')
        tile_images.append(tile_img)
        if len(tile_images) <= 3:
            print(f"  - {tile_file}: {tile_img.size}")
    
    # Step 4: Initialize embedding model
    print(f"\n[Step 4] Initializing QwenVisionEmbedding model...")
    print("  (This may take a moment for first-time initialization)")
    
    try:
        embedder = QwenVisionEmbedding("Qwen/Qwen3-VL-2B-Instruct")
        print("  ✓ Model initialized successfully")
    except Exception as e:
        print(f"ERROR during model initialization: {e}")
        return
    
    # Step 5: Extract embeddings for all tiles
    print(f"\n[Step 5] Extracting embeddings for {len(tile_images)} tiles...")
    print("  (Using batch processing for efficiency)")
    
    try:
        batch_embeddings = embedder.extract_batch_embeddings(tile_images)
        print(f"  ✓ Embeddings extracted successfully")
        print(f"  - Embedding shape: {batch_embeddings.shape}")
        print(f"  - Embedding dtype: {batch_embeddings.dtype}")
    except Exception as e:
        print(f"ERROR during embedding extraction: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 6: Display statistics
    print(f"\n[Step 6] Embedding Statistics")
    print(f"  - Total embeddings: {batch_embeddings.shape[0]}")
    print(f"  - Embedding dimension: {batch_embeddings.shape[1]}")
    print(f"  - Mean embedding norm: {torch.norm(batch_embeddings, p=2, dim=1).mean():.4f}")
    print(f"  - Min embedding value: {batch_embeddings.min():.4f}")
    print(f"  - Max embedding value: {batch_embeddings.max():.4f}")
    print(f"  - Mean embedding value: {batch_embeddings.mean():.4f}")
    
    # Step 7: Similarity analysis (optional)
    print(f"\n[Step 7] Tile Similarity Analysis (first 4x4 region)")
    # Compute pairwise similarities for a small region
    first_16 = batch_embeddings[:16]
    similarities = torch.nn.functional.cosine_similarity(
        first_16.unsqueeze(1), 
        first_16.unsqueeze(0), 
        dim=2
    )
    print(f"  - Mean similarity between tiles: {similarities.mean():.4f}")
    print(f"  - Max similarity: {similarities.max():.4f}")
    print(f"  - Min similarity (off-diagonal): {similarities[similarities < 0.9999].min():.4f}")
    
    # Step 8: Save embeddings
    output_path = "/home/lotems/Documents/DL_Oren/CSC-BSR/embeddings_test_output.pt"
    torch.save({
        'embeddings': batch_embeddings,
        'tile_files': tile_files,
        'original_image': test_image,
        'board_shape': (8, 8)
    }, output_path)
    print(f"\n[Step 8] Embeddings saved to: {output_path}")
    
    # Cleanup
    import shutil
    shutil.rmtree(temp_dir)
    print(f"\n[Cleanup] Temporary directory removed")
    
    print("\n" + "=" * 80)
    print("✓ Test completed successfully!")
    print("=" * 80)
    
    return batch_embeddings, tile_files


if __name__ == "__main__":
    embeddings, tiles = test_embedding_pipeline()
