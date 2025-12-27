"""
Debug script to understand the processor output structure
"""

import torch
from PIL import Image
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig

# 4-bit Quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
)

device = torch.device("cuda")

model = Qwen3VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen3-VL-2B-Instruct",
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
).eval()

processor = AutoProcessor.from_pretrained(
    "Qwen/Qwen3-VL-2B-Instruct", 
    trust_remote_code=True,
    min_pixels=128*28, 
    max_pixels=256*28
)

# Load a test image
test_image = "/home/lotems/Documents/DL_Oren/CSC-BSR/data/game2_per_frame/tagged_images/frame_000200.jpg"
img = Image.open(test_image).convert('RGB')

# Test with single image
print("=" * 80)
print("Testing with 1 image")
print("=" * 80)
inputs = processor(text=[""], images=[img], return_tensors="pt").to(device)
print(f"inputs.pixel_values shape: {inputs.pixel_values.shape}")
print(f"inputs.image_grid_thw: {inputs.image_grid_thw}")

with torch.no_grad():
    visual_output = model.visual(inputs.pixel_values, grid_thw=inputs.image_grid_thw)
    
visual_features = visual_output[0]
print(f"visual_features shape: {visual_features.shape}")
print(f"Tokens per image (h*w): {inputs.image_grid_thw[0, 1] * inputs.image_grid_thw[0, 2]}")

# Test with 4 images
print("\n" + "=" * 80)
print("Testing with 4 images (same image repeated)")
print("=" * 80)
images = [img] * 4
inputs = processor(text=[""] * 4, images=images, return_tensors="pt").to(device)
print(f"inputs.pixel_values shape: {inputs.pixel_values.shape}")
print(f"inputs.image_grid_thw:\n{inputs.image_grid_thw}")

with torch.no_grad():
    visual_output = model.visual(inputs.pixel_values, grid_thw=inputs.image_grid_thw)

visual_features = visual_output[0]
print(f"visual_features shape: {visual_features.shape}")
print(f"Expected tokens: {(inputs.image_grid_thw[:, 1] * inputs.image_grid_thw[:, 2]).sum()}")
print(f"Tokens per image: {(inputs.image_grid_thw[:, 1] * inputs.image_grid_thw[:, 2]).tolist()}")


