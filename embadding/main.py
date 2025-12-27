import torch
from PIL import Image
from typing import List
# Qwen3-VL uses the same class structure as Qwen2 but with improved weights/ViT
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig

class QwenVisionEmbedding:
    def __init__(self, model_name: str = "Qwen/Qwen3-VL-2B-Instruct"):
        self.device = torch.device("cuda")
        
        # 4-bit Quantization: Cuts VRAM usage from 5GB down to ~2GB
        # This leaves plenty of room for your MCTS tree and other RL logic
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True
        )

        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto", # Automatically uses your 4070
            trust_remote_code=True
        ).eval()
        
        # Optimization: Limit pixels since chess squares are small
        # This speeds up the vision tower significantly
        self.processor = AutoProcessor.from_pretrained(
            model_name, 
            trust_remote_code=True,
            min_pixels=128*28, 
            max_pixels=256*28
        )

    @torch.no_grad()
    def extract_embedding(self, image: Image.Image) -> torch.Tensor:
        # Optimization: Don't use messages/chat templates for raw ViT extraction 
        # unless necessary. We can go straight to the visual tower.
        
        # Ensure image is the right size for the processor
        # Add empty text to satisfy processor requirements
        inputs = self.processor(
            text=[""],
            images=[image], 
            return_tensors="pt"
        ).to(self.device)

        # We go directly to the 'visual' tower. 
        # This bypasses the LLM "Thinking" part, saving ~80% of the time.
        # The processor returns 'image_grid_thw' which tells us the grid layout
        grid_thw = inputs.get('image_grid_thw')
        visual_output = self.model.visual(inputs.pixel_values, grid_thw=grid_thw)
        
        # The visual model returns a tuple (patches, layer_outputs)
        # We use the main patch embeddings
        if isinstance(visual_output, tuple):
            visual_features = visual_output[0]
        else:
            visual_features = visual_output
        
        # Global Average Pooling to get a 1D vector (embedding)
        embedding = torch.mean(visual_features, dim=0)
            
        return embedding.cpu()
    
    @torch.no_grad()
    def extract_batch_embeddings(self, images: List[Image.Image]) -> torch.Tensor:
        """
        Extracts embeddings for a list of images (e.g., all 64 squares) in one pass.
        Note: Due to how Qwen3-VL batching works, we process each image individually
        to ensure correct per-image embeddings.
        """
        batch_embeddings = []
        
        for image in images:
            # Process one image at a time to ensure correct embedding extraction
            inputs = self.processor(
                text=[""],
                images=[image], 
                return_tensors="pt"
            ).to(self.device)
            
            # Extract embedding for this single image
            grid_thw = inputs.get('image_grid_thw')
            visual_output = self.model.visual(inputs.pixel_values, grid_thw=grid_thw)
            
            # The visual model returns a tuple (patches, layer_outputs)
            if isinstance(visual_output, tuple):
                visual_features = visual_output[0]
            else:
                visual_features = visual_output
            
            # Mean pool the patches to get image embedding
            embedding = torch.mean(visual_features, dim=0, keepdim=True)
            batch_embeddings.append(embedding)
        
        # Stack all embeddings
        batch_embeddings = torch.cat(batch_embeddings, dim=0)
        return batch_embeddings.cpu()

if __name__ == "__main__":
    # Test with the 2B model
    try:
        extractor = QwenVisionEmbedding("Qwen/Qwen3-VL-2B-Instruct")
        print("Qwen3-VL initialized and ready for chessboard analysis!")
    except Exception as e:
        print(f"Error: {e}")
        print("Tip: Run 'huggingface-cli login' to ensure your environment is authenticated.")