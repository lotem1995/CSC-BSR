"""
Fine-tuning with LoRA (Low-Rank Adaptation)

LoRA is WAY more efficient than full fine-tuning:
- Uses 10x less VRAM
- 10x faster
- Same quality results
"""

from peft import get_peft_model, LoraConfig, TaskType
import torch
from torch.utils.data import DataLoader
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig


class QwenLoRAFineTuner:
    """
    Fine-tunes Qwen3-VL using LoRA adapters (efficient).
    """
    
    def __init__(self, model_name: str = "Qwen/Qwen3-VL-2B-Instruct"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model with quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True
        )
        
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
        
        self.processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True,
            min_pixels=128*28,
            max_pixels=256*28
        )
        
        # Configure LoRA
        lora_config = LoraConfig(
            r=8,  # LoRA rank (lower = smaller, but less expressive)
            lora_alpha=16,
            target_modules=["q_proj", "v_proj"],  # Which modules to adapt
            lora_dropout=0.1,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        
        # Apply LoRA to model
        self.model = get_peft_model(self.model, lora_config)
        
        print(f"Model parameters: {self.model.num_parameters()}")
        print(f"Trainable parameters: {self.model.num_parameters(only_trainable=True)}")
        print(f"LoRA efficiency: {100 * self.model.get_nb_trainable_parameters()[0] / self.model.get_nb_trainable_parameters()[1]:.2f}%")
    
    def train_batch(self, images, labels):
        """Fine-tune batch with LoRA (much faster!)"""
        # ... same as before but with LoRA handling
        pass


# In practice, it's as simple as:
# from peft import get_peft_model, LoraConfig
# 
# lora_config = LoraConfig(
#     r=8,
#     lora_alpha=16,
#     target_modules=["q_proj", "v_proj"],
# )
# model = get_peft_model(model, lora_config)
# # Now train as normal - only 1% of parameters are trained!