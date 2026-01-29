# CSC-BSR (Chessboard Square Classification and Board-State Reconstruction)

This codebase supports Project 1 in “Introduction to Deep Learning” 2025~2026 at Ben Gurion University. The project’s objective is to classify the type of each square on a chessboard and reconstruct the entire board state from input data.

This is a link to our drive - https://drive.google.com/drive/folders/1yPKveqW-QtuvII25boNpUeg8psyurwJX?usp=sharing

- 🚀 [Quick Start](#setting-up-the-environment)
- 🧠 [Run Predictions](#running-predictions-on-images)
- 🏋️ [Training](#model-training-notes)
  - [Quick Start: train.py](#quick-start-complete-training-pipeline)
  - [Data Placement](#data-placement-instructions)
  - [Manual Training](#data-preparation-manual-method)

## Setting up the environment
To set up the Python environment, create a virtual environment using `venv` or `conda`. Then install the required packages with the following command:

```bash
pip install -U pip
pip install -r requirements.txt
```

**Note:** This project was tested against Python 3.14, but should work with other versions as well.

### Quick Training Command Reference

```bash
# Complete training pipeline (recommended)
python train.py --stage all --epochs 10 --batch-size 32

# Individual stages
python train.py --stage preprocess  # Prepare dataset
python train.py --stage embedding   # Train DINO backbone
python train.py --stage ood         # Train OOD guard

# See all options
python train.py --help
```

See the [Model Training Notes](#model-training-notes) section for detailed instructions.

### Note on PyTorch installation
Since PyTorch installation can vary based on your system configuration (OS, CUDA version, etc.), please refer to the official [PyTorch installation guide](https://pytorch.org/get-started/locally/) for the most suitable command to install PyTorch and torchvision.
**The requirements.txt file does not include PyTorch due to these variations.**

### Note on HuggingFace Transformers installation
The code uses `huggingface-hub` and `transformers` libraries. For that, you need to register your HuggingFace account’s token by running:
```bash
hf auth login
```
For more information, check the [HuggingFace documentation](https://huggingface.co/docs/huggingface_hub/en/guides/cli).
## Running predictions on images

To predict the board state from a chessboard image, run the following command:
```bash
python predict_board.py
```

**What it does:**
* Loads the fine-tuned DINO backbone, FEN classifier, and OOD guard models
* Processes a chessboard image by slicing it into 64 tiles (8x8 grid)
* Predicts the piece type for each square
* Generates a visual representation of the prediction
* Saves the output to `results/prediction_visual.png`

**Customizing the input image:**
By default, the script processes `data/game4_per_frame/tagged_images/frame_039084.jpg`. To use a different image, modify the `target_image` variable in the `__main__` section of the script.
You can also use these main functions to create your own testing script - a basic example would probably use these 3 functions: `load_models`, `predict_board`, `generate_ood_board` in a logic similar to this
```python
import numpy as np
from PIL import Image
from predict_board import load_models, predict_board
from drawing.draw_board import generate_ood_board

# 1. Initialize models once
classifier = load_models()

# 2. Load your image as a Numpy array (RGB)
image_path = "path/to/your/chess_board.jpg"
with Image.open(image_path) as img:
    image_np = np.array(img.convert("RGB"))

# 3. Run prediction
prediction_tensor = predict_board(image_np, classifier)
print("Prediction Tensor:\n", prediction_tensor)

# 4. Save visualization
generate_ood_board(prediction_tensor, "output_visualization.png")
```

**Output:**

* Console output showing prediction progress and model loading status
* Visual board representation saved to `results/prediction_visual.png`
* 8x8 tensor with piece classifications (0-12: pieces, 13: OOD/unknown)

## Model Training Notes

### Data Preparation

Prepare the dataset so training scripts can access images and labels.
To prepare a custom dataset, use `preprocessing/dataset_config.example.yaml` as a template to create your own `preprocessing/dataset_config.yaml` file, or move your data to match the default paths in the example.

To create the dataset, execute: 
```bash
python preprocessing/build_dataset.py --config preprocessing/dataset_config.example.yaml
```
### Training the embedding model

As detailed in the report, the DINO model was fine-tuned to serve as the embedding backbone.
To train the DINO embedding model, run the following command:

```bash
python embedding/fine_tune.py --splits-dir data/splits --path-root . --epochs 1 --batch-size 2 --num-workers 2 --embedding-model dino-small --strategy backbone
```

In order to check the performance of the other strategies, run embedding/experiment_runner.py with the appropriate arguments:
```bash
python embedding/experiment_runner.py --epochs 1 --batch-size 2 --num-workers 2
```
### Training the classifier

how we trained the binary_ood - we took the 3rd epoch since after that the validation results got down
```bash
python embedding/train_binary_ood.py --dino-size small --batch-size 8 --epochs 5
```


## Other things we tried (detailedon the site)
### VAE Classifier for Out-of-Distribution Detection
The directory `VAE` contains code for training a Variational Autoencoder (VAE) classifier, as described in the report.
To train the VAE classifier, run:
```bash
python -m VAE.train_VAE
```
The training will stop automatically if the validation loss increases. Models are saved in the checkpoint directory.

For a simple evaluation, run:
```bash
python VAE/model_evaluation.py
```
Make sure the file includes the right model you want to load!

### DSPy Classifier
The directory `DSPy-Classifier` contains code for training a DSPy classifier, as described in the report. It is not integrated into the main pipeline since DSPy couldn't communicate with ollama properly during the project period.

**Note:** Be aware that `requirements.txt` does not include the DSPy library and ollama, so you need to install them manually.
