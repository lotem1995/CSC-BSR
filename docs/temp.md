

## Training
<div class="bs-cards" markdown="block">

<div class="bs-card" markdown="block">
### Pre-Processing and Creating the Dataset

<!-- In this project, we received a couple of chess games, each divided into game frames (individual snapshots of the game at each move). To build a concrete standard dataset, we wrote a script that generates the dataset from the frames using a YAML (YAML Ain't Markup Language) configuration file, a human-readable data-serialization language. -->

The dataset consists of chess games, each divided into multiple frames (individual snapshots of the game at each move). To create a standardized dataset, we developed a preprocessing script that processes these frames based on a YAML (YAML Ain't Markup Language) configuration file, which is a human-readable data-serialization language.

<!-- We split the data by games into train, val, and test sets. This approach helps us mitigate data leakage and excessive similarity between these sets. -->

We divided the dataset into training, validation, and test sets based on entire games. This method helps prevent data leakage and reduces similarity between the sets.

<!-- During preprocessing, we converted each game's frame into images of the chessboard tiles. The result is a directory called `preprocessed_data` containing a tile image for each frame from all the games in the `data` directory. There is also a `splits` directory with the manifest JSON file and three CSV files: `train.csv`, `val.csv` and `test.csv`. -->

During preprocessing, we transformed each game's frame into individual images of the chessboard tiles. The output is a directory named `preprocessed_data`, which contains tile images for each frame from all the games in the `data` directory. Additionally, there is a `splits` directory that includes a manifest JSON file and three CSV files: `train.csv`, `val.csv`, and `test.csv`.

<!-- You can read more about this step on the [Pre-Processing](preprocessing.md) page. -->
[Read more →](preprocessing.md){: .btn .btn-outline }
</div>

<div class="bs-card" markdown="block">
### Tile Embedding
The next step in the architecture is a transformer-based encoding of tile images. Instead of creating an embedding from scratch, we opted to fine-tune an existing transformer for this task. Here, we tested several options and models, including LoRA fine-tuning of QWEN3-VL-7B and DINOv2 fine-tuning. After testing, we chose to use a fine-tuned DINOv2 model for tile embedding. This model takes an image of a tile as input and outputs a 1024-dimensional embedding vector representing the tile.

For fine-tuning the DINOv2 model, we tested training only the classification head as well as fine-tuning the entire model (backbone + head). We found that fine-tuning the entire model yielded better results. Originally, we used a classification head just for the fine-tuning process, as we looked for a way for the embedding to know how good it is. However, after fine-tuning, we discarded the classification head and used only the backbone for generating embeddings.

<!-- You can read more about this step on the [Tile Embedding](tile_embedding.md) page. -->
[Read more →](tile_embedding.md){: .btn .btn-outline }
</div>
<div class="bs-card" markdown="block">
### Tile and Board Classification
After obtaining the tile embeddings, we proceed to classify each tile and reconstruct the board state. We experimented with various architectures for this task, including MLP, KNN, and Mahalanobis distance-based classifiers. Ultimately, we selected a KNN-based classifier for tile classification and board reconstruction.

For OOD (Out-Of-Distribution) detection, we added a special “unknown” class to the classifier trained during fine-tuning. This class allows the classifier to identify tiles that do not represent standard chess pieces, such as unfamiliar or novel objects. During preprocessing, any frames containing OOD pieces were specifically tagged to train the model to recognize and separate these cases from regular chessboard pieces. Originally, such frames were tagged only with a regular FEN tag, but for our purposes, we marked them distinctly for OOD detection.

<!-- You can read more about this step on the [Tile and Board Classification](tile_and_board_classification.md) page. -->
[Read more →](tile_and_board_classification.md){: .btn .btn-outline }
</div>

<div class="bs-card" markdown="block">
## Results
n this section, we will present the results of our architecture and methods. We will discuss the model's performance on the test set, including accuracy metrics for tile classification and board reconstruction. We will also analyze the model’s ability to handle out-of-distribution scenarios.

<!-- TODO: include results here -->

<!-- You can read more about this step on the [Results](results.md) page. -->
[Read more →](results.md){: .btn .btn-outline }
</div>

<div class="bs-card" markdown="block">
## Conclusion
In this project, we developed a comprehensive architecture for chessboard square classification and board-state reconstruction. We utilized a transformer-based tile embedding approach combined with a KNN-based classifier to achieve accurate results. Our model demonstrated strong performance on the test set and effectively handled out-of-distribution scenarios.

<!-- TODO: include conclusions here -->
</div>

<div class="bs-card" markdown="block">
## Libraries and Tools
- Python 3.13/3.14
- PyTorch
- CUDA 13.1
- Hugging Face Transformers
- Numpy
- Pandas
- Scikit-learn
- Matplotlib
- tqdm
- loguru
- OpenCV
- YAML
</div>

<div class="bs-card" markdown="block">
## Running the Code
Instructions for running the code will be provided in the `README.md` file of the repository. Please refer to that document for detailed setup and execution guidelines, and ensure you follow each step carefully to get the code running. Our codebase ran on an RTX 4070 Laptop GPU, so having a similar or better GPU is recommended for optimal performance. This also demonstrates our model's lightweight nature, as the RTX 4070 Laptop GPU is roughly equivalent to an RTX 4060 Desktop or sits just below an RTX 4060 Ti Desktop.
</div>
<div class="bs-card" markdown="block">
## Other Things We Tried
Throughout the project, we experimented with various ideas and methods that we ultimately did not pursue in our final architecture. Some of these include:

- Trying to use DSPy and Ollama with QWEN3-VL for solving the project.
- Using VAE (Variational Autoencoder, a type of neural network for encoding and reconstructing data) for OOD detection.
- Implementing different classifier architectures, such as MLP (Multi-Layer Perceptron, a neural network) and Mahalanobis distance-based classifiers (which use a statistical distance metric).

<!-- You can read more about these experiments on the [Other Things We Tried](other_things_we_tried.md) page. -->
[Read more →](other_things_we_tried.md){: .btn .btn-outline }
</div>