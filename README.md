# CSC-BSR (Chessboard Square Classification and Board-State Reconstruction)
This codebase supports Project 1 in "Introduction to Deep Learning" 2025~2026 at Ben Gurion University. The project's objective is to classify the type of each square on a chessboard and reconstruct the entire board state from input data.


how we trained the binary_ood - we took the 3rd epoch since after that the validation results got down
python embedding/train_binary_ood.py --dino-size small --batch-size 8 --epochs 5


how to train the VAE - no need for argument, it will auto stop if the val loss will go up - models will be saved in checkpoint dir
python -m VAE.train_VAE

and for a simple evaluation, use python VAE/model_evaluation.py and make sure the file include the right model you want to load