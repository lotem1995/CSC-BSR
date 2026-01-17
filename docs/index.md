---
title: Home
nav_order: 1
---

<div class="bs-hero" markdown="block">

# CSC-BSR (Chessboard Square Classification and Board-State Reconstruction)
CSC-BSR is a project for the course Introduction to Deep Learning by Oren Friefeld and Roy Amoyal at Ben Gurion University. The project was presented by Lotem Sakira, Gilad Lifshitz, and Adi Shoam.
In this project, we try to classify a chessboard game into a FEN format. We also reconstruct the board from the fen, while handling out-of-distribution scenarios.
On this GitHub Pages site, we will explore the final architecture and our reasoning. We will also review previous ideas and methods we tested but did not pursue.

<span class="fs-5">
[Architecture](#architecture){: .btn .btn-purple .mr-2 }
[Results](results.md){: .btn .btn-outline .mr-2 }
[Code](https://github.com/lotem1995/CSC-BSR){: .btn .btn-blue }
</span>
</div>

{: .repro }
## Architecture
Here, we will briefly describe the project's architecture. Each section has a dedicated page with more details.

## Architecture

<div class="bs-cards" markdown="block">

<div class="bs-card" markdown="block">
### Pre-Processing
Converting the frame into tile images  
[Read more →](preprocessing.md){: .btn .btn-outline }
</div>

<div class="bs-card" markdown="block">
### Tile Embedding
Fine-tuned DINOv2 backbone based tile embedding. 
[Read more →](tile_embedding.md){: .btn .btn-outline }
</div>

<div class="bs-card" markdown="block">
### Tile + Board Classification and board reconstruction
KNN tile classifier + DINOv2 binary OOD detection + board reconstruction. 
[Read more →](tile_and_board_classification.md){: .btn .btn-outline }
</div>
</div>