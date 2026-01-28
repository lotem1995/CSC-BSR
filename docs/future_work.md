---
title: What didn’t work + Future work
nav_order: 99
---

# What didn’t work + Future work

<div class="bs-hero">
	<p class="fs-5">A short log of approaches we explored, why they didn’t pan out, and what we would do next.</p>

	<div class="bs-cards">
		<div class="bs-card">
			<h3 class="mt-0">What didn’t work</h3>
			<p class="mb-0">Ideas we tried early to validate quickly, then dropped (or deferred) based on results and integration friction.</p>
		</div>
		<div class="bs-card">
			<h3 class="mt-0">Future work</h3>
			<p class="mb-0">Two concrete directions that can improve robustness (OOD) and unlock end-to-end decision-making.</p>
		</div>
	</div>
</div>

## What didn’t work

### 1) DSPy + local LLM (Ollama) for classification
We started with a "VLM / prompting first" approach in [DSPy-Classifier/dspy-chess-classifier.py](https://github.com/lotem1995/CSC-BSR/blob/main/DSPy-Classifier/dspy-chess-classifier.py).

{: .warning }
DSPy did not integrate cleanly with Ollama for our setup.

- Ollama expected requests in a specific input format/encoding.
- DSPy’s smoothest path is via OpenAI-compatible APIs; we hit repeated friction trying to make the local stack behave the same.
- Outcome: good for a quick baseline idea, but not a reliable foundation for experiments we needed to iterate on.

{: .decision }
We moved to a vision-encoder + lightweight head pipeline, where inputs/outputs are deterministic and easier to debug.

#### Code pointers (DSPy attempt)
{: .repro }
Start here if you want to reproduce/debug this attempt:

- [DSPy-Classifier/dspy-chess-classifier.py](https://github.com/lotem1995/CSC-BSR/blob/main/DSPy-Classifier/dspy-chess-classifier.py): main DSPy program
	- `Config` / `setup_dspy()`: wires DSPy to `ollama_chat/<model>` + `api_base`
	- `ChessPieceClassifier.forward()`: whole-board (preferred)
	- `ChessPieceClassifier.forward_square()`: per-square (used by evaluation)
	- `DSPyOptimizer.optimize()`: tries `BootstrapFewShot`
- [DSPy-Classifier/train_chess_classifier.py](https://github.com/lotem1995/CSC-BSR/blob/main/DSPy-Classifier/train_chess_classifier.py): SLURM/cluster training wrapper + checkpoint/log handling

### 2) VAE-based OOD detection
We also explored a VAE as an out-of-distribution detector (see the code in [VAE/](https://github.com/lotem1995/CSC-BSR/tree/main/VAE/)).

{: .result }
It did not perform well enough to justify the complexity.

What we tried:
- **Model**: convolutional encoder/decoder with a latent bottleneck.
- **Loss**: reconstruction (MSE) + KL divergence.
- **Goal**: treat “unfamiliar” images as high-error (OOD) samples.

Directions we tested:
- **Reconstruction error**: flag samples with the largest reconstruction error.
- **Latent cycling**: pass through the model multiple times and compare drift in latent space (more cycles sometimes improved results).
- **Online noise optimization**: optimize input noise at inference to maximize difference, then compare (heavier, fragile).

{: .note }
In practice, the VAE tended to be sensitive to nuisance factors (lighting, motion blur, cropping) rather than “semantic OOD” in the chess sense.

#### Code pointers (VAE attempt)
{: .repro }
Start here if you want to rerun the OOD experiments:

- [VAE/VAE_nn.py](https://github.com/lotem1995/CSC-BSR/blob/main/VAE/VAE_nn.py): VAE architecture + `loss_function()` (MSE + KL)
- [VAE/train_VAE.py](https://github.com/lotem1995/CSC-BSR/blob/main/VAE/train_VAE.py): training loop (dataloaders + checkpoints + early stopping)
- [VAE/model_evaluation.py](https://github.com/lotem1995/CSC-BSR/blob/main/VAE/model_evaluation.py): evaluation utilities (reconstruction error + drift)
- [VAE/detect_ood.py](https://github.com/lotem1995/CSC-BSR/blob/main/VAE/detect_ood.py): single-image OOD via multi-cycle latent drift (`predict_single_image_ood()`)
- [VAE/calculate_treshold.py](https://github.com/lotem1995/CSC-BSR/blob/main/VAE/calculate_treshold.py): cycle/threshold selection helpers
- [VAE/single_photo.py](https://github.com/lotem1995/CSC-BSR/blob/main/VAE/single_photo.py): quick reconstruction visualization sanity-check

## Future work

### OOD with spatial consistency across neighboring tiles
{: .decision }
Use the relationship between adjacent tiles to detect “impossible” or unlikely local patterns.

- Add an OOD score that incorporates neighborhood agreement (e.g., smoothness/consistency priors, local relational features).
- This should reduce false OOD flags caused by single-tile noise.

### Predict the next move as part of a policy (RL)
{: .decision }
Add a head that predicts the next move, trained as a policy (reinforcement learning).

- Treat board-state reconstruction as part of a perception-to-action pipeline.
- The policy head can provide an additional self-consistency signal: “does this board state lead to plausible next moves?”.
