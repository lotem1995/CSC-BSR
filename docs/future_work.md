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
We started with a "VLM / prompting first" approach in [DSPy-Classifier/dspy-chess-classifier.py](../DSPy-Classifier/dspy-chess-classifier.py).

{: .warning }
DSPy did not integrate cleanly with Ollama for our setup.

- Ollama expected requests in a specific input format/encoding.
- DSPy’s smoothest path is via OpenAI-compatible APIs; we hit repeated friction trying to make the local stack behave the same.
- Outcome: good for a quick baseline idea, but not a reliable foundation for experiments we needed to iterate on.

{: .decision }
We moved to a vision-encoder + lightweight head pipeline, where inputs/outputs are deterministic and easier to debug.

<details>
	<summary><strong>Code map (DSPy attempt)</strong></summary>

	{: .repro }
	Where the logic lives and what it does:

	- [DSPy-Classifier/dspy-chess-classifier.py](../DSPy-Classifier/dspy-chess-classifier.py): core DSPy program.
		- `Config` + `setup_dspy()`: configures `dspy.LM` with `ollama_chat/<model>` and an Ollama `api_base`.
		- `BoardStateSignature` / `PieceClassificationSignature`: output schemas for “whole board in one call” vs “single square”.
		- `ChessPieceClassifier.forward()`: preferred path (1 call per frame) returning JSON + FEN + confidence.
		- `ChessPieceClassifier.forward_square()`: legacy per-square call; used by `evaluate()`.
		- `load_real_dataset()`: expands each FEN into 64 square-level examples (useful for metrics, expensive).
		- `DSPyOptimizer.optimize()`: tries `BootstrapFewShot` to auto-select demonstrations and improve a metric.

	- [DSPy-Classifier/train_chess_classifier.py](../DSPy-Classifier/train_chess_classifier.py): training/cluster wrapper.
		- Handles SLURM env detection, logging, checkpoint folders, and dynamic import of the classifier module.
</details>

### 2) VAE-based OOD detection
We also explored a VAE as an out-of-distribution detector (see the code in [VAE/](../VAE/)).

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

<details>
	<summary><strong>Code map (VAE attempt)</strong></summary>

	{: .repro }
	Where the logic lives and what it does:

	- [VAE/VAE_nn.py](../VAE/VAE_nn.py): the model definition.
		- Convolutional encoder/decoder with a latent vector.
		- `loss_function()`: reconstruction MSE (sum) + KL divergence.

	- [VAE/train_VAE.py](../VAE/train_VAE.py): training loop.
		- Uses `preprocessing.load_dataset.get_train_dataloader()` / `get_val_dataloader()`.
		- Early stopping + periodic checkpoints + “best model” saving.

	- [VAE/model_evaluation.py](../VAE/model_evaluation.py): analysis scripts.
		- Reconstruction-error anomaly ranking (`show_top_anomalies`).
		- Multi-cycle drift computation (`get_multicycle_scores`).

	- [VAE/detect_ood.py](../VAE/detect_ood.py): the OOD detector experiment.
		- `predict_single_image_ood()`: encodes → decodes → re-encodes for `cycles` and sums latent drift.
		- Scans a dataset and visualizes the top detected OOD samples.

	- [VAE/calculate_treshold.py](../VAE/calculate_treshold.py): choosing hyperparameters.
		- `find_optimal_amount_of_cycles()`: picks cycles by maximizing ID/OOD separation.
		- `analyze_ood_threshold()`: ROC-based threshold selection.

	- [VAE/single_photo.py](../VAE/single_photo.py): quick sanity check.
		- Loads a checkpoint and visualizes reconstruction on a single tile image.
</details>

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
