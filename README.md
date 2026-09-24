# Multimodal Image Captioning

Status: Phases 1–12 verified on the user's Mac. Person 1 is complete; explicit handoff acknowledgment is pending.
Last updated: 2026-09-23.

## Goal and current scope

Build an image-to-caption system with a ResNet50 + LSTM baseline, then a ViT + Transformer captioning model, attention visualizations, BLEU/CIDEr evaluation, and a working application.

This is an advanced academic/engineering project, not a research-paper reproduction. Development uses a limited COCO dataset to keep work manageable on an Apple-silicon Mac. Full-COCO training is no longer required.

Two people work sequentially. Person 1 completes phases 1–12 and hands over verified infrastructure. Person 2 starts only after an explicit handoff. Implement one phase at a time; do not advance until the current check passes and the user requests the next phase. Architectural or interface changes require explicit approval.

## Dataset

| Use | Images | Reference captions | Status |
| --- | ---: | ---: | --- |
| Training | 1,000 | 5,008 | Verified |
| Validation | 250 | 1,250 | Verified |
| Final test | 250 planned | Not measured | Not selected/downloaded yet |

Official COCO 2017 train IDs form the training pool. Official validation IDs are deterministically divided into 2,500 project-validation and 2,500 held-out-test IDs. The working validation set is a 250-image subset. This is a custom project evaluation split, not a standard COCO benchmark split.

Full partition IDs in splits.json do not mean all image files are downloaded. All captions for an image stay in the same partition. Original references remain unchanged. Test data must not guide training or model selection.

Development image fingerprint:
`8d5298ee64037de6ea8b62dc25f3aaae54f7d421b448e1100dcc9df3a018c214`

## Verified environment

Python 3.11 environment; Apple MPS selected and checked on the user's Mac.

| Package | Observed version |
| --- | --- |
| torch | 2.14.0 |
| torchvision | 0.29.0 |
| transformers | 4.57.6 |
| numpy | 2.4.6 |
| Pillow | 12.3.0 |
| PyYAML | 6.0.3 |

requirements.txt contains initial version ranges, not a cross-platform lockfile. The Phase 1 report records the actual environment.

## Setup and verified commands

From the project root, activate the project's own environment:

```bash
source .venv/bin/activate
python -m pip check
python -m src.check_environment
```

Dataset setup (reuses completed downloads):

```bash
python -m src.data.prepare --config configs/baseline_debug.yaml --val-images 250
python -m src.data.validate --config configs/baseline_debug.yaml --val-images 250
```

Preprocessing, tokenizer and loader checks:

```bash
python -m src.data.check_preprocessing --config configs/baseline_debug.yaml --val-images 250 --samples-per-split 4
python -m src.data.check_tokenizer --config configs/baseline_debug.yaml --val-images 250 --offline
python -m src.data.check_dataloader --config configs/baseline_debug.yaml --val-images 250 --batches 2
python -m src.models.check_baseline --config configs/baseline_debug.yaml --val-images 250 --samples 2
```

The first tokenizer setup omits --offline. The first baseline construction may download ImageNet encoder weights. No BERT neural-network weights are used by the tokenizer.

## What has been verified

1. Environment/configuration: dependency consistency, syntax, device selection, tensor forward/backward on MPS.
2. Dataset: image-caption mapping, split separation, image decoding/dimensions, and no byte-identical images across the development splits.
3. Images: RGB, whole-image square resize to 224×224, float32 tensor, ImageNet normalization, deterministic transforms, and visual preview. Square resizing preserves the scene but may distort aspect ratio. No augmentation is enabled.
4. Captions: saved bert-base-uncased WordPiece tokenizer, 30,522 entries, BOS=101, EOS=102, PAD=0, UNK=100. Maximum 32 tokens includes BOS/EOS. Training truncation: 6/5,008 (0.12%); validation: 1/1,250 (0.08%); no UNK captions. Offline reload reproduces IDs.
5. Batching: 5,008 training pairs, 1,250 validation-loss pairs, and 250 image-level generation samples. At batch size 8: 626, 157 and 32 batches respectively. Dynamic padding, one shift, final partial batch, grouped references, generator-state reproducibility, and MPS transfer passed.
6. Baseline: ResNet50 ImageNet V2 features initialize LSTM hidden/cell states. Forward, backward, frozen BatchNorm, greedy generation, EOS/padding, and mode restoration passed on two samples. Total parameters: 50,654,586; trainable: 27,146,554; frozen: 23,508,032. Diagnostic loss 10.314627 is from an untrained decoder, not a caption-quality result.

## Stable interfaces

- `load_config(path)` resolves paths relative to the project root.
- `load_image(path)` and `build_image_transform(image_size, split, mean, std)` provide image preprocessing.
- `CaptionTokenizer.encode(caption)` returns an unpadded list including BOS/EOS.
- Only `CaptionCollator` shifts sequences: inputs = full[:-1], targets = full[1:]. No model or loss repeats the shift.
- Caption batches: images float32 [B,3,H,W]; input_ids int64 [B,T]; attention_mask bool [B,T] (True means valid); target_ids int64 [B,T], padding=-100; image_ids list[int]. T <= 31 at the current length limit.
- Generation batches: images, image_ids, references (all original reference captions per image).
- `build_dataloaders(config, val_images=250)` returns train, validation, generation_validation and tokenizer.
- `build_model(config, tokenizer, load_pretrained=None)` creates the model; checkpoint reconstruction can later disable pretrained downloading.
- `model.forward(images, input_ids, attention_mask)` returns logits [B,T,V].
- `model.generate(images, max_new_tokens, strategy='greedy', beam_size=1)` returns sequences [B,L] excluding BOS, retaining generated EOS and padding after EOS; attention is None for the baseline.

## Phase 7: verified on MPS

Files: src/training/losses.py, optimizer.py, scheduler.py, trainer.py, check_trainer.py, and package initializer.

Token-masked cross-entropy averages valid targets. Epoch reporting uses total token loss divided by total valid tokens, not an unweighted mean of batch averages. AdamW includes trainable parameters only and accepts explicitly supplied parameter groups. ReduceLROnPlateau steps after validation; early stopping follows the configured monitor/mode. Gradient clipping follows backward (and unscaling when AMP is enabled).

The current Mac run keeps mixed_precision=false. The optional AMP implementation is CUDA-only and is not verified by an MPS run. Requesting this AMP path on another device raises a clear error rather than silently changing precision.

Trainer accepts a model-independent batch/forward contract and an optional epoch-end callback for Phase 8 persistence. No caption checkpoints, resume implementation, TensorBoard logging or best-weight restoration are supplied in Phase 7. A later metric callback can provide additional validation metrics without model-specific logic inside the Trainer.

Verified command:

```bash
python -m src.training.check_trainer --config configs/baseline_debug.yaml --val-images 250
```

This is a disposable check using six training pairs and four validation pairs, batch size two, three optimizer updates, and two validation batches. No full-dataset epoch is run. Updated weights are discarded when the process exits. The check also verifies padding loss/gradients, scheduler behavior, early-stopping bookkeeping, weight updates, and validation without updates. Report goes to artifacts/trainer_checks/<unique-run-id>/report.json.

Verified on 2026-09-23: PHASE 7 TRAINER CHECK: PASS. The disposable run completed three optimizer updates with train_loss=10.3382 and val_loss=10.2756. Masked loss and padding gradients, scheduler/early-stopping bookkeeping, trainable weight updates, frozen encoder, and validation without updates or gradients all passed. These losses are diagnostic, not caption-quality measurements.

Report on the user's Mac: `artifacts/trainer_checks/20260923T050822_983202Z_ea0e9dc8/report.json`. No captioning checkpoint was saved. The MPS check does not verify CUDA AMP.

New interface: `Trainer(config, device, optimizer, scheduler=None, loss_fn=None)`; `fit(model, train_loader, validation_loader, epochs=None, max_train_batches=None, max_val_batches=None, on_epoch_end=None, metric_fn=None)`; `validate(model, loader, max_batches=None)`. Move the model to its device before constructing its optimizer. fit's epochs argument is a total target epoch count, not additional epochs. Epoch callbacks run after scheduling and early-stopping bookkeeping.

## Phase 8: verified on CPU and MPS

New files: `src/training/checkpoints.py`, `src/training/run_logging.py`, and `src/training/check_checkpointing.py`. Existing Trainer/model/data/config interfaces are unchanged.

Version 1 checkpoints save model, optimizer, scheduler, AMP scaler, epoch/global step, complete metric history, early-stopping state, Python/NumPy/PyTorch/device random states and loader-generator states. Resume is at completed epoch boundaries, on the same device and PyTorch version, with matching configuration and run context and num_workers=0. Mid-epoch continuation is not implemented. Reconstruct objects first, load state before creating any loader iterator, then call fit with the total target epoch count. A larger fit epochs argument does not require changing the stored config.

A run context identifies its data/tokenizer/preprocessing: production callers must include the selected manifest fingerprint, tokenizer fingerprint, preprocessing description, and any batch limits. The synthetic test uses a clearly labelled synthetic context. Context equality is checked before changing model weights; a path alone is not a data fingerprint. A failed restore after state loading begins should be retried with fresh model/trainer objects.

Files in each run: `checkpoints/index.json` references unique `.pt` files for last and best; `metrics.json` preserves full epoch records; `metrics.csv` exports common scalar fields; `training.log` captures Trainer messages. The checkpoint is written first, then one atomic index replacement commits both references. Obsolete previously indexed checkpoints are removed afterward. A crash may leave an unreferenced file; it is never selected by the index. Normal operation retains at most two checkpoint files. Full baseline optimizer checkpoints will be much larger than this tiny check.

`resolve_checkpoint(run_dir, which='last')` returns the referenced path. `load_checkpoint(path, model, trainer, loaders, context)` restores training. Use last for continuation; best for later evaluation. `RunLogger(run_dir, loaders, context, resume=False)` plugs into the existing on_epoch_end callback. `write_history(run_dir, trainer.history)` regenerates metric files after resume. The checkpoint history is authoritative if metric writes were interrupted. TensorBoard remains disabled; this phase uses JSON, CSV and text logs.

Run:

```bash
python -m src.training.check_checkpointing --config configs/baseline_debug.yaml
```

The tiny synthetic fixture runs the real Trainer on CPU and the selected accelerator. It compares uninterrupted and resumed training, restored optimizer/RNG/shuffle states, rejects an incompatible context, and checks best/last references and persistent logs. Controlled worsening monitor scores keep epoch 1 as best while epoch 2 becomes last. CPU comparisons are exact; accelerator continuation comparisons allow 1e-5 tolerance. CUDA mixed precision is not tested. The fixture is a test helper, not a replacement captioning architecture.

Verified on 2026-09-23: PHASE 8 CHECKPOINT AND LOGGING CHECK: PASS on CPU and MPS. Both paths passed checkpoint reload, rejection of incompatible context before weight changes, optimizer/counter/history/RNG restoration, uninterrupted-versus-resumed training comparison, and best/last selection and persistent-log checks. Each two-epoch path completed six optimizer updates.

CPU epoch 2: train_loss=2.7415, val_loss=2.8112; MPS epoch 2: train_loss=2.9311, val_loss=2.8104. Each resumed path reproduced its corresponding uninterrupted results. These are synthetic-fixture losses, not baseline caption-quality results. The deliberately worsening check_score explains improved=False even though validation loss decreased.

Report on the user's Mac: `artifacts/checkpoint_checks/20260923T053057_469748Z_0347d8c9/report.json`, with tiny fixture checkpoints and logs in the same run directory. No COCO downloads or baseline training occurred during the Phase 8 synthetic check.

## Phase 9: trained and verified on MPS

New files: `src/training/train_baseline.py` and `src/training/check_training_run.py`. No existing model, Trainer, checkpoint, data or configuration interfaces change.

The baseline uses the verified 1,000 training images (5,008 caption pairs) and 250 validation images (1,250 pairs), a frozen pretrained ResNet50 encoder, batch size 8, AdamW, and three target epochs. Each epoch has 626 training batches and 157 validation batches. Three completed epochs with no skipped updates produce 1,878 optimizer updates. The selected checkpoint minimizes token-weighted validation cross-entropy. Caption quality remains unmeasured until BLEU/CIDEr evaluation.

From the project root:

```bash
python -m src.training.train_baseline --config configs/baseline_debug.yaml --val-images 250 --epochs 3 --run-dir artifacts/runs/baseline_debug_v1
python -m src.training.check_training_run --run-dir artifacts/runs/baseline_debug_v1
```

The new-run directory must not already exist. The training command uses all prepared pairs in each epoch and performs no new data downloads. Pretrained encoder weights are already cached from Phase 6. Actual running time must be measured on the user's Mac; no fixed duration is promised. Baseline checkpoints include optimizer state and are approximately 400 MiB each; allow about 1.5 GB free for latest/best files and a temporary new save.

If interrupted after at least one epoch commits, resume with the same config, source code, data and device:

```bash
python -m src.training.train_baseline --config configs/baseline_debug.yaml --val-images 250 --epochs 3 --resume artifacts/runs/baseline_debug_v1
```

The epoch argument is the total target, not three additional epochs. Work in an incomplete epoch is repeated. If no checkpoint was committed, start in a new run directory. Do not run training and the checker concurrently on the same run.

Run artifacts: `config.json`, `context.json`, `initial_state.json`, a copied `tokenizer/` directory, the Phase 8 checkpoint index/files, metric JSON/CSV and training log, then `run_summary.json`. Context binds the selected manifest, actual selected image bytes, encoded caption pairs, tokenizer, preprocessing, relevant source files and full-epoch batch limits. Resume rejects changes to these fingerprints. The source fingerprint intentionally excludes the post-training checker so checker-only fixes do not invalidate training.

The final checker verifies completed epochs, expected target tokens and updates, metric history, best selection, finite saved weights, unchanged frozen encoder and changed trainable weights. It reconstructs the best model without downloading pretrained weights and repeats validation over all 250 images. Saved/reloaded validation losses must match within relative/absolute tolerance 1e-4; validation creates no gradients. It writes `training_validation.json` on success.

Verified on 2026-09-23: PHASE 9 BASELINE TRAINING CHECK: PASS. The user completed three full epochs on MPS with 1,878 optimizer updates.

| Epoch | Training loss | Validation loss |
| --- | ---: | ---: |
| 1 | 5.0627 | 4.3228 |
| 2 | 3.9842 | 3.9090 |
| 3 | 3.4883 | 3.6852 |

Best epoch: 3. Saved best validation loss: 3.685195; reloaded validation loss: 3.685195 at displayed precision. Configuration/source/data/tokenizer fingerprints, complete epoch/token/update counts, best selection, finite checkpoint weights, unchanged frozen encoder and changed trainable parameters passed. The logged training interval was approximately 6 minutes 47 seconds (11:25:27–11:32:14), including validation and checkpoint saves.

Verified run: `artifacts/runs/baseline_debug_v1`. Verification report: `artifacts/runs/baseline_debug_v1/training_validation.json`. Validation loss improved each epoch; generated-caption quality is reported in Phase 10.

## Phase 10: verified BLEU/CIDEr evaluation

New files only: `requirements-eval.txt`, `src/evaluation/metrics.py`, `src/evaluation/check_metrics.py`, `src/evaluation/evaluate_baseline.py`. Trained weights, checkpoint formats, data, model, Trainer, existing config and existing dependencies are unchanged. Optional evaluation dependency: `pycocoevalcap==1.2` (brings pycocotools); its Stanford PTB tokenizer requires a working Java command. Check `java -version` before the metric check; if unavailable on the Mac, install an available JDK and put its `java` binary on PATH. The small evaluator package downloads once; it does not download COCO images or change the model weights.

Run from the project root with the existing `.venv`:

```bash
python -m pip install -r requirements-eval.txt
java -version
python -m compileall -q src/evaluation
python -m src.evaluation.check_metrics
python -m src.evaluation.evaluate_baseline --run-dir artifacts/runs/baseline_debug_v1 --output-dir artifacts/evaluations/baseline_debug_v1_validation_greedy_v1
```

The metric check runs without the captioning model or COCO images. It verifies Java-backed COCO PTB normalization, corpus BLEU-1..4, CIDEr, per-image ID alignment, empty predictions and invalid input rejection. The metric check passed on the user's Mac with pycocoevalcap 1.2, NumPy 2.4.6, and Java 26.0.1; the authoring workspace had separately checked Java 17.

The evaluator verifies the exact Phase 9 run context and best checkpoint, loads saved tokenizer/model weights locally without downloading a pretrained encoder, and generates one greedy caption for **each of the 250 validation images**. Reference captions remain the original complete COCO texts; no references are passed to generation and no token-length truncation is applied to references. Default maximum new token count is 31, within the trained 32-token BOS/EOS budget. It computes all corpus metrics over the complete 250-image set at once, then saves `predictions.json`, `per_image.json`, `tokenized.json` and `metrics.json` to the new output directory. `predictions.json` is COCO-style `{image_id, caption}` rows. The output directory must be new; do not overwrite an earlier evaluation.

`score_captions(references: dict[int,list[str]], predictions: dict[int,str])` is the shared metric interface for later model comparison; it returns corpus and per-image scores, PTB-normalized text, and scoring protocol. BLEU is raw 0–1 corpus BLEU (orders 1–4), not an average of per-image BLEU. CIDEr is the package's raw score with reference document frequencies computed from these 250 image groups, clipping and length penalty. CIDEr is **not a percentage**. Keep the exact validation image IDs, reference grouping and evaluator protocol fixed when comparing the Transformer later. This is a custom validation subset used for model selection, not an official COCO test-server score; the held-out project test partition has not been downloaded/evaluated.

Verified on 2026-09-23: `PHASE 10 METRIC CHECK: PASS` and `PHASE 10 BASELINE BLEU/CIDEr EVALUATION: PASS`. All 250 validation images were generated and scored against their 1,250 original references using greedy decoding and the Phase 9 best checkpoint (epoch 3).

| Metric | Corpus score (raw evaluator scale) |
| --- | ---: |
| BLEU-1 | 0.589657 |
| BLEU-2 | 0.392498 |
| BLEU-3 | 0.231648 |
| BLEU-4 | 0.129139 |
| CIDEr | 0.398109 |

Diagnostic counts: zero empty captions, 155 distinct captions, 250 EOS endings, zero token-limit hits. The scores above are console-rounded to six decimal places; `metrics.json` preserves full numeric precision. Evaluation artifacts on the user's Mac: `artifacts/evaluations/baseline_debug_v1_validation_greedy_v1/{predictions.json,per_image.json,tokenized.json,metrics.json}`. The scoring success and diagnostics do not guarantee that individual captions are accurate; inspect examples during the qualitative phase. There is no official COCO test-server score.

## Phase 11: standalone baseline inference — verified on MPS

New files: `src/inference/predict.py` and `src/inference/check_inference.py`. Existing `src/inference/__init__.py` stays unchanged. `BaselineCaptioner(run_dir, device='auto').predict(image_path, max_new_tokens=None)` runs greedy decoding on one local image and returns caption, token IDs, EOS flag, chosen device, best epoch and checkpoint hash. It loads the saved tokenizer and verified best checkpoint, checks their run identity and the trained model/input source hashes, and uses the Phase 3 validation transform. This path does not require COCO annotations, Java, scoring libraries, a dataset scan, or a weights download. CPU inference is allowed even when training used MPS; no training resume occurs.

From the project root, using the existing virtual environment and a downloaded validation image:

```bash
python -m compileall -q src/inference/predict.py src/inference/check_inference.py
python -m src.inference.predict --run-dir artifacts/runs/baseline_debug_v1 --image data/raw/coco2017/val2017/000000491757.jpg --output artifacts/runs/baseline_debug_v1/inference_example.json
python -m src.inference.check_inference --run-dir artifacts/runs/baseline_debug_v1 --image data/raw/coco2017/val2017/000000491757.jpg
```

The image ID 491757 appeared in the verified preprocessing preview. If its local relative path differs, pass its actual file path to both commands. The optional output path must be new. The checker runs this same single-image predictor twice, verifies identical results and no gradients, rejects a missing path and an invalid token limit, and writes `artifacts/runs/baseline_debug_v1/inference_validation.json`. Results for an arbitrary image need not match batch-of-eight Phase 10 token IDs exactly due to possible numerical differences in batch processing.

Verified on 2026-09-24: the user's Mac loaded the best checkpoint (epoch 3) on MPS, captioned the prepared validation image 491757 as `a cat is sitting on a bed.`, and saved `artifacts/runs/baseline_debug_v1/inference_example.json`. The repeatability, eval-mode/no-gradient, missing-image and token-limit checks passed, and the checker saved `artifacts/runs/baseline_debug_v1/inference_validation.json`. The earlier `touch src/inference/predict.py` failed because its parent directory was absent at that moment, but the final commands succeeded after the inference files were present. Phase 12 remains pending.

## Phase 12: Person 1 validation and handoff — verified

New files: `src/check_handoff.py` and `docs/PERSON1_HANDOFF.md`.
Run from the project root in the existing environment:

```bash
python -m pip check
python -m compileall -q src/check_handoff.py
python -m src.check_handoff --run-dir artifacts/runs/baseline_debug_v1 --evaluation-dir artifacts/evaluations/baseline_debug_v1_validation_greedy_v1 --output-dir artifacts/handoff/person1_v1
```

The output directory must be new. This check hashes the selected development images,
checks the current training context and tokenizer snapshot, links training/evaluation/
inference to the same best checkpoint, verifies all 250 prediction IDs and original
references, and recomputes corpus/per-image BLEU/CIDEr from saved captions. It also
checks saved tokenization/diagnostics and the passing inference report. It performs
no optimizer updates or caption regeneration. Java must remain available for PTB
tokenization. A changed scoring environment is rejected for explicit review.

On success, `artifacts/handoff/person1_v1/report.json` records results and artifact
hashes, and `environment.freeze.txt` records installed Python packages. The latter
is an environment snapshot, not a cross-platform lockfile. The report distinguishes
successful checking from the user's explicit handoff acknowledgment.

Read `docs/PERSON1_HANDOFF.md` for the transfer inventory, stable interfaces, observed
results, absolute-path portability limitation, and pending held-out test evaluation.
Verified on the user's Mac on 2026-09-24 at 10:26:27: PHASE 12 PERSON 1 HANDOFF CHECK: PASS. The initial attempt failed because docs/PERSON1_HANDOFF.md was missing; after creating it, the complete check passed. Current data/source/tokenizer fingerprints, checkpoint identity, all 250 prediction IDs and references, recomputed BLEU/CIDEr and saved tokenized captions passed. The checker created artifacts/handoff/person1_v1/report.json and environment.freeze.txt. Person 1 is complete and ready for handoff. Person 2 starts only after explicit confirmation of completion and handoff.

## Roadmap and ownership

| Phase | Owner | Work | Status |
| --- | --- | --- | --- |
| 1 | Person 1 | Environment and repository setup | Verified |
| 2 | Person 1 | COCO setup | Verified |
| 3 | Person 1 | Image preprocessing | Verified |
| 4 | Person 1 | Caption/tokenizer processing | Verified |
| 5 | Person 1 | Dataset/DataLoader | Verified |
| 6 | Person 1 | ResNet50 + LSTM | Verified |
| 7 | Person 1 | Reusable training framework | Verified |
| 8 | Person 1 | Checkpoints, resume, logging | Verified on CPU and MPS |
| 9 | Person 1 | Baseline training on limited data | Trained and verified on MPS |
| 10 | Person 1 | BLEU/CIDEr evaluation | Verified on 250 validation images |
| 11 | Person 1 | Standalone baseline inference | Verified on MPS |
| 12 | Person 1 | Validation and handoff | Verified; awaiting explicit handoff acknowledgment |
| 13 | Person 2 | Pretrained ViT encoder | Pending explicit handoff |
| 14 | Person 2 | Visual patch tokens | Pending |
| 15 | Person 2 | Visual projection | Pending |
| 16 | Person 2 | Transformer decoder | Pending |
| 17 | Person 2 | Masked self-attention | Pending |
| 18 | Person 2 | Image-text cross-attention | Pending |
| 19 | Person 2 | Complete multimodal architecture | Pending |
| 20 | Person 2 | Small-subset training | Pending |
| 21 | Person 2 | Architecture fine-tuning | Pending |
| 22 | Person 2 | Final training on agreed limited COCO subset | Pending; replaces full-COCO requirement |
| 23 | Person 2 | Greedy inference | Pending |
| 24 | Person 2 | Beam search | Pending |
| 25 | Person 2 | Final BLEU/CIDEr evaluation | Pending |
| 26 | Person 2 | Attention visualization | Pending |
| 27 | Person 2 | Qualitative/error analysis | Pending |
| 28 | Shared integration | Gradio/Streamlit app | Pending |
| 29 | Shared integration | Final integration/documentation | Pending |

Person 1 must deliver working training, evaluation and inference, trained baseline weights, environment/data instructions, and documented interfaces before handoff. Person 2 reuses the common infrastructure. Both roles involve AI/ML; the ownership split is engineering/baseline versus advanced architecture.

## Artifacts and results policy

Diagnostic reports currently live under artifacts/setup, preprocessing, tokenizer, dataloader and baseline_checks. Phase 7 adds trainer_checks; Phase 8 adds checkpoint_checks with synthetic-fixture checkpoints and logs. The tokenizer is under data/processed/tokenizers. Phase 9 produced and verified a trained baseline checkpoint in artifacts/runs/baseline_debug_v1. Phase 10 measured BLEU-4=0.129139 and raw CIDEr=0.398109 on the custom 250-image validation set. A forward/backward pass or three-step check is not evidence of caption quality.

Update this README after each verified phase and before handoff. Preserve user-added notes. Never mark a phase verified based only on generated code or expected output. Keep failures and limitations explicit; do not fabricate training or evaluation results.
