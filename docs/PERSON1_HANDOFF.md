# Person 1 handoff

Status: Phases 1–11 verified on the user's Mac.
Phase 12 awaits successful completion of the final handoff check.
Person 2 starts only after explicit confirmation of completion and handoff.

## Delivered baseline

- Dataset: COCO 2017 limited development subset.
- Training: 1,000 images and 5,008 caption pairs.
- Validation: 250 images and 1,250 reference captions.
- Architecture: frozen pretrained ResNet50 encoder with a trainable LSTM decoder.
- Training: three epochs, batch size 8, and 1,878 optimizer updates.
- Best checkpoint: epoch 3.
- Best validation loss: 3.685195 at displayed precision.
- Validation BLEU-1: 0.589657.
- Validation BLEU-2: 0.392498.
- Validation BLEU-3: 0.231648.
- Validation BLEU-4: 0.129139.
- Raw CIDEr: 0.398109; this is not a percentage.
- MPS inference for image 491757: "a cat is sitting on a bed."

These scores use the custom 250-image validation subset.
They are not official COCO test-server results.

## Files to hand over

Preserve the project source, configs, requirements files, README and docs.

Also include:

- data/metadata/coco2017/: canonical splits and development manifest.
- Selected training and validation images under data/raw/coco2017/.
- Existing COCO annotation files.
- data/processed/: tokenizer and processed artifacts.
- The complete artifacts/runs/baseline_debug_v1/ directory.
- Every checkpoint referenced by that run's checkpoints/index.json.
- artifacts/evaluations/baseline_debug_v1_validation_greedy_v1/.
- Earlier phase validation reports.
- artifacts/handoff/person1_v1/ after the final check succeeds.

Recreate the Python environment rather than copying .venv.
Use requirements files and consult environment.freeze.txt for observed versions.
The freeze file is an environment snapshot, not a portable lockfile.
COCO PTB tokenization requires Java.

## Interfaces to preserve

- Images: RGB, deterministic 224x224 square resize and ImageNet normalization.
- Tokenizer: bert-base-uncased, vocabulary size 30,522.
- Special tokens: BOS=101, EOS=102, PAD=0.
- Maximum caption length: 32 tokens including BOS and EOS.
- Caption batches contain images, input_ids, attention_mask, target_ids and image_ids.
- Images have shape [B,3,H,W]; caption tensors have shape [B,T].
- attention_mask is boolean; True means a valid token.
- Ignored target padding is -100.
- The collator performs the only caption shift; T is at most 31.
- Generation batches contain images, image_ids and grouped original references.
- Model forward returns logits [B,T,V].
- Model generate returns sequences [B,L], excluding BOS and retaining EOS
  and padding after EOS. Baseline attention is None.
- Current baseline generation uses greedy decoding.
- Trainer owns optimization, scheduling and early stopping.
- Epoch-end callbacks handle checkpoint and log persistence.
- Trainer epochs means total target epochs, not additional epochs.
- Version 1 checkpoints support completed-epoch resume with saved training,
  RNG and loader states.
- score_captions(references, predictions) accepts dictionaries keyed by image ID.
- BaselineCaptioner(run_dir, device="auto").predict(image_path) returns
  caption text, generated token IDs and checkpoint metadata.

Changes to established contracts require an explicit
ARCHITECTURE CHANGE REQUIRED proposal and approval.

## Portability

Saved configurations and contexts contain absolute paths and source hashes.
The current workflow is verified at the original Mac project location.

Moving to another computer or directory may require a reviewed migration.
Do not silently edit checkpoint contracts, hashes or saved paths.
Training resume has stricter device requirements than inference.

## Phase 12 validation

The final checker:

- Verifies current source, selected-image and tokenizer fingerprints.
- Matches training, evaluation and inference to the same best checkpoint.
- Verifies all 250 validation IDs, original references and generated tokens.
- Recomputes corpus and per-image BLEU/CIDEr from saved predictions.
- Checks saved tokenization, evaluation diagnostics and inference results.
- Records artifact hashes and installed Python package versions.

The checker does not retrain the model or regenerate captions.
Earlier successful phase logs provide evidence for their runtime checks.
A passing handoff report does not itself confirm that handoff was acknowledged.

## Limitations and next work

The baseline uses a small dataset and three training epochs.
Individual generated captions can still be inaccurate.

Keep validation IDs, reference grouping and scoring protocol fixed for
future model comparisons.

A limited held-out test set remains to be prepared and evaluated later.
Do not use held-out test data for tuning.
Full-COCO training is not required.

After Phase 12 passes, preserve the baseline artifacts and explicitly confirm
Person 1 completion and handoff.

The next possible phase is Phase 13: pretrained ViT encoder.
No Person 2 implementation is included in this handoff.
