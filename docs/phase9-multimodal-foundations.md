# Phase 9 — Multimodal AI foundations (CPU)

Phase 9 adds a tiny, fully differentiable **late-fusion** image/audio/text
classifier to EveryO. Feature extraction is deterministic and does not rely on
GPU acceleration, pretrained models, cloud inference or external datasets.

| Step | Implementation | Acceptance criterion |
| --- | --- | --- |
| 9.1 | Image input normalization and shape validation | Images in `[0,1]` |
| 9.2 | `image_patch_features` | RGB or grayscale 2x2 patch descriptors |
| 9.3 | `audio_frame_features` | Four-frame RMS / zero crossings |
| 9.4 | `text_hash_features` | Stable UTF-8 bag-of-bytes descriptors |
| 9.5 | `MultimodalBatch`, `prepare_multimodal_batch` | Aligned batches, integer labels |
| 9.6 | `TinyMultimodalNet` | Trainable separate encoders and late fusion |
| 9.7 | `contrastive_alignment_loss` | Symmetric differentiable InfoNCE |
| 9.8 | `examples/cpu_multimodal_demo.py` | Train, evaluate, `.evo` save/load |

Run:

```bash
python -m pytest -q tests/test_phase9_multimodal.py
python examples/cpu_multimodal_demo.py
```

This is a **feature-level foundation**, not a vision transformer, speech
recognizer, speech synthesizer, or large multimodal generative model. Images
are dense arrays in channel-last order, audio is normalized mono waveform,
and text features are hashed UTF-8 byte frequencies. No sophisticated
pretraining, image resizing, resampling, sample-rate conversion, or speech
transcription is claimed. Evaluate on held-out data before real deployment;
the tiny example reports training-set accuracy only.
