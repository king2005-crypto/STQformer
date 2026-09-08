# STQFormer

### Spatio-Temporal Quaternion Transformer for Video Frame Denoising

**Accepted at the 2026 IEEE International Conference on Image Processing (ICIP 2026).**


STQFormer is an unsupervised framework for endoscopic video denoising, designed to preserve spatial details, color relationships, and temporal consistency without paired clean–noisy training data. It combines a shared frame-wise Swin encoder, quaternion-inspired spatial processing, temporal self-attention, and a Restormer-style decoder to restore the center frame of a five-frame video sequence. A two-stage training strategy progressively introduces temporal modeling after spatial learning.

This repository provides example code, training configurations, data preparation utilities, and evaluation scripts to help researchers run STQFormer and conduct further experiments.
