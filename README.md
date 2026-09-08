# STQFormer

### Spatio-Temporal Quaternion Transformer for Video Frame Denoising

**Accepted at the 2026 IEEE International Conference on Image Processing (ICIP 2026).**


STQFormer is an unsupervised framework for endoscopic video denoising, designed to preserve spatial details, color relationships, and temporal consistency without paired clean–noisy training data. It combines a shared frame-wise Swin encoder, quaternion-inspired spatial processing, temporal self-attention, and a Restormer-style decoder to restore the center frame of a five-frame video sequence. A two-stage training strategy progressively introduces temporal modeling after spatial learning.

This repository provides example code, training configurations, data preparation utilities, and evaluation scripts to help researchers run STQFormer and conduct further experiments.

## IEEE Copyright Notice

© 2026 IEEE. Personal use of this material is permitted. Permission from IEEE must be obtained for all other uses, in any current or future media, including reprinting/republishing this material for advertising or promotional purposes, creating new collective works, for resale or redistribution to servers or lists, or reuse of any copyrighted component of this work in other works.

This author-provided manuscript includes the copyright notice on its first page. The DOI and final publication details will be added when available.

Posting and reuse are governed by the [IEEE paper posting policy](https://conferences.ieeeauthorcenter.ieee.org/get-published/post-your-paper/). The copyright notice does not itself grant permission to redistribute the manuscript on additional platforms. The manuscript is separate from the software in this repository.
