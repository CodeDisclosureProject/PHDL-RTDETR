# PHDL-RTDETR: Physics-guided Regularization for Injection Molded Defect Detection

## Overview
PHDL-RTDETR is an innovative defect detection model designed for industrial injection-molded part inspections. It integrates physics-guided regularization constraints with advanced multi-scale feature fusion techniques to enhance detection accuracy, robustness, and computational efficiency.

## Key Features
- **Physics-guided Network (PhyGuideNet)**: Integrates bias normalization and Laplacian smoothing diffusion to enhance feature consistency, suppress noise, and improve small target detection.
- **Hybrid Attention Gated Convolution (HAGConv)**: A novel module combining spatial, channel, and edge-aware attention mechanisms to improve high-frequency feature extraction and defect localization.
- **Detail Enhance Attention CARAFE (DEA-CARAFE)**: A content-aware upsampling strategy that preserves fine-grained defect details, strengthens boundary information, and mitigates information loss during feature upscaling.
- **Learnable Weighted Fusion (LW-Fusion)**: A dynamic feature fusion module that adaptively adjusts feature weights, optimizing the balance between defect detection accuracy and computational efficiency.

## Architecture
PHDL-RTDETR extends the RT-DETR architecture with physics-guided constraints to enhance detection performance. The framework includes:

1. **PhyGuideNet**: A physics-guided mechanism that stabilizes feature extraction and enhances defect boundary clarity.
2. **HAGConv**: An advanced convolution module that integrates hybrid attention mechanisms for improved feature extraction.
3. **DEA-CARAFE**: Enhanced upsampling module focusing on preserving defect detail and boundary information.
4. **LW-Fusion**: Adaptive fusion module for optimal feature integration.

## Performance
PHDL-RTDETR demonstrates significant improvements over baseline detectors:
- **Precision**: 90.1%
- **mAP**: 88.4%
- **FPS**: 75
- **Computational Efficiency**: 54% reduction in FLOPs compared to baseline detectors

The model has been rigorously evaluated on multiple datasets:
- Custom injection-molded parts defect dataset
- PCB defect detection dataset
- Fabric defect detection dataset
- Steel surface defect detection dataset

PHDL-RTDETR consistently outperforms state-of-the-art frameworks such as Faster R-CNN, YOLOv8, and RT-DETR, especially in small target detection and scenarios with complex backgrounds.

## Applications
- Industrial quality control for injection-molded parts
- Automated defect inspection systems
- Real-time manufacturing quality assurance

## Advantages
- Superior detection of small-scale defects
- Enhanced robustness against noise and complex backgrounds
- Optimal balance between computational efficiency and detection accuracy
- Strong generalization capabilities across different defect detection scenarios

## Repository Structure
- `Ours/`: Core implementation of the PHDL-RTDETR components
  - `PhyGuideNet.py`: Physics-guided Network module
  - `HAGConv.py`: Hybrid attention gated convolution module
  - `DEA-CARAFE.py`: Detail enhance attention content-aware upsampling
  - `LW-Fusion.py`: Learnable weighted fusion module
- `models/`: Model definitions and backbone architectures
- `utils/`: Utility functions for data processing and evaluation
- `engine/`: Training and inference engine
- `data/`: Data loading and preprocessing
- `cfg/`: Configuration files

