# CPPED Project Overview

## Project Purpose

CPPED is a computer vision research project focused on detecting construction personal protective equipment (PPE) in images and video frames. The goal is to train, evaluate, and compare multiple object detection models so the final paper can identify which architecture performs best for construction-site safety monitoring.

The project is organized as a reproducible research workspace: raw data, processed data, annotations, model implementations, experiments, notebooks, and paper assets are separated into clear directories. This makes it easier to expand the study with additional datasets, models, metrics, and visual results.

## Goal

The planned paper will compare object detection architectures on diferent datasets for construction PPE detection. The main comparison targets are:

- Detection accuracy across PPE categories.
- Ability to detect both compliant and non-compliant cases.
- Robustness to CCTV-style image quality, perspective, blur, and visual noise.
- Training and inference practicality for future real-world deployment.
- Consistency across public datasets and project-specific test data.

## Repository Structure

```text
CPPED/
├── configs/              # Central configuration files for training and experiments
├── data/                 # Raw, processed, annotated, and public datasets
├── experiments/          # Misc experiments
├── notebooks/            # Exploratory analysis, training notes, and result review
├── paper/                # Paper draft and figures for publication
├── scripts/              # Utility scripts for project workflows
├── src/                  # Source code for data processing, models, and utilities
├── tests/                # Future automated tests and validation checks
├── metrics/              # Future metrics for the models
├── comparisons/          # Comparisons for trained models 
├── requirements.txt      # Python dependency list
```

## Data Organization

The `data/` directory separates datasets by stage and origin:

```text
data/
├── raw/                  # Original videos and extracted raw material
├── processed/            # Cleaned or transformed frames   ready for annotation/training
├── annotations/          # Complete datasets (Not public)
└── public_datasets/      # External PPE datasets used for benchmarking
```

Current data assets include:

- `data/raw/raw_test/videos/`: raw CCTV and point-of-view construction videos.
- `data/processed/processed_test/`: 770 processed image frames.
- `data/annotations/test_dataset_0/`: annotation files and class definitions.
- `data/public_datasets/`: public datasets from CHVG, Kaggle/Roboflow, and Ultralytics.

The project currently tracks PPE-related classes such as:

The core classes tracked:

- `helmet`
- `vest`
- `gloves`
- `no_helmet`
- `no_vest`
- `no_gloves`

Less priority classes for future tracking
- `work_boots`
- `goggles`
- `no_work_boots`
- `no_goggles`


This class design supports both positive PPE detection and safety-violation detection.

## Data Processing Pipeline

The preprocessing code is located in `src/data/preprocess/`.

### Frame Extraction

`extract_frames.py` extracts useful frames from construction videos. It uses a YOLO person detector to keep frames that contain people, then applies structural similarity filtering to reduce near-duplicate frames. This helps create a cleaner dataset with less repeated visual information.

Main steps:

1. Load construction video files.
2. Sample frames evenly across each video.
3. Keep only frames where a person is detected.
4. Remove visually similar duplicate frames using SSIM.
5. Save selected frames for annotation or later processing.

### CCTV Simulation

`cctv_effect.py` applies transformations that simulate lower-quality surveillance footage. This includes zoom-out framing, fisheye distortion, blur, haze/dust, and vignette effects.

This is useful for testing whether models remain reliable under realistic construction-site camera conditions.

## Model Structure

Model code is stored in `src/models/`.

```text
src/models/
├── ultralytics_train.py  # Shared Ultralytics YOLO training entry point
└── yolo/
    ├── v8/               # YOLOv8-specific training and evaluation space
    └── v11/              # YOLOv11-specific training and evaluation space
└── some_other_model_or_architecture/

```

The current structure is designed for side-by-side model comparison. Each model architecture can have its own training and evaluation scripts while still following the same overall project layout.

The existing Ultralytics training entry point supports configurable training parameters such as:

- model weights
- dataset YAML path
- number of epochs
- image size
- batch size
- learning rate
- optimizer
- device
- experiment output directory

## Training datasets

The current training datasets that are gonna be used are the public datasets from kaggle, ultralytics and an independent research paper(CHVG). The kaggle one passes the eye test the best for now.


## Notebooks

The `notebooks/` directory is organized around the research workflow:

These notebooks provide a convenient space for analysis and paper-ready figures without mixing exploratory work into the production source code.

## Paper Workspace

The `paper/` directory contains the publication materials:

```text
paper/
├── paper.tex             # LaTeX manuscript draft
└── figures/              # Figures, plots, diagrams, and result visuals
```

This keeps the research write-up connected to the codebase, datasets, and experiments that support it.

## Planned Model Comparison

The project is prepared to compare multiple detector families and versions. The current structure explicitly includes YOLOv8 and YOLOv11 spaces, and it can be extended with additional architectures later.

Some comparison dimensions worth considering:

- mAP50 and mAP50-95.
- Precision and recall per class.
- False positives for PPE items.
- False negatives for missing PPE violations.
- Inference speed.
- Model size.
- Performance on clean public datasets versus CCTV-like processed data.
- Qualitative inspection of difficult cases.


## Summary

CPPED is structured as a research-oriented object detection project for construction PPE monitoring. It already separates data preparation, model development, experiment tracking, notebooks, and paper writing into dedicated areas. This organization supports the main research objective: training and comparing multiple object detection architectures to determine the most effective model for detecting PPE compliance and safety violations in construction environments.

## TODO from now

Next steps from this point (in order would be):

- Configure the training parameters for each yolo model and run training on public datasets
- Using the self labeled test dataset to get the metrics that are gonna be the benchmark to beat `the benchmark are the ultralytics yolo models and weights that are gonna be trained on the datasets(no pipeline)`
- Run experiments to beat the benchmark
- Expand the training data at some point(would be nice)
- Compare all the metrics and draw a conclussion