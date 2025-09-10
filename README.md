Dual-Timepoint Tumor Synthesis in Mammograms

This repository contains an implementation of a GAN + Transformer-based architecture for synthesizing tumors in mammograms using prior and current images.
The model uses:

Dual Transformer Encoders

Variational Latent Space

Differentiable Blending Module

Swin Transformer Discriminator

Project Structure
.
├── dataset.py        # Dataset loader
├── generator.py      # Generator network
├── decoder.py       # Transformer-based decoder
├── encoder.py       # Transformer-based encoder
├── discriminator.py # Swin Transformer discriminator
├── losses.py        # Loss functions
├── logger.py        # Training logger
├── config.py        # Hyperparameters & constants
├── seed.py          # Random seed setup
├── train.py         # Training script
├── test.py          # Testing / inference script
└── README.md        # Project documentation

1. Dataset Preparation

Organize your dataset like this:

data/
  train/
    PATIENT001/
      prior/
        LCC.png
        LMLO.png
        RCC.png
        RMLO.png
      current/
        LCC.png
        LMLO.png
        RCC.png
        RMLO.png
      masks/                 # optional per-view masks
        LCC.png             # include only if available
    PATIENT002/
      ...
  val/
    PATIENT003/ ...
  test/
    PATIENT004/ ...

Notes:

Prior and current mammograms are stored per patient.

Standard mammographic views: LCC, LMLO, RCC, RMLO.

Tumor masks are optional → if missing, tumor loss is skipped automatically.

2. Train the Model

Run the training script:

python3 train.py


This will:

Train the model on data/train

Save logs to logs/ (if enabled)

Save generated samples periodically during training

3. Test / Inference

To evaluate the model on the test split:

python3 test.py --split test


Loads the latest trained model.

Generates synthesized tumors.

Saves outputs into the eval_out/ directory.

4. Outputs

After running test.py, you’ll find results in:

eval_out/
  sample_00000.png
  sample_00001.png
  ...
  metrics.json


Each sample_xxxxx.png contains a visualization panel:

[ Prior | Current | Synthetic | Generated Tumor | Tumor Probability | Blending Mask ]

5. Quick Commands
Task	Command
Train model	python3 train.py
Test model	python3 test.py --split test
View results	ls eval_out/