# Hate Speech Detection using BERT

This repository contains an implementation of a hate speech detection model using BERT (Bidirectional Encoder Representations from Transformers). The model classifies text into three categories: `Neutral`, `Offensive`, or `Hate Speech`.

## Overview

The project leverages BERT's powerful language understanding capabilities to detect and classify hate speech in text data. Key features include:

- Fine-tuned BERT model for hate speech classification
- Support for multi-class classification (Neutral, Offensive, Hate Speech)
- Easy-to-use inference pipeline for testing new text inputs
- Comprehensive evaluation metrics
- Detailed training and validation procedures

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/hate-speech-detection-bert.git
cd hate-speech-detection-bert
```

If you prefer manual installation, install the core dependencies:
```bash
pip install torch transformers scikit-learn pandas
```

## Usage

### Data Preparation

Place your dataset in CSV format with the following columns:
- `text`: The input text to be classified
- `label`: The corresponding label (Neutral, Offensive, or Hate Speech)

### Training

To train the model on your dataset:

```bash
python train.py
```

The script will:
- Load and preprocess the dataset
- Fine-tune the BERT model
- Save the trained model
- Output training metrics and validation results

### Testing

To test the trained model on new text:

```bash
python test.py
```

Example usage in Python:

```python
from model import test_model

text = "Your text here"
prediction = test_model(model, text, tokenizer, device)
print(f"Prediction: {prediction}")
```

## Model Architecture

The project utilizes the following components:

- Base Model: `bert-base-uncased` from Hugging Face Transformers
- Classification Head: Linear layer for 3-class classification
- Training Strategy: Fine-tuning approach with frozen BERT layers

### Training Parameters

- Optimizer: AdamW
- Loss Function: Cross-Entropy Loss
- Learning Rate: 2e-5
- Batch Size: 32
- Epochs: 3

## Model Performance

The model is evaluated using the following metrics:
- Precision
- Recall
- F1-Score
- Accuracy

Detailed performance metrics are generated during training and testing phases.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Acknowledgments

- Thanks to the Hugging Face team for their Transformers library
- BERT paper authors for their groundbreaking work
- Contributors to the hate speech detection research community
