# Marine Species Classification — Report

Generated on: 2025-12-22 09:20:47
Device: cuda
Number of classes: 23

## Introduction
This report documents the full pipeline for marine species classification using a pretrained ResNet18 as a feature extractor and a custom MLP head.

## Dataset
Images are organized by class folders. Features are 512-D vectors extracted from ResNet18 (ImageNet weights).
Train/validation/test splits preserve class distribution.

## Preprocessing & Feature Extraction
- Resize to 224x224, normalize with ImageNet stats.
- Extract 512-D embeddings from the penultimate layer of ResNet18.

## Model Architecture
Custom MLP (`MarineModel`) stacked on 512-D inputs with configurable hidden layers and activations; final linear outputs logits for CrossEntropyLoss.

## Hyperparameter Optimization
### Random Search
- Search space over hidden layers, activation, learning rate (log-scale), batch size, optimizer, epochs.
- Best params (validation):
```json
{
  "hidden_layers": [
    436,
    372,
    392,
    374
  ],
  "activation": "tanh",
  "learning_rate": 4.114074596244534e-05,
  "batch_size": 64,
  "optimizer": "Adam",
  "epochs": 20
}
```
- Best validation result: 76.84391080617496

### Genetic Algorithm
- Chromosome encodes layer count, neurons per layer, activation, learning rate (log-scale), batch size, optimizer, epochs.
- Best chromosome:
```json
{
  "hidden_layers": 4,
  "neurons": 435,
  "activation": "tanh",
  "learning_rate": 9.379880878313753e-05,
  "batch_size": 64,
  "optimizer": "RMSProp",
  "epochs": 14
}
```
- Best GA validation (re-evaluated): 78.44482561463694

## Training Setup
- Loss: CrossEntropyLoss
- Optimizers explored: SGD, Adam, RMSProp, Adagrad
- Epochs vary per run (see params)
- Batch sizes as per configuration

## Results
- Baseline Test Acc: 77.54%
- Random Search Best Test Acc: 78.56%
- GA Best Test Acc: 78.66%

Accuracy comparison (CSV): `accuracy_comparison.csv`.

## Classification Reports
### baseline
See: `classification_report_baseline.txt`
### random_search_best
See: `classification_report_random_search_best.txt`
### ga_best
See: `classification_report_ga_best.txt`

## Confusion Matrices
### baseline
![Confusion Matrix for baseline](figures\confusion_matrix_baseline.png)
### random_search_best
![Confusion Matrix for random_search_best](figures\confusion_matrix_random_search_best.png)
### ga_best
![Confusion Matrix for ga_best](figures\confusion_matrix_ga_best.png)

## Learning Curves
![Accuracy Curves](figures\accuracy_curves.png)

## Discussion
Summarize performance differences, overfitting signals (train vs val), and misclassification trends from confusion matrices.

## Conclusion
Present the best-performing configuration and insights for future improvements (data augmentation, deeper heads, scheduler, regularization).