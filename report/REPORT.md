# Marine Species Classification — Final Report

**Generated on:** 2025-12-22 10:19:01
**Device:** cuda
**Number of classes:** 23
**Dataset:** Sea Animals (23 classes)

---

## Executive Summary

**🏆 Best Performing Model: ga_best**
**📊 Test Accuracy: 78.66%**

This report documents a comprehensive comparison of neural network architectures and hyperparameter optimization strategies for marine species classification.

---

## 1. Introduction

### Objective
Develop and compare multiple MLP-based classifiers for marine species image classification using transfer learning with ResNet18 as a feature extractor.

### Approach
- **Feature Extraction:** ResNet18 (pretrained on ImageNet) generates 512-D embeddings
- **Classification Head:** Custom MLP with configurable architecture
- **Optimization:** Baseline, Random Search, Genetic Algorithm, and From-Scratch implementations

---

## 2. Dataset

### Source & Structure
- **Classes:** 23 marine species categories
- **Organization:** Folder-based structure (one folder per class)
- **Preprocessing:**
  - Resize to 224×224 pixels
  - Normalize with ImageNet statistics (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

### Data Splits
Stratified splits to preserve class distribution:
- **Training:** 70%
- **Validation:** 15%
- **Test:** 15%

---

## 3. Feature Extraction

### ResNet18 Backbone
- **Pretrained Weights:** ImageNet
- **Output:** 512-dimensional feature vectors from penultimate layer
- **Mode:** Frozen backbone (inference only, no fine-tuning)

---

## 4. Model Architectures

### 4.1 Baseline PyTorch MLP
- **Architecture:** [512] → [256, 128] → [23]
- **Activation:** ReLU
- **Optimizer:** Adam (lr=1e-3)
- **Batch Size:** 64
- **Epochs:** 12

### 4.2 Random Search Optimized
- **Search Space:**
  - Hidden layers: 1-5
  - Neurons per layer: 32-512
  - Activation: {ReLU, Tanh, Sigmoid}
  - Learning rate: log-uniform [1e-5, 1e-1]
  - Batch size: {16, 32, 64, 128}
  - Optimizer: {SGD, Adam, RMSProp, Adagrad}
  - Epochs: 3-20
- **Best Configuration:**
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

### 4.3 Genetic Algorithm Optimized
- **Chromosome Encoding:** Same hyperparameters as random search
- **Selection:** Tournament (k=3)
- **Crossover:** Uniform
- **Mutation Rate:** 20%
- **Elitism:** Best individual preserved each generation
- **Best Chromosome:**
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

### 4.4 From-Scratch NumPy/CuPy MLP
- **Implementation:** Manual forward/backward propagation
- **Architecture:** [512] → [256, 128] → [23] (matching baseline)
- **Activation:** ReLU
- **Optimizer:** Gradient Descent (lr=0.1)
- **Framework:** NumPy/CuPy for GPU acceleration

---

## 5. Training Configuration

### Loss Function
- **PyTorch Models:** CrossEntropyLoss (combines LogSoftmax + NLLLoss)
- **From-Scratch:** Multi-class cross-entropy

### Hardware
- **Device:** cuda

---

## 6. Results

### 6.1 Test Accuracy Comparison

| Rank | Model | Test Accuracy |
|------|-------|---------------|
| 3 | ga_best ⭐ | 78.66% |
| 2 | random_search_best | 78.56% |
| 1 | baseline | 77.54% |
| 4 | from_scratch_mlp | 13.86% |

**Winner:** ga_best with 78.66% test accuracy

Detailed comparison saved to: `accuracy_comparison.csv`

### 6.2 Per-Model Performance
- Baseline Test Acc: 77.54%
- Random Search Best Test Acc: 78.56%
- GA Best Test Acc: 78.66%
- From-Scratch MLP Test Acc: 13.86%

---

## 7. Classification Reports

Detailed per-class precision, recall, and F1-scores for each model:

### baseline
See: [`classification_report_baseline.txt`](classification_report_baseline.txt)

### random_search_best
See: [`classification_report_random_search_best.txt`](classification_report_random_search_best.txt)

### ga_best ⭐
See: [`classification_report_ga_best.txt`](classification_report_ga_best.txt)

---

## 8. Confusion Matrices

Visual analysis of model predictions vs ground truth:

### baseline
![Confusion Matrix for baseline](figures\confusion_matrix_baseline.png)

### random_search_best
![Confusion Matrix for random_search_best](figures\confusion_matrix_random_search_best.png)

### ga_best ⭐
![Confusion Matrix for ga_best](figures\confusion_matrix_ga_best.png)

---

## 9. Learning Curves

### 9.1 Individual Model Training Curves

Loss and accuracy progression for each model:

#### baseline
![Training curves for baseline](figures\training_curves_baseline.png)

#### random_search_best
![Training curves for random_search_best](figures\training_curves_random_search_best.png)

#### ga_best ⭐
![Training curves for ga_best](figures\training_curves_ga_best.png)

### 9.2 Combined Accuracy Comparison

All models compared on the same axes:

![Accuracy Curves Comparison](figures\accuracy_curves_comparison.png)

---

## 10. Discussion

### Key Findings
1. **Best Model:** ga_best achieved 78.66% test accuracy
2. **Optimization Impact:** Hyperparameter tuning (Random Search/GA) vs baseline comparison
3. **Overfitting Analysis:** Train vs validation accuracy divergence patterns
4. **Misclassification Trends:** Confusion matrix reveals challenging class pairs

### Observations
- Transfer learning with ResNet18 provides strong feature representations
- Small MLP heads (2-3 layers) sufficient for 512-D embeddings
- Learning rate and batch size critical hyperparameters

---

## 11. Conclusion

### Summary
The ga_best configuration demonstrated superior performance with 78.66% test accuracy, validating the effectiveness of automated hyperparameter optimization.

### Future Work
- **Data Augmentation:** Rotation, flip, color jittering
- **Regularization:** Dropout, L2 penalty
- **Learning Rate Scheduling:** Cosine annealing, warm restarts
- **Ensemble Methods:** Model averaging, stacking
- **Fine-tuning:** Unfreeze ResNet18 layers for end-to-end training

---

## Appendix

### File Structure
```
report/
├── REPORT.md                          # This file
├── accuracy_comparison.csv             # Accuracy table
├── classification_report_*.txt         # Per-model reports
└── figures/
    ├── confusion_matrix_*.png          # Confusion matrices
    ├── training_curves_*.png           # Individual training curves
    └── accuracy_curves_comparison.png  # Combined comparison
```

### Tools & Libraries
- **PyTorch:** Deep learning framework
- **torchvision:** Pretrained models
- **NumPy/CuPy:** Numerical computing
- **scikit-learn:** Metrics and evaluation
- **Matplotlib:** Visualization

---

*Report generated automatically on 2025-12-22 10:19:01*