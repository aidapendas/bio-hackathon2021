# Antibody Heavy-Light Chain Pairing — Bio-Hackathon 2021

**Computational Biology Hackathon · April 2021 · Team AminoBugs**

---

## The Problem

Antibodies are proteins used by the immune system to identify and neutralise threats — and they are one of the most important drug modalities in modern medicine. Each antibody is made up of two components: a **heavy chain** and a **light chain**. These chains are naturally paired in the body, but when producing therapeutic antibodies at scale, correctly identifying which heavy chain pairs with which light chain is a non-trivial prediction problem.

This hackathon challenged us to build a machine learning model to **predict correct heavy-light chain antibody pairing** from amino acid sequences alone — a binary classification task with direct relevance to antibody drug design.

---

## Approach

Antibody sequences were encoded using **one-hot encoding** of amino acids, with heavy chains padded to length 150 and light chains to length 130, producing high-dimensional sequence representations. We explored two model architectures:

### 1. Linear / Dense Model (`main.py`, `Linear_optimization.ipynb`)
A multi-input dense neural network taking separate heavy and light chain encodings, concatenating them, and passing through dense layers to a binary classification output. We used **AutoKeras** for automated hyperparameter search across multiple trials.

### 2. 1D Convolutional Model (`main_CNN_op.py`, `Conv_optimization.ipynb`)
A 1D CNN architecture applied to the sequence representations before concatenation and classification — designed to capture local sequence motifs relevant to chain pairing.

Both models were optimised across multiple trials (`Trial1`, `Trial2`, `Trial3`) with early stopping on validation loss.

---

## Repository Structure

```
├── main.py                    # Dense/Linear model with AutoKeras optimisation
├── main_CNN_op.py             # 1D CNN model variant
├── main_Linear_op.py          # Linear model variant
├── Conv_optimization.ipynb    # CNN hyperparameter search and results
├── Linear_optimization.ipynb  # Linear model hyperparameter search and results
├── predict.py                 # Inference script
├── predictlocal.py            # Local inference script
├── Hackaton_AK_Trial1/        # Saved best model — Trial 1
├── Hackaton_AK_Trial2/        # Saved best model — Trial 2
├── Hackaton_AK_Trial3/        # Saved best model — Trial 3
├── Heavy-Light-model2/        # Final saved model
├── trainset_shuffled/         # Training data
├── testset/                   # Test data
├── HighLow_dataset.csv        # Dataset
└── Demo_code/                 # Demo scripts
```

---

## Technical Stack

- **Python** — core language
- **TensorFlow / Keras** — model architecture and training
- **AutoKeras** — automated neural architecture search and hyperparameter optimisation
- **NumPy / Pandas** — data processing and sequence encoding
- **scikit-learn** — train/test splitting

---

## Biological Context

Antibodies are a cornerstone of modern drug discovery — from cancer immunotherapy to autoimmune disease treatments. Correctly predicting heavy-light chain pairing is relevant to:
- **Antibody engineering** — designing therapeutics with desired binding properties
- **Drug discovery pipelines** — screening and validating antibody candidates at scale
- **Computational immunology** — understanding natural immune responses

This work sits at the intersection of sequence biology and deep learning, an area now central to platforms like AlphaFold 3, which extends structure prediction to full molecular complexes including antibody-antigen interactions.

---

## Award

🏆 **Mentor's Choice Award** — selected by the hackathon mentor

## Team

Developed collaboratively by the AminoBugs team during the 2021 Computational Biology Hackathon.

