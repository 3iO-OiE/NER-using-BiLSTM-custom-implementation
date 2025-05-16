# Medical Named Entity Recognition (NER) using BiLSTM

This project implements a **Named Entity Recognition (NER)** system for **medical texts** using a **BiLSTM** model. The system is trained on annotated case reports formatted in the **BRAT standoff annotation format**.

Developed in Kaggle.

## 📊 Objective

Extract structured information (entities like Age, History, Sex, Symptoms, Clinical Events, etc.) from unstructured medical case descriptions.

## 🔎 Example

**Text:**

```
CASE: A 28-year-old previously healthy man presented with a 6-week history of palpitations.
```

**Annotations (BRAT format):**

```
T1  Age 8 19  28-year-old
T2  History 20 38  previously healthy
T3  Sex 39 42  man
T4  Clinical_event 43 52  presented
T5  Sign_symptom 78 90  palpitations
... (see dataset for more)
```

## 🧬 Model

We use a **Bi-directional LSTM (BiLSTM)** architecture with PyTorch for sequence labeling.

### Key components:

* Tokenizer and Vocabulary builder
* Entity label encoder
* BiLSTM model (with optional CRF layer)
* Custom Dataset class
* Training loop with early stopping
* Evaluation (F1-score, confusion matrix)

## ✨ Quick Inference with Trained Model

We provide a pre-trained model file: `bilstm_ner_model.pth`.
You can run inference using your own medical text files.

### Note on Execution

> This project is developed entirely in a **Kaggle notebook**, not as separate Python scripts. The training, evaluation, and model saving/loading are all done interactively inside the notebook.

## Custom implementation using PyTorch

This project was implemented from scratch using only PyTorch and Python:

| Component              | Built From Scratch? | Details                                                      |
| ---------------------- | ------------------- | ------------------------------------------------------------ |
| **Data Preprocessing** | ✅                   | Custom BRAT-format parser, tokenizer, and label encoder      |
| **Vocabulary Builder** | ✅                   | Manually implemented vocabulary class for words and tags     |
| **NER Dataset Loader** | ✅                   | Custom PyTorch `Dataset` class handling tokens and labels    |
| **Model (BiLSTM)**     | ✅                   | BiLSTM implemented directly using PyTorch layers             |
| **Training Loop**      | ✅                   | Manual batching, loss computation, backprop, optimizer steps |
| **Evaluation**         | ✅                   | Manual metrics calculation using `sklearn`                   |

No external NER or deep learning libraries like spaCy, HuggingFace, or Flair were used.

## 🗂️ Project Structure

```
medical-ner-bilstm/
├── data/              # Input text and annotation files
    ├── case01.txt
    ├── case01.ann
    ├── case02.txt
    ├── case02.ann
    ├── ...          
├── bilstm_ner_model.pth          # Trained model checkpoint
├── ner-project.ipynb  # Main notebook
└── README.md
```

## 📖 Technologies Used

* Python 3.10+
* PyTorch
* NumPy / Pandas
* scikit-learn
* tqdm / seaborn / matplotlib

## Status

Working prototype with support for BRAT-style annotation parsing, vocabulary building, BiLSTM-based training, and evaluation.

## Contributing

Feel free to fork and contribute! Ideas for improvement:

* Add CRF layer for better performance
* Integrate transformers (like BioBERT)
* Support IOB tagging scheme for broader NLP compatibility



