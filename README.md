# Multilingual Abuse Detection (Code-Mixed Languages)

## Overview

This project focuses on detecting abusive content in **code-mixed languages** like Tamil-English, Hindi-English, Kannada-English, and Malayalam-English.

In real-world social media, people mix languages, use slang, and write informally. Because of this, normal NLP models struggle. This project tries to handle that by using a multilingual transformer model.

The system classifies text as:

* Abusive
* Not Abusive

It also supports **speech input**, where audio is first converted to text and then classified.

---

## Tech Stack

* Python
* PyTorch
* XLM-RoBERTa
* Sarvam AI API (Speech-to-Text)

---

## How the system works

1. Input is given (text or speech)
2. If speech → converted to text using **Sarvam AI Speech-to-Text API**
3. Text is cleaned (removing noise, handling slang, etc.)
4. Passed to model for classification
5. Output → Abusive / Not Abusive

---

## Dataset

The dataset is included in this repository.

It consists of code-mixed data from:

* Tamil-English
* Hindi-English
* Kannada-English
* Malayalam-English

The data contains real-world social media text with:

* mixed languages
* slang
* spelling variations

---

## Training

There are **separate training scripts for each language pair**:

```id="tq7v1n"
train_tamil_eng.py
train_hindi_eng.py
train_kannada_eng.py
train_malayalam_eng.py
```

Each script trains a model specific to that language combination.

### Example:

```bash id="0w7n6v"
python train_tamil_eng.py
```

---

## Trained Model

Trained models are **not included** in this repository due to file size limitations.

You can train the models using the scripts provided above.

---

## Results

The model performs reasonably well across languages:

* Tamil-English → ~82% accuracy
* Kannada-English → ~77%
* Malayalam-English → ~72%
* Hindi-English → ~71%

---

## Notebook

You can also refer to the notebook:

```id="f3yx0d"
Tamil_Eng_training.ipynb
```

Make sure to **clear outputs before running/uploading** to avoid large file size.

---

## Installation

```bash id="4d7r9k"
pip install -r requirements.txt
```

---

## Sample Outputs

* "yeh bahut bakwas hai" → Abusive
* "super video bro nice work" → Not Abusive

---


## Future Improvements

* Improve accuracy
* Handle sarcasm better
* Add more languages
* Real-time deployment

---


## Final Note

This project shows how multilingual and messy real-world data can be handled using transformer-based models, along with speech input using Sarvam AI.
