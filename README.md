# 🌐 Hindi → Bengali Neural Machine Translation

> Fine-tuning and evaluating state-of-the-art NMT models for low-resource Indic language translation, with a live Django-powered web interface.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Models](#models)
- [Evaluation Results](#evaluation-results)
- [Installation](#installation)
- [Usage](#usage)
  - [1. Prepare the Dataset](#1-prepare-the-dataset)
  - [2. Train the Models](#2-train-the-models)
  - [3. Evaluate the Models](#3-evaluate-the-models)
  - [4. Run the Web Interface](#4-run-the-web-interface)
- [Django Integration Details](#django-integration-details)
- [Tech Stack](#tech-stack)
- [License](#license)

---

## Overview

This project explores the fine-tuning of two state-of-the-art multilingual Neural Machine Translation (NMT) models — **Facebook's NLLB-200-distilled-600M** and **AI4Bharat's IndicTrans2-indic-indic-dist-320M** — for the **Hindi → Bengali** translation pair, a linguistically rich but relatively low-resource corridor within the Indic language family.

Fine-tuned models are comprehensively evaluated using four industry-standard metrics: **BLEU**, **METEOR**, **BERTScore**, and **COMET**. The best-performing model is then served through a modern **Django-based NLP Lab web application** that allows users to translate text in real-time and compute live evaluation scores against a user-provided reference translation.

---

## ✨ Features

- **Multi-domain Parallel Corpus** covering Agriculture, Cricket, Disease, Entertainment, Gadgets, Judiciary, News, Recipe, Tourism, and more.
- **Dual Model Fine-tuning** — both NLLB (multilingual transformer) and IndicTrans2 (Indic-specialized) trained and evaluated side-by-side.
- **Four Evaluation Metrics** — SacreBLEU, METEOR, BERTScore (F1), and COMET computed post-training.
- **Live Django Web App** — real-time translation with on-the-fly metric computation against user-supplied reference sentences.
- **Singleton Model Loading** — prevents OOM errors by loading large GPU models exactly once per server lifetime.
- **Glassmorphism Dark UI** — premium, responsive frontend with animated gradients and a loading state.

---

## 📁 Project Structure

```
Exploratory/
│
├── Hindi/                          # Source-side Hindi corpus
│   ├── AGRICULTURE/
│   ├── CRICKET/
│   ├── DISEASE/
│   └── ...                         # (11 domain subdirectories)
│
├── Bengali/                        # Target-side Bengali corpus (mirrors Hindi/)
│   ├── AGRICULTURE/
│   ├── CRICKET/
│   └── ...                         # (11 domain subdirectories)
│
├── prepare_data.py                 # Loads & aligns parallel corpus → saves HF dataset (80/20 split)
├── train_nllb.py                   # Fine-tunes facebook/nllb-200-distilled-600M
├── train_indictrans2.py            # Fine-tunes ai4bharat/indictrans2-indic-indic-dist-320M
├── evaluate_models.py              # Batch-evaluates both models (BLEU, METEOR, BERTScore, COMET)
├── requirements.txt                # Python dependencies
├── main.ipynb                      # Exploratory notebook
│
└── django_integration/             # Django web app components
    ├── services.py                 # TranslationService singleton (model loading + inference + metrics)
    ├── views.py                    # Django views (render UI + /api/translate/ endpoint)
    ├── urls.py                     # URL routing
    └── index.html                  # Premium glassmorphism frontend template
```

The trained model checkpoints will be saved to:
```
nllb_finetuned_hi_bn/final_model/
indictrans2_finetuned_hi_bn/final_model/
```

---

## 📊 Dataset

The parallel corpus is organized by **domain** with aligned Hindi and Bengali `.txt` files (one sentence per line). Domains covered include:

| Domain | Description |
|---|---|
| AGRICULTURE | Agricultural news and advice |
| BOX-OFFICE | Film box office reports |
| CONVERSATIONAL | Conversational/dialog text |
| CRICKET | Cricket match commentary & news |
| DISEASE | Health and disease information |
| ENTERTAINMENT | Entertainment news |
| GADGET | Technology & gadget reviews |
| JUDICIARY | Legal and court proceedings |
| NEWS-ARTICLES | General news |
| RECIPE | Food and cooking instructions |
| TOURISM | Travel and tourism content |

**`prepare_data.py`** scans both `Hindi/` and `Bengali/` directories, pairs each file by name and domain, validates line-count alignment, and saves an 80/20 train/test split as a Hugging Face `DatasetDict` to `./hf_dataset/`.

---

## 🤖 Models

| Model | Parameters | Source |
|---|---|---|
| `facebook/nllb-200-distilled-600M` | 600M | Meta AI — Multilingual (200 languages) |
| `ai4bharat/indictrans2-indic-indic-dist-320M` | 320M | AI4Bharat — Indic language specialist |

**Language codes used:**
- Source: `hin_Deva` (Hindi, Devanagari script)
- Target: `ben_Beng` (Bengali, Bengali script)

Both models are fine-tuned for **3 epochs** using the HuggingFace `Seq2SeqTrainer` with:
- Learning rate: `2e-5`
- Batch size: `8` (train & eval)
- Weight decay: `0.01`
- FP16 mixed-precision (when GPU is available)

> **Note for IndicTrans2:** The `IndicProcessor` from [IndicTransToolkit](https://github.com/VarunGumma/IndicTransToolkit) is applied for script normalization and pre/post-processing during both training and inference.

---

## 📈 Evaluation Results

Models are evaluated post-training using `evaluate_models.py` on the held-out 20% test set.

| Metric | NLLB-600M (Fine-tuned) | IndicTrans2-320M (Fine-tuned) |
|---|---|---|
| **BLEU** | 38.36 | — |
| **METEOR** | 0.6043 | — |
| **BERTScore (F1)** | 0.9123 | — |
| **COMET** | 0.8848 | — |

> _Fill in IndicTrans2 results after running `evaluate_models.py` on your trained checkpoint._

---

## ⚙️ Installation

### Prerequisites
- Python 3.9+
- CUDA-compatible GPU (strongly recommended; 16GB+ VRAM for training)
- [Hugging Face account](https://huggingface.co/) with a read-access token (required for IndicTrans2)

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/Exploratory.git
cd Exploratory
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

`requirements.txt` includes:
```
transformers>=4.35.0
datasets>=2.14.0
evaluate>=0.4.0
sacrebleu>=2.3.1
bert_score>=0.3.13
unbabel-comet>=2.2.0
accelerate>=0.24.1
torch>=2.0.0
numpy
sentencepiece
protobuf
nltk
mosestokenizer
indic-nlp-library
IndicTransToolkit @ git+https://github.com/VarunGumma/IndicTransToolkit.git
```

### 3. Authenticate with Hugging Face (for IndicTrans2)

```bash
huggingface-cli login
```
Provide a token with **read** permissions from your [HF account settings](https://huggingface.co/settings/tokens).

---

## 🚀 Usage

### 1. Prepare the Dataset

```bash
python prepare_data.py
```

This will:
- Read all aligned `Hindi/` and `Bengali/` domain files
- Create an 80/20 train/test split
- Save the dataset to `./hf_dataset/`

Output:
```
Loaded XXXX parallel sentence pairs.
Train size: XXXX
Test size:  XXXX
Saved dataset to: ./hf_dataset
```

---

### 2. Train the Models

**Train NLLB:**
```bash
python train_nllb.py
```
Checkpoint saved to: `./nllb_finetuned_hi_bn/final_model/`

**Train IndicTrans2:**
```bash
python train_indictrans2.py
```
Checkpoint saved to: `./indictrans2_finetuned_hi_bn/final_model/`

> ⚡ **Recommended:** Run training on a GPU instance (e.g., Lightning AI, Colab, or a local CUDA machine). Training on CPU is extremely slow.

---

### 3. Evaluate the Models

```bash
python evaluate_models.py
```

This script:
- Loads the fine-tuned (or base) models
- Runs batch inference over the 20% test split
- Prints BLEU, METEOR, BERTScore, and COMET scores for each model

---

### 4. Run the Web Interface

The `django_integration/` folder contains all files needed to serve the translation UI inside a Django project.

#### Step 1 — Set up a Django project (if you don't have one)

```bash
pip install django
django-admin startproject nlp_lab
cd nlp_lab
python manage.py startapp mt_app
```

#### Step 2 — Copy integration files

```bash
cp ../django_integration/services.py  mt_app/services.py
cp ../django_integration/views.py     mt_app/views.py
cp ../django_integration/urls.py      mt_app/urls.py

mkdir -p mt_app/templates/mt_app/
cp ../django_integration/index.html   mt_app/templates/mt_app/index.html
```

#### Step 3 — Update `settings.py`

Add `'mt_app'` to `INSTALLED_APPS`:
```python
INSTALLED_APPS = [
    ...
    'mt_app',
]
```

#### Step 4 — Wire up `nlp_lab/urls.py`

```python
from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('mt_app.urls')),
]
```

#### Step 5 — Update Model Paths in `services.py`

Edit the `BASE_PATH` variable to point to the directory where your fine-tuned model folders (`nllb_finetuned_hi_bn/`, `indictrans2_finetuned_hi_bn/`) are stored:

```python
BASE_PATH = "/path/to/your/models/"
```

If the fine-tuned checkpoints are not present, the service automatically falls back to the base pretrained models from Hugging Face.

#### Step 6 — Run the server

```bash
python manage.py runserver 0.0.0.0:8000
```

---

## 🖥️ Django Integration Details

### API Endpoint

**`POST /api/translate/`**

**Request body (JSON):**
```json
{
  "text": "यहाँ अपना हिंदी पाठ दर्ज करें।",
  "model": "nllb",
  "reference": "(optional) ground-truth Bengali sentence"
}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `text` | string | ✅ | Hindi source sentence |
| `model` | string | ✅ | `"nllb"` or `"indictrans2"` |
| `reference` | string | ❌ | Reference Bengali text for live metric scoring |

**Response (JSON):**
```json
{
  "translation": "এখানে আপনার বাংলা অনুবাদ।",
  "model_used": "nllb",
  "scores": {
    "bleu": 42.15,
    "meteor": 0.6521,
    "bertscore": 0.9204,
    "comet": 0.8912
  }
}
```

> `scores` is only returned when a `reference` translation is provided in the request.

### Architecture: Singleton Service

`TranslationService` uses the **Singleton pattern** to load both models into GPU memory exactly **once** on application startup. This avoids repeated loading of 600M+ parameter models on each request, which would otherwise cause Out-of-Memory (OOM) errors and severe latency.

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **Base Models** | `facebook/nllb-200-distilled-600M`, `ai4bharat/indictrans2-indic-indic-dist-320M` |
| **Training Framework** | HuggingFace `transformers`, `Seq2SeqTrainer` |
| **Preprocessing** | `IndicTransToolkit` (IndicProcessor), `sentencepiece` |
| **Evaluation** | `sacrebleu`, `meteor`, `bert_score`, `unbabel-comet` |
| **Dataset Management** | HuggingFace `datasets` |
| **Web Backend** | Django |
| **Frontend** | Vanilla HTML/CSS/JS with Glassmorphism dark UI |
| **Compute** | Lightning AI (GPU), CUDA via PyTorch |

---

## 📄 License

This project is intended for research and educational purposes.

---

<div align="center">
  <sub>Built with ❤️ for low-resource Indic language NLP research.</sub>
</div>
