# 🚀 Turbo-Alignment
> Library for industrial alignment


## Table of Contents
- [What is Turbo-Alignment?](#-what-is-turbo-alignment)
- [Key Features](#-key-features)
- [Supported Methods](#-supported-methods)
- [Implemented metrics](#-implemented-metrics)
- [How to Use](#-how-to-use)
- [Installation](#-installation)
- [Development](#-development)
- [Library Roadmap](#-library-roadmap)
- [FAQ](#-faq)
- [License](#-license)

<a name="-what-is-turbo-alignment"></a>
## 🌟 What is Turbo-Alignment?

Turbo-Alignment is a library designed to streamline the fine-tuning and alignment of large language models, leveraging advanced techniques to enhance efficiency and scalability.

<a name="-key-features"></a>
## ✨ Key Features

- **📊 Comprehensive Metrics and Logging**: Includes a wide range of metrics such as self-bleu, KL divergence, diversity, etc. all supported out of the box.
- **🛠️ Streamlined Method Deployment**: Simplifies the process of deploying new methods, allowing for quick development and integration of new datasets and trainers into your pipelines.
- **📚 Ready-to-Use Examples**: Convenient examples with configurations and instructions for basic tasks.
- **⚡ Fast Inference**: Optimized for quick inference using vLLM.
- **🔄 End-to-End Pipelines**: From data preprocessing to model alignment.
- **🌐 Multimodal Capabilities**: Extensive support for various multimodal functions like Vision Language Modeling.
- **🔍 RAG Pipeline**: Unique pipeline for end2end retrieval-augmented generation training.

<a name="-supported-methods"></a>
## 🛠️ Supported Methods


Turbo-Alignment supports a wide range of methods for model training and alignment, including:
- **🎯** Supervised Fine-Tuning (SFT)
- **🏆** Reward Modeling (RM)
- **👍** Direct Preference Optimization (DPO)
- **🧠** Kahneman & Tversky Optimization (KTO) Paired/Unpaired
- **🔄** Contrastive Preference Optimization (CPO)
- **🎭** Identity Preference Optimisation (IPO)
- **🌟** Sequence Likelihood Calibration with Human Feedback (SLiC-HF)
- **📊** Statistical Rejection Sampling Optimization (RSO)
- **🌁** Vision Language Modeling using MLP from (LLaVA) or C-Abstractor from (HoneyBee) trainable projection model
- **🗂️** Retrieval-Augmented Generation (RAG)

<a name="-implemented-metrics"></a>
## 🧮 Implemented Metrics
- **🔠** Distinctness
- **🌈** Diversity
- **🔵** Self-BLEU
- **➗** KL-divergence
- **🏆** Reward
- **📏** Length
- **🌀** Perplexity
- **🌟** METEOR
- **🔍** Retrieval Utility

<a name="-how-to-use"></a>
## 🤖 How to Use

Turbo-Alignment offers an intuitive interface for training and aligning large language models. Refer to the detailed examples and configuration files in the documentation to get started quickly with your specific use case. User-friendly guid available [here](docs/GUIDE.md).

The most crucial aspect is to prepare the dataset in the required format, after which the pipeline will handle everything automatically.
Examples of datasets are available [here](docs/dataset_example.md).

## Table of use-cases
- [Training](#-train)
  - [Supervised Fine-Tuning](#-sft-train)
  - [Preference Tuning](#-preftune-train)
    - [Reward Modeling](#-rm-train)
    - [DPO, CPO, IPO, KTO (Paired)](#-dpo-train)
    - [KTO (Unpaired)](#-kto-train)
  - [Multimodal](#-multimodal-train)
  - [RAG](#-rag-train)
- [Inference](#-inference)
  - [Supervised Fine-Tuning](#-sft-inference)
  - [Multimodal](#-multimodal-inference)
  - [RAG](#-rag-inference)
- [Sampling](#-sampling)
  - [Random](#-random-sampling)
  - [RM](#-rm-sampling)
  - [RSO](#-RSO-sampling)
- [Common](#-common)
  - [Preprocess](#-preprocess-common)
  - [Merge adapters to base](#-merge-adapters-to-base-common)

<a name="-train"></a>
# Train

<a name="-sft-train"></a>
## Supervised Fine-Tuning
- **📚 Dataset type** prepare your dataset  in the `ChatDataset`, examples available [here](docs/dataset_example.md#-chat-dataset) format.
- **📝 Configs Example**: [sft.json](configs/exp/train/sft/sft.json)
- **🖥️ CLI launch command**
```bash
python -m turbo_alignment train_sft --experiment_settings_path configs/exp/train/sft/sft.json
```
<a name="-preftune-train"></a>
## Preference Tuning
<a name="-rm-train"></a>
### Reward Modeling
- **📚 Dataset type** prepare your dataset  in the `PairPreferencesDataset` format, examples available [here](docs/dataset_example.md#-pair-preferences)
- **📝 Configs Example**: [rm.json](configs/exp/train/rm/rm.json)
- **🖥️ CLI launch command**
```bash
python -m turbo_alignment train_rm --experiment_settings_path configs/exp/train/rm/rm.json
```

<a name="-dpo-train"></a>
### DPO, IPO, CPO, KTO (Paired)
- **📚 Dataset type** prepare your dataset in the `PairPreferencesDataset` format, examples available [here](docs/dataset_example.md#pair-preferences)
- **📝 Configs Example**: [dpo.json](configs/exp/train/dpo/dpo.json)
- **🖥️ CLI launch command**
```bash
python -m turbo_alignment train_dpo --experiment_settings_path configs/exp/train/dpo/dpo.json
```

<a name="-kto-train"></a>
### KTO (Unpaired)
- **📚 Dataset type** prepare your dataset in the `KTODataset` format, examples available [here](docs/dataset_example.md#-kto-dataset)
- **📝 Configs Examples**: [kto.json](configs/exp/train/kto/kto.json)
- **🖥️ CLI launch command**
```bash
python -m turbo_alignment train_kto --experiment_settings_path configs/exp/train/kto/kto.json
```

<a name="-multimodal-train"></a>
⌛️ in progress..
## Multimodal Tasks 
To start multimodal training, you should:
- **Prepare the multimodal dataset**. See examples [here](docs/dataset_example.md#-multimodel-dataset).
- **Preprocess the data (OPTIONAL)**. If you plan to run many experiments on the same dataset, you should preprocess it. The preprocessing stage includes reading `pixel_values` from images, encoding them with the specified encoder, and saving them in safetensors format. Later, during training, the pipeline will skip the stage of reading and encoding images and only extract prepared encodings from the safetensors files.
- **Suitable config**: [llava.json](tests/fixtures/configs/train/multimodal/llama_llava_base_clip.json),[c_abs.json](tests/fixtures/configs/train/multimodal/llama_c_abs_clip_pickle.json)

<a name="-rag-train"></a>
⌛️ in progress..
## RAG
To launch RAG:
- **Choose a base encoder**, create a document index.
- For **end-to-end**:
  - **Train both** the retriever and the generator.
  - **Prepare the data** in `"dataset_type": "chat"` **with query -> response.**
  - **Suitable config**: [end2end_rag](configs/exp/train/rag/end2end_rag.json)

- For **sft-rag**:
  - **Train only** generator
  - **Prepare the data** in `"dataset_type": "chat"` with **query+retrieved_documents -> response.**
  - **Suitable config**: [sft_with_retrieval_utility](configs/exp/train/sft/llama/sft_with_retrieval_utility.json)


<a name="-inference"></a>
# Inference
⌛️ in progress..


<a name="-sampling"></a>
# Sampling
⌛️ in progress..

<a name="-common"></a>
# Common
⌛️ in progress..


<a name="-installation"></a>
## 🚀 Installation

### 📦 Python Package
```bash
pip install turbo-alignment
```

### 🛠️ From Source
For the latest features before an official release:
```bash
pip install git+https://github.com/turbo-llm/turbo-alignment.git
```

### 📂 Repository
Clone the repository for access to examples:
```bash
git clone https://github.com/turbo-llm/turbo-alignment.git
```

<a name="-development"></a>
## 🌱 Development

Contributions are welcome! Read the [contribution guide](https://github.com/turbo-llm/turbo-alignment/blob/main/CONTRIBUTING.md) and set up the development environment:
```bash
git clone https://github.com/turbo-llm/turbo-alignment.git
cd turbo-alignment
poetry install
```

<a name="-library-roadmap"></a>
## 📍 Library Roadmap

- Increasing number of tutorials
- Enhancing test coverage
- Implementation of Online RL methods like PPO and Reinforce
- Facilitating distributed training
- Incorporating low-memory training approaches


## ❓ FAQ
### How do I install Turbo-Alignment?
See the [Installation](#-installation) section for detailed instructions.

### Where can I find docs?
Guides and docs are available [here](docs/GUIDE.md).

### Where can I find tutorials?
Tutorials are available [here](tutorials/tutorial.md).


## 📝 License
This project is licensed, see the [LICENSE](https://github.com/turbo-llm/turbo-alignment/-/blob/main/LICENSE) file for details.
