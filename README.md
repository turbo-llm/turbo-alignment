<div align="center">
  <img src="https://your-logo-url.com/logo.png" alt="turbo-alignment" height="110">
</div>

# 🚀 Turbo-Alignment
> Library for industrial alignment.


## Table of Contents
- [What is Turbo-Alignment?](#-what-is-turbo-alignment)
- [Key Features](#-key-features)
- [Supported Methods](#-supported-methods)
- [Implemented metrics](#implemented-metrics)
- [Installation](#-installation)
  - [Python Package](#-python-package)
  - [From Source](#-from-source)
  - [Repository](#-repository)
- [How to Use](#-how-to-use)
- [Command Line Interface (CLI)](#cli)
- [Development](#-development)
- [Library Roadmap](#-library-roadmap)
- [FAQ](#-faq)
- [License](#-license)

## 🌟 What is Turbo-Alignment?

Turbo-Alignment is a library designed to streamline the fine-tuning and alignment of large language models, leveraging advanced techniques to enhance efficiency and scalability.

## ✨ Key Features

- **📊 Comprehensive Metrics and Logging**: Includes a wide range of metrics such as self-bleu, KL divergence, and diversity, all supported out of the box.
- **🛠️ Streamlined Method Deployment**: Simplifies the process of deploying new methods, allowing for quick development and integration of new datasets and trainers into your operational flow.
- **🌐 Multimodal Capabilities**: Extensive support for various multimodal functions like Vision Language Modeling.
- **🔍 RAG Pipeline**: Unique pipeline for end2end retrieval-augmented generation training.
- **📚 Ready-to-Use Examples**: Convenient examples with configurations and instructions for basic tasks.
- **⚡ Fast Inference**: Optimized for quick inference using vLLM.
- **🔄 End-to-End Pipelines**: From data preprocessing to model alignment.
- **🔧 Adaptive Integration**: Facilitates the incorporation of new methodologies into established systems, ensuring compatibility and extending functionality without disrupting existing operations.

## 🛠️ Supported Methods


Turbo-Alignment supports a wide range of methods for model training and alignment, including:
- **🎯** Supervised Fine-Tuning ([SFT](#SFT))
- **🏆** Reward Modeling ([RM](#RM))
- **👍** Direct Preference Optimization ([DPO](#DPO))
- **🌟** Sequence Likelihood Calibration with Human Feedback ([SLiC](#SLiC))
- **🧠** Kahneman & Tversky Optimization ([KTO](#KTO)) Paired/Unpaired
- **📊** Statistical Rejection Sampling Optimization([RSO](#RSO))
- **🔄** Contrastive Preference Optimization ([CPO](#CPO))
- **🎭** Identity Preference Optimisation ([IPO](#IPO))
- **🗂️** Retrieval-Augmented Generation ([RAG](#RAG))
- **🌁** Vision Language Modeling using MLP from [LLaVA](#LLaVA) or C-Abstractor from [HoneyBee](#HoneyBee) trainable projection model

## Implemented Metrics
- **🔠 Distinctness**
- **🌈 Diversity**
- **➗ KL-divergence**
- **📏 Length**
- **🌟 METEOR**
- **🌀 Perplexity**
- **🏆 Reward**
- **📜 ROUGE**
- **🔵 Self-BLEU** 
- **📐RAGAS**
- **🔍 Retrieval Utility**

## 🤖 How to Use

Turbo-Alignment offers an intuitive interface for training and aligning large language models. Refer to the detailed examples and configuration files in the documentation to get started quickly with your specific use case.

User-friendly [guide](docs/GUIDE.md)

The most crucial aspect is to prepare the dataset in the required format, after which the pipeline will handle everything automatically.
Examples of datasets are available [here](docs/dataset_example.md).

## Table of use-cases
- [Supervised Fine-Tuning](#-sft)
- [Preference Tuning](#-preftune)
- [Multimodal](#-multimodal)
- [RAG](#-rag)

<a name="-sft"></a>
## Supervised Fine-Tuning
- **Prepare your dataset** in the `chat` format.
- **Suitable config**: [sft.json](configs/exp/train/sft/sft.json)
```bash
python -m turbo_alignment train_sft --experiment_settings_path ./fixtures/configs/train/sft/base.json
```
<a name="-preftune"></a>
## Reward Modeling / Direct Preference Optimization / Kahneman & Tversky Optimization (Paired)  / CPO / IPO
- **Prepare your dataset** in the `pair_preferences` format.
- **Suitable config**: [rm](configs/train/rm/rm.json), [dpo](tests/fixtures/configs/train/dpo/base.json), [kto](tests/fixtures/configs/train/kto/base.json)

<a name="-multimodal"></a>
## Multimodal Tasks 
To start multimodal training, you should:
- **Prepare the multimodal dataset**. See [Dataset Formatting Guide] for examples.
- **Preprocess the data (OPTIONAL)**. If you plan to run many experiments on the same dataset, you should preprocess it. The preprocessing stage includes reading `pixel_values` from images, encoding them with the specified encoder, and saving them in safetensors format. Later, during training, the pipeline will skip the stage of reading and encoding images and only extract prepared encodings from the safetensors files.
- **Suitable config**: [llava.json](tests/fixtures/configs/train/multimodal/llama_llava_base_clip.json),[c_abs.json](tests/fixtures/configs/train/multimodal/llama_c_abs_clip_pickle.json)

<a name="-rag"></a>
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

<a name="cli"></a>
## 🖥️ Command Line Interface (CLI)

Turbo-Alignment CLI allows quick fine-tuning and testing of models:

**Inference:**
```bash
python -m turbo_alignment inference_rag --inference_settings_path path/to/rag_inference_settings.json
python -m turbo_alignment inference_chat --inference_settings_path path/to/chat_inference_settings.json
python -m turbo_alignment inference_multimodal --inference_settings_path path/to/multimodal_inference_settings.json
python -m turbo_alignment inference_classification --inference_settings_path path/to/classification_inference_settings.json
```

**Train:**
```bash
python -m turbo_alignment train_rm --experiment_settings_path path/to/rm_experiment_settings.json
python -m turbo_alignment train_rag --experiment_settings_path path/to/rag_experiment_settings.json
python -m turbo_alignment train_sft --experiment_settings_path path/to/sft_experiment_settings.json
python -m turbo_alignment train_dpo --experiment_settings_path path/to/dpo_experiment_settings.json
python -m turbo_alignment train_kto --experiment_settings_path path/to/kto_experiment_settings.json
python -m turbo_alignment train_multimodal --experiment_settings_path path/to/multimodal_experiment_settings.json
python -m turbo_alignment train_classification --experiment_settings_path path/to/classification_experiment_settings.json
```

**Sampling:**
```bash
python -m turbo_alignment rm_sample --experiment_settings_path path/to/rm_sampling_settings.json
python -m turbo_alignment rso_sample --experiment_settings_path path/to/rso_sampling_settings.json
python -m turbo_alignment random_sample --experiment_settings_path path/to/random_sampling_settings.json
```

<a name="-installation"></a>
##

 🚀 Installation

### 📦 Python Package
⌛ in progress..

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

### Where can I find the user-guide?
Guide is available [here](docs/GUIDE.md).


## 📝 License
This project is licensed, see the [LICENSE](https://github.com/turbo-llm/turbo-alignment/-/blob/main/LICENSE) file for details.