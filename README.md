# 🚀 Turbo-Alignment
> Library for industrial alignment.


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
## Multimodal train
⌛️  in progress..


<a name="-rag-train"></a>
## RAG (Retrieval-Augmented Generation) 
<a name="-sft-rag-train"></a>
### SFT-RAG
- **📚 Dataset type**: prepare your dataset in `ChatDataset`, examples available [here](docs/dataset_example.md#-chat-dataset) format.
- **📝 Configs Example**: [sft_with_retrieval_utility](configs/exp/train/sft/llama/sft_with_retrieval_utility.json)
- **🖥️ CLI launch command**: 
```bash
python -m turbo_alignment train_sft --experiment_settings_path configs/exp/train/sft/llama/sft_with_retrieval_utility.json
```
<a name="-e2e-rag-train"></a>
### End2End-RAG
- **📚 Dataset type**: prepare your dataset in `ChatDataset`, examples available [here](docs/dataset_example.md#-chat-dataset) format.
- **📝 Configs Example**: [end2end_rag](configs/exp/train/rag/end2end_rag.json)
- **🖥️ CLI launch command**:
```bash
python -m turbo_alignment train_rag --experiment_settings_path configs/exp/train/rag/end2end_rag.json
```

<a name="-inference"></a>
# Inference
<a name="-chat-inference"></a>
## Chat Inference
- **📚 Dataset type** prepare your dataset  in the `ChatDataset`, examples available [here](docs/dataset_example.md#-chat-dataset) format.
- **📝 Configs Example**: [sft.json](configs/exp/inference/generation/default_llama_adapter.json)
- **🖥️ CLI launch command**
```bash
python -m turbo_alignment inference_chat --inference_settings_path configs/exp/inference/generation/default_llama_adapter.json
```

<a name="-classification-inference"></a>
## Classification Inference
- **📚 Dataset type** prepare your dataset  in the `ClassificationDataset`, examples available [here](docs/dataset_example.md#-classification-dataset) format.
- **📝 Configs Example**: [classification_inference.json](configs/exp/inference/classification/classification_inference.json)
- **🖥️ CLI launch command**
```bash
python -m turbo_alignment inference_classification --inference_settings_path configs/exp/train/sft/sft.json
```

<a name="-multimodal-inference"></a>
## Multimodal Inference
- **📚 Dataset type** prepare your dataset  in the `MultimodalDataset`, examples available [here](docs/dataset_example.md#-multimodal-dataset) format.
- **📝 Configs Example**: [mlp.json](configs/exp/inference/multimodal/mlp.json)
- **🖥️ CLI launch command**
```bash
python -m turbo_alignment inference_multimodal --inference_settings_path configs/exp/inference/multimodal/mlp.json
```

<a name="-rag-inference"></a>
## RAG Inference
- **📚 Dataset type** prepare your dataset  in the `ChatDataset`, examples available [here](docs/dataset_example.md#-chat-dataset) format.
- **📝 Configs Example**: [rag_inference.json](configs/exp/inference/rag/rag_inference.json)
- **🖥️ CLI launch command**
```bash
python -m turbo_alignment inference_rag --inference_settings_path configs/exp/inference/rag/rag_inference.json
```

<a name="-sampling"></a>
# Sampling
<a name="-random-sampling"></a>
## Random Sampling
- **📚 Dataset type** prepare your dataset  in the `SamplingRMDataset`, examples available [here](docs/dataset_example.md#-sampling-dataset) format.
- **📝 Configs Example**: [random.json](tests/fixtures/configs/sampling/base.json)
- **🖥️ CLI launch command**
```bash
python -m turbo_alignment random_sample --experiment_settings_path tests/fixtures/configs/sampling/base.json
```

<a name="-rso-sampling"></a>
## RSO Sampling
- **📚 Dataset type** prepare your dataset  in the `SamplingRMDataset`, examples available [here](docs/dataset_example.md#-sampling-dataset) format.
- **📝 Configs Example**: [rso.json](tests/fixtures/configs/sampling/rso.json)
- **🖥️ CLI launch command**
```bash
python -m turbo_alignment rso_sample --experiment_settings_path tests/fixtures/configs/sampling/rso.json
```

<a name="-rm-sampling"></a>
## Reward Model Sampling
- **📚 Dataset type** prepare your dataset  in the `SamplingRMDataset`, examples available [here](docs/dataset_example.md#-sampling-dataset) format.
- **📝 Configs Example**: [rm.json](tests/fixtures/configs/sampling/rm.json)
- **🖥️ CLI launch command**
```bash
python -m turbo_alignment rm_sample --experiment_settings_path tests/fixtures/configs/sampling/rm.json
```

<a name="-common"></a>
# Common
<a name="-merge_adapters_to_base"></a>
## Merge Adapters to base model
- **📝 Configs Example**: [llama.json](configs/utils/merge_adapters_to_base/llama.json)
- **🖥️ CLI launch command**
```bash
python -m turbo_alignment merge_adapters_to_base --settings_path configs/utils/merge_adapters_to_base/llama.json
```

<a name="-preprocess_multimodal_dataset"></a>
## Preprocess Multimodal Dataset
- **📝 Configs Example**: [coco2014_clip.json](configs/utils/preprocess/coco2014_clip.json)
- **🖥️ CLI launch command**
```bash
python -m turbo_alignment preprocess_multimodal_dataset --settings_path configs/utils/preprocess/coco2014_clip.json
```


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


## References

- DPO Trainer implementation inspired by Leandro von Werra et al. (2020) TRL: Transformer Reinforcement Learning. GitHub repository, GitHub. Available at: [https://github.com/huggingface/trl](https://github.com/huggingface/trl).

- Registry implementation inspired by Matt Gardner, Joel Grus, Mark Neumann, Oyvind Tafjord, Pradeep Dasigi, Nelson F. Liu, Matthew Peters, Michael Schmitz, and Luke S. Zettlemoyer. 2017. AllenNLP: A Deep Semantic Natural Language Processing Platform. Available at: [arXiv:1803.07640](https://arxiv.org/abs/1803.07640).

- Liger Kernels implementation inspired by Hsu, Pin-Lun, Dai, Yun, Kothapalli, Vignesh, Song, Qingquan, Tang, Shao, and Zhu, Siyu, 2024. Liger-Kernel: Efficient Triton Kernels for LLM Training. Available at: [https://github.com/linkedin/Liger-Kernel](https://github.com/linkedin/Liger-Kernel).
