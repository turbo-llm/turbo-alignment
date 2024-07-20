# Hello! 🎉
After reading this guide, you'll understand what's happening in the turbo-alignment configs.
more configs you can find [here](tests/fixtures/configs)

## Table of Contents
- [Default Settings](#default-settings)
  - [Model Settings](#model-settings)
  - [Tokenizer Settings](#tokenizer-settings)
  - [Generation Settings](#generation-settings)
  - [Dataset Settings](#dataset-settings)
- [Inference](#-inference-stuff)
  - [Default Generation](#default-generation-inference)
  - [vLLM Generation](#vllm-generation-inference)
  - [Classification](#classification-inference)
  - [RAG](#rag-inference)
    - [Offline RAG](#offline-rag-inference)
    - [Online RAG](#online-rag-inference)
- [Train](#train-stuff)
  - [LORA](#lora-adapter-train)
  - [P-tune](#p-tuning-train)
  - [Custom](#custom-modifications)
  - [Classification](#llm-for-classification)
  - [RAG](#end-2-end-rag)
  - [Reward Model](#reward-model)
- [Multimodal](#multimodal-tasks)



## Default Settings
### Model Settings
Basic settings to load the model.

```json
"model_settings": {
        "model_path": "/from_s3/model", "model_type": "causal",
        "adapter_path": "/from_s3/adapters",
        "transformers_settings": {}, "model_kwargs": {}
      }
```

**model_kwargs**  -- the place to specify something like "attn_implementation": "flash_attention_2" <br>
**model_type** -- For classification/RM "seq_cls" is needed, for most others "causal" is suitable<br>
**adapter_path** -- for vLLM-inference, it's necessary to merge the model in advance using <br>

merge config for vLLM [here](configs/utils/convert_to_base/llama.json)

### Tokenizer Settings
Basic settings to load the tokenizer <br>
```json
"tokenizer_settings": {"use_fast": false, "tokenizer_path": "/from_s3/tokenizer"}
```
**use_fast** -- critical for some models

### Generation Settings 
**transformers_settings** -- standard model generation settings <br>
**custom_settings** -- useful for display. for example, you can choose to display or not display the prompt/special tokens

```json
"generation_settings": [{
          "transformers_settings": {"num_beams": 3,"max_new_tokens": 500,"repetition_penalty": 1.02 },
          "custom_settings": {"generation_eos_token": "</RS>", "skip_special_tokens": false}
        }]
```
**generation_eos_token** -- varies between models, or you might have trained your own

### Dataset Settings
**sources** -- the name and path of the dataset. You can choose **num_samples** or **sample_rate** to control how much of the dataset to use <br>
**chat_settings** -- how your messages will be processed for input into the llm
```json
"dataset_settings": {
    "sources": [{"name": "val","records_path": "/from_s3/dataset/val_chat.jsonl", "sample_rate": 1}],
    "prompt_template": {"role_tag_mapping": {"bot": "<bot>","user": "<user>"},
        "prefix_template": "<RS>{role}",
        "suffix_template": "</RS>"},
    "dataset_type": "chat",
    "max_tokens_count": 150,
  }
```
**prompt_template** -- For each message from **dataset[’messages’][i]** which looks like {role: role, content: content}, we get a string of **[Prefix_template + content + Suffix_template]** then we combine all obtained strings into a single text.

**dataset_type** -- a important parameter that must match the type of method you want to run

## 🚀 Inference Stuff!

### Default Generation inference

- Let's look at the model+adapter ([default inference](configs/exp/inference/generation/default_llama_adapter.json))

- Make sure that in **role_tag_mapping** you have all roles from your dataset, and they correspond to the tokens from your model.

###  vLLM Generation inference 
Suppose you want to use vLLM: for speeding up inference or CUDA_OUT_OF_MEMORY_ERROR👹:

- Then this config is suitable: [vllm inference](configs/exp/inference/generation/vllm.json)
- In fact, only 2 minor changes: <br>
 (a) add **use_vllm: true** <br>
 (b) specify the path directly to the model, adapter loading will not work. [merge config](configs/utils/convert_to_base/llama.json)

### Troubleshooting:
- If the sizes of the dictionary in the base_model/adapter do not match, pay attention to `special_tokens_setter`, you might have duplicate special tokens after train.

### Classification inference:
Don't forget to specify that your model is for classification😏   [classification inference ](configs/exp/inference/classification/classification_inference.json)


```json
"model_settings": { "model_type": "seq_cls",
          "model_kwargs": {"num_labels": 2,"problem_type": "single_label_classification"},
"dataset_type": "classification”
```



### RAG inference
#### Offline RAG inference
Prepare a dataset where for each query you pre-find suitable passages → form them into **"dataset_type": "chat"* and launch just like in [Default Generation inference](#-default-generation-inference).
#### Online RAG inference
You can load your encoder and index, and perform passage retrieval online:
For this, check this config: [rag_inference](configs/exp/inference/rag/rag_inference.json)
if you're dealing with RAG, this part should already be familiar to you.

```json
{"question_encoder_settings": {}
"index_settings": {}
"retrieval_settings": {}}
```


## 🏋️Train Stuff! <a name="train-stuff"></a>

#### embeddings_initialization_strategy:

```json
{"embeddings_initialization_strategy": {"<RS>": "<s>", "<super_bot>": "the best bot ever"}}
``` 
During training, you might want to add your special tokens, e.g., for RAG **<doc_sep>** is useful, for multimodal tasks you might want to specify a particular **<modal_name>**.

In this case, we add the token **<RS>** by initializing its weights with the token **<s>**, and the token "<super_bot>" by averaging the tokens that split the string "the best bot ever"

#### train_dataset_settings
similiar as inference_dataset but 
Pay attention to **"only_answer_loss": true**:
This parameter means the model will calculate the error only on the last message from dataset[’messages’]. In most cases, you want the last message to be from the role: **bot**, otherwise, you're training the model to mimic the user! 😏
```json
{"keep_start": "bool", "keep_end": "bool"}
```
 **CUT**: if keep_start -> [:max_tokens_count] elif keep_end -> [-max_tokens_count:]; <br>
 cuts off the last fully entered message in the dialogue. <br>

Only for cherry_pick_settings: <br>
if **random_cut = True**, then the end is chosen as a random bot message from messages.

### LORA Adapter train
Check out this config: [LORA Adapter](configs/exp/train/sft/llama/default_chat_tuning_llama_7b.json)


### P-tuning train 
No problem u can use
PrefixTuning|Lora|PromptTuning| PTuning <br>
Check this out: [P-Tuning](configs/exp/train/sft/llama/ptuning.json)<br>

```json
"peft_settings": {"name": "P_TUNING","num_virtual_tokens": 32}
```

### Custom Modifications
#### is model pay attention to specific doc?
We prepare the dataset in advance with separators **<doc_sep>** initialized simply as **<sep>** (or you could use 'Document' or whatever else you like).
Add the corresponding metric **retrieval_utility** ([link](configs/exp/train/sft/llama/sft_with_retrieval_utility.json) and see what happens!

#### any other creative idea you come up with:
- Prepare the dataset 🧑‍🔬
- Specify the metric 🔍
- Train/Watch 👀

### LLM for classification
[classification train](configs/exp/train/classification/classification.json)

```json
{"dataset_type": "classification",
"model_type": "seq_cls",
"model_kwargs": {"num_labels": 2, "return_dict": true, "problem_type": "single_label_classification"},
"peft_setting": {"task_type": "SEQ_CLS"}}}
```



### end-2-end-rag 
- Check this out if you're inspired by an article and want to train both the encoder and the generator at once: [end2end_rag](configs/exp/train/rag/end2end_rag.json)
- Just fill in the appropriate settings **"question_encoder_settings", "retrieval_settings", "index_settings".**
- **"dataset_type": "chat"**

### Reward Model
Preferences are everything → prepare the pair_preference dataset→ run config [RM](configs/exp/train/rm/rm.json)
```json
    {"add_labels": false,
    "dataset_type": "pair_preferences"
    "model_settings": {"model_type": "seq_cls"}}
```

### Multimodal Tasks
In this detailed guide, we will consider the multimodal pipeline capabilities.

In turbo-alignment, you can train your own Vision Language Models (VLMs), such as LLaVA, Qwen-VL, IDEFICS, HoneyBee, etc. 

## Multimodal Architecture
The architecture of VLM contains only three parts: a multimodal encoder that encode images into some representation, a projector that maps representations from encoders to language model tokens, and a language model. During training, we completely freeze the encoders and train only the projector with the language model. Currently only the `image` modality is supported, but we will add `audio` support in the future.

The encoders are stored in `turbo_alignment/modeling/multimodal/encoders`. The encoder class takes multimodal features (pixel values in the case of images) and encodes them using the encoder model. To add a new image encoder, simply create a new file in `turbo_alignment/modeling/multimodal/encoders/image` and write a class that inherits from `BaseImageEncoder`. Note that your class should contain the method `def encode(self, inputs: torch.Tensor) -> torch.Tensor:` and some properties like `emb_dim` (dimension of each encoded patch), `device` and `n_modality_embs` (the number of patches your encoder returns). Take [CLIP encoder](turbo_alignment/modeling/multimodal/encoders/image/clip.py) as an example.

Readers are stored in `turbo_alignment/common/data/multimodal`. Each encoder has its own reader. That's because encoder models like CLIP are trained with its specific processor (reader in our pipeline). To add a new image reader, create a class by inheriting from `BaseImageReader` and implement the method `def read(self, path: str) -> torch.Tensor:`. See our [CLIP reader](turbo_alignment/common/data/multimodal/image/clip.py).

All projectors are stored in `turbo_alignment/modeling/multimodal/projectors`. Basically, Projector is a class that takes multimodal features (i.e., encoded image or audio) and performs some operations on them to adapt to the language model. For example, it can simply map each patch of the coded image to a dimension of the language model's tokens, like MLP from LLaVA, or perform some convolutions, like C-Abstractor from HoneyBee. If you want to contribute and add a new projector to our pipeline, see the [LLaVA MLP](turbo_alignment/modeling/multimodal/projectors/llava.py) projector implementation as an example.



## Multimodal Training Pipeline

### Config
An example of a multimodal config can be found here: ([llama_llava_base_clip.json](tests/fixtures/configs/train/multimodal/llama_llava_base_clip.json)). This test configuration shows how to train a VLM with CLIP as image encoder and LLaMA as LM with MLP projector.


#### Dataset Settings:
This section covers all dataset settings (i.e. `train_dataset_settings`, `val_dataset_settings` and `dataset_settings` in `cherry_pick_settings`). 

First, a `modality_token_mapping`. For all modalities (image and audio), specify the names for the modality tokens in this JSON:

```json
"modality_token_mapping": {
    "image": "<img>",
    "audio": "<audio>"
}
```

Next, a `modality_reader_settings_mapping`, which is just a mapping of modality (`image` or `audio`) and modality reader settings. Basically, the modality reader is a class that reads and processes the image to prepare it for the modality encoder. These settings include `reader_type` (type of modality reader, currently `clip` and `imagebind` are supported for image modality) and `reader_path` (path to model). For `clip`, the modality reader includes a call to the CLIPProcessor, which is typically stored in the same directory as the CLIPModel.

```json
"modality_reader_settings_mapping": {
    "image": {
        "reader_type": "clip",
        "reader_path": "tests/fixtures/models/clip_tiny"
    },
    "audio": null
}
```

The next key is `n_modality_embeddings`. Projectors like MLP from LLaVA map each patch of the encoded image as an input token for the language model. Thus, for the MLP Projector, `n_modality_embeddings` should be equal to the number of patches (the pipeline will throw an error if you set this parameter incorrectly and will specify the correct value).

```json
"n_modality_embeddings": 225
```

The next two keys are `start_modality_token` and `end_modality_token`. These tokens will be inserted before and after modality object in the encoded dialog.

```json
"start_modality_token": "<MS>",
"end_modality_token": "</MS>"
```

Finally, don't forget that multimodal dataset has its own type `multimodal`.

```json
"dataset_type": "multimodal"
```

#### Other important keys:
Use `modality_encoder_settings_mapping` to configure the modality encoders. First, `modality_encoder_type` is the type of modality encoder (as mentioned, only `clip` and `imagebind` are currently supported for images). Next, `encoder_path` is the path to your encoder model. Finally, `is_pickle` is the parameter that you should only set to `true` if you have performed a preprocessing step and modality encoders should not do anything with the input data (because in case of preprocessing they have already extracted all representations).

```json
"modality_encoder_settings_mapping": {
    "image": {
        "modality_encoder_type": "clip",
        "encoder_path": "tests/fixtures/models/clip_tiny",
        "is_pickle": false
    },
    "audio": null
}
```

Key `modality_projector_mapping` maps modality to Projector type. For image modality, `llava` (MLP) and `c_abs` (C-Abstractor from HoneyBee) are currently implemented. Feel free to add your own Projectors!
```json
"modality_projector_mapping": {
    "image": "llava",
    "audio": null
}
```

If you want to initialize `modality_projector_initialization_mapping` your Projector with some existing weights, pass the path to weights here.
```json
"modality_projector_initialization_mapping": {
    "image": null,
    "audio": null
}
```

### Launch
```bash
python -m turbo_alignment train_multimodal --experiment_settings_path tests/fixtures/configs/train/multimodal/llama_llava_base_clip.json
```


## Multimodal Inference

### Config
Now we will consider an configuration file for inference: ([llama_llava_clip_pickle.json](tests/fixtures/configs/inference/multimodal/llama_llava_clip_pickle.json))
Unlike SFT Inference pipeline, multimodal one takes `projections_path` (path to the trained projectors) and `n_modality_embeddings`. Both of them are in `model_settings`, for example:
```json
"model_settings": {
    "model_path": "tests/fixtures/models/llama2_tiny",
    "projections_path": "tests/fixtures/models/llama2_tiny_multimodal_clip_mlp/projections/modality_adapters.pt",
    "n_modality_embeddings": 225,
    "model_type": "causal",
    "transformers_settings": {},
    "adapter_path": "tests/fixtures/models/llama2_tiny_multimodal_clip_mlp/adapter"
}
```

Also specify `modality_encoder_settings_mapping` and `modality_projector_mapping` (as in training config):
```json
"modality_encoder_settings_mapping": {
    "image": {
        "modality_encoder_type": "clip",
        "is_pickle": false,
        "encoder_path": "tests/fixtures/models/clip_tiny"
    },
    "audio": null
},
"modality_projector_mapping": {
    "image": "llava",
    "audio": null
}
```

In `dataset_settings`, add `modality_token_mapping` and `modality_reader_settings_mapping` (again, there is no difference with training config):

```json
"modality_token_mapping": {
    "image": "<img>",
    "audio": "<audio>"
},
"modality_reader_settings_mapping": {
    "image": {
        "reader_type": "clip",
        "reader_path": "tests/fixtures/models/clip_tiny",
    },
    "audio": null
}
```


### Launch
```bash
python -m turbo_alignment inference_multimodal --experiment_settings_path tests/fixtures/configs/inference/multimodal/llama_llava_clip_pickle.json
```


## How to speed up multimodal pipeline with dataset preprocessing
To speed up the training process, you can consider preprocessing your data. Without preprocessed data, the multimodal training pipeline will read images with your reader before training, and encode them on each iteration of the training loop. 

To start preprocessing, all you need is a directory with images and a valid preprocessing config. For our example, we will use the test config [images.json](tests/fixtures/configs/utils/preprocess/images.json).

The config contains information about the modality, modality reader, modality encoder, path to data with images, and output path. You can set the output path to be the same as the input path. 

```json
{
    "modality": "image",
    "reader_settings": {
        "reader_type": "clip",
        "reader_path": "tests/fixtures/models/clip_tiny"
    },
    "encoder_settings": {
        "modality_encoder_type": "clip",
        "encoder_path": "tests/fixtures/models/clip_tiny"
    },
    "dataset_path": "tests/fixtures/datasets/multimodal/images",
    "batch_size": 256,
    "output_file_path": "tests/fixtures/datasets/multimodal/images"
}
```

Then run the script using the cli interface:
```bash
python -m turbo_alignment preprocess_multimodal_dataset --settings_path tests/fixtures/configs/utils/preprocess/images.json
```

The result of the preprocessing script is the file `tests/fixtures/datasets/multimodal/images/image.clip.safetensors`. The output safetensors file is a dict with the image path as key and the encoded image tensor as value. 

You should also make some changes in the training configuration. In `modality_reader_settings_mapping` set `reader_type` to `pickle`. This reader simply opens the safetensors file and reads pre-encoded modality objects from it.

```json
"modality_reader_settings_mapping": {
    "image": {
        "reader_type": "pickle",
        "reader_path": null
    },
    "audio": null
}
```

Taking into account the `modality_encoder_settings_mapping`, set the `is_pickle` key to `true`. Then the encoder will not re-encode objects and will simply return the input data as it is (because it was pre-encoded).
```json
"modality_encoder_settings_mapping": {
    "image": {
        "modality_encoder_type": "clip",
        "is_pickle": true,
        "encoder_path": "tests/fixtures/models/clip_tiny"
    },
    "audio": null
}
```

As you can guess, the preprocessing trick could be applied to the inference pipeline as well. Just make the changes described above.
