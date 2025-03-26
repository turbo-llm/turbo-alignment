# Hello! 🎉
After reading this guide, you'll understand what's happening in the turbo-alignment configs.
More configs you can find [here](../tests/fixtures/configs)

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
- [Train](#train-stuff)
  - [LORA](#lora-adapter-train)
  - [P-tune](#p-tuning-train)
  - [Custom](#custom-modifications)
  - [Classification](#llm-for-classification)
  - [Reward Model](#reward-model)



## Default Settings
### Model Settings
Basic settings to load the model.

```json
"model_settings": 
    {
        "model_path": "/from_s3/model", 
        "model_type": "causal",
        "adapter_path": "/from_s3/adapters",
        "transformers_settings": {}, 
        "model_kwargs": {}
    }
```

**model_kwargs**  -- the place to specify something like "attn_implementation": "flash_attention_2" <br>
**model_type** -- For classification/RM "seq_cls" is needed, for most others "causal" is suitable<br>
**adapter_path** -- path to trained LORA-adapter if needed <br>

### Tokenizer Settings
Basic settings to load the tokenizer <br>
```json
"tokenizer_settings": 
    {
        "use_fast": false, 
        "tokenizer_path": "/from_s3/tokenizer"
    }
```
**use_fast** -- critical for some models

### Generation Settings 
**transformers_settings** -- standard model generation settings <br>
**custom_settings** -- useful for display. for example, you can choose to display or not display the prompt/special tokens

```json
"generation_settings": [
    {
        "transformers_settings": {
            "num_beams": 3,
            "max_new_tokens": 500,
            "stop_strings": ["<|eot_id|>", "<|end_of_text|>"],
            "repetition_penalty": 1.02
        },
        "custom_settings": {
            "skip_special_tokens": false
        }
    }
]
```
**stop_strings** -- varies between models, or you might have trained your own. You can use one/multiple tokens or strings.

### Dataset Settings
**sources** -- the name and path of the dataset. You can choose **num_samples** or **sample_rate** to control how much of the dataset to use <br>
**chat_settings** -- how your messages will be processed for input into the LLM
```json
"dataset_settings": {
    "sources": [
        {
            "name": "val",
            "records_path": "/from_s3/dataset/val_chat.jsonl", "sample_rate": 1
        }
    ],
    "prompt_template": {
        "role_tag_mapping": {
            "bot": "assistant",
            "user": "user",
            "system": "system"
        },
        "prefix_template": "<|start_header_id|>{role}<|end_header_id|>\n\n",
        "suffix_template": "<|eot_id|>"
    },
    "dataset_type": "chat",
    "max_tokens_count": 150
  }
```
**prompt_template** -- For each message from **dataset[’messages’][i]** which looks like {role: role, content: content}, we get a string of **[Prefix_template + content + Suffix_template]** then we combine all obtained strings into a single text.

**dataset_type** -- a important parameter that must match the type of method you want to run

## 🚀 Inference Stuff!

### Default Generation inference

- Let's look at the model+adapter ([default inference](../tests/fixtures/configs/inference/sft/base.json))

- Make sure that in **role_tag_mapping** you have all roles from your dataset, and they correspond to the tokens from your model.

###  vLLM Generation inference 
Suppose you want to use vLLM: for speeding up inference or CUDA_OUT_OF_MEMORY_ERROR👹:

- Then this config is suitable: [vllm inference](../tests/fixtures/configs/inference/sft/vllm_base.json)
- In fact, only 2 minor changes can requires: <br>
 (a) add **use_vllm: true** <br>
 (b) add **tensor_parallel_size: ...** if you want to split your model into multiple cards

### Troubleshooting:
- If the sizes of the dictionary in the base_model/adapter do not match, pay attention to `special_tokens_setter`, you might have duplicate special tokens after train.

### Classification inference:
Don't forget to specify that your model is for classification😏   [classification inference ](../tests/fixtures/configs/inference/classification/base.json)


```json
"model_settings": {
    "model_type": "seq_cls",
    "model_kwargs": {
            "num_labels": 2,
            "problem_type": "single_label_classification"
        }
    },
"dataset_type": "classification"
```



## 🏋️Train Stuff! <a name="train-stuff"></a>

#### embeddings_initialization_strategy:

```json
"embeddings_initialization_strategy": {
    "<RS>": "<s>", 
    "<super_bot>": "the best bot ever"
}
``` 
During training, you might want to add your special tokens.

In this case, we add the token **<RS>** by initializing its weights with the token **<bos>**, and the token "<super_bot>" by averaging the tokens that split the string "the best bot ever"

#### train_dataset_settings
similiar as inference_dataset but 
Pay attention to **"only_answer_loss": true**:
This parameter means the model will calculate the error only on the last message from dataset[’messages’]. In most cases, you want the last message to be from the role: **bot**, otherwise, you're training the model to mimic the user! 😏
```json
"keep_end": "bool"
```
 **CUT**: if keep_end=False -> [:max_tokens_count] elif keep_end=True -> [-max_tokens_count:]; <br>
 cuts off the last fully entered message in the dialogue. <br>

Only for cherry_pick_settings: <br>
if **random_cut = True**, then the end is chosen as a random bot message from messages.

### LORA Adapter train
Check out this config: [LORA Adapter](../tests/fixtures/configs/train/sft/base.json)


### P-tuning train 
No problem u can use
PrefixTuning|Lora|PromptTuning| PTuning <br>
Check this out: [P-Tuning](../tests/fixtures/configs/train/sft/prompt_tuning.json)<br>

```json
"peft_settings": {
    "name": "P_TUNING",
    "num_virtual_tokens": 32
}
```

### Custom Modifications
#### is model pay attention to specific doc?
We prepare the dataset in advance with separators **<doc_sep>** initialized simply as **<sep>** (or you could use 'Document' or whatever else you like).

#### any other creative idea you come up with:
- Prepare the dataset 🧑‍🔬
- Specify the metric 🔍
- Train/Watch 👀

### LLM for classification
[classification train](../tests/fixtures/configs/train/classification/base.json)

```json
{
    "dataset_type": "classification",
    "model_type": "seq_cls",
    "model_kwargs": {
            "num_labels": 2, 
            "return_dict": true, 
            "problem_type": "single_label_classification"
        },
    "peft_setting": {
        "task_type": "SEQ_CLS"
    }
}
```