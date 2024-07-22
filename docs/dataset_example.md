## Dataset Descriptions

[All datasets](../tests/fixtures/datasets) are in **JSONL format**, where:


### Common Attributes
- `id`: str - a unique string identifier

### Dataset Types:
- [Chat Dataset](#-chat-dataset)
- [Pair Preferences Dataset](#-pair-preferences-dataset)
- [KTO Dataset](#-kto-dataset)
- [Sampling Dataset](#-sampling-dataset)
- [Multimodal Dataset](#-multimodal-dataset)
- [Classification Dataset](#-classification-dataset)
- [DPPO Dataset](#-ddpo-dataset)



<a name="-chat-dataset"></a>
### For Chat Dataset

- `messages`: `list[ChatMessage]` — chat history, where `ChatMessage` is structured as:
  - `role` - the role of the speaker (e.g., user, bot).
  - `content` - the message content.

Example:
```json
{
  "id": 0,
  "source": "example",
  "messages": [
    {"role": "user", "content": "Can you play chess?"},
    {"role": "bot", "content": "Yes, of course"}
  ]
}
```



<a name="-pair-preferences-dataset"></a>
### Pair Preferences Dataset

- `context`: `list[ChatMessage]` — chat history
- `answer_w`: `ChatMessage` — the good response
- `answer_l`: `ChatMessage` — the poor response

Example:
```json
{
  "id": 0,
  "source": "example", 
  "context": [
    {"role": "user", "content": "Can you play chess?"}
  ],
  "answer_w": {"role": "bot", "content": "Yes, of course"},
  "answer_l": {"role": "bot", "content": "Get out, I don't want to talk to you!"}
  }
```



<a name="-kto-dataset"></a>
### For KTO Dataset
- `context`: `list[ChatMessage]` — chat history
- `answer`: `ChatMessage` — response
- `is_desirable`: `bool` — whether the response is desirable

Example:
```json
{
  "id": 0,
  "source": "example",
  "context": [
    {"role": "user", "content": "Can you play chess?"}
  ],
  "answer": {"role": "bot", "content": "Yes, of course"},
  "is_desirable": true
}
{
  "id": 1,
  "source": "example",
  "context": [
    {"role": "user", "content": "Can you play chess?"}
  ],
  "answer": {"role": "bot", "content": "Get out, I don't want to talk to you!"},
  "is_desirable": false
}
```



<a name="-sampling-dataset"></a>
## For Sampling Dataset
- `messages`: `list[ChatMessage]` — chat history
- `answers`: `list[ChatInferenceOutput]` - list of answers, where `ChatInferenceOutput` is structured as:
  - `id` : The number or id of generated completion
  - `content` The content of generated completion

Example:
```json
{
  "id": "0", 
  "dataset_name": "example", 
  "messages": [
    {"role": "user", "content": "hi"}, 
    {"role": "bot", "content": "hi"}, 
    {"role": "user", "content": "how are you"}
  ], 
  "answers": [
    {"content": "good", "id": "0"}, 
    {"content": "not bad", "id": "1"}
  ]
}
```



<a name="-multimodal-dataset"></a>
## For Multimodal Dataset
⌛️ in progress..
examples [here](../tests/fixtures/datasets/multimodal)



<a name="-classification-dataset"></a>
### For Classification Dataset
- `messages`: `list[ChatMessage]` — chat history
- `label`: `int` — class labels

Example: 
```json
{
  "id": 0,
  "source": "example",
  "messages": [
    {"role": "user", "content": "Can you play chess?"},
    {"role": "bot", "content": "Yes, of course"}
  ],
  "label": 1
}
{
  "id": 1,
  "source": "example",
  "messages": [
    {"role": "user", "content": "Can you play chess?"},
    {"role": "bot", "content": "Get out, I don't want to talk to you!"}
  ],
  "label": 0
}
```



## For DDPO Dataset
⌛️ in progress..