## Dataset Descriptions

[All datasets](../tests/fixtures/datasets) are in **JSONL format**, where:


### Common Attributes
- `id`: str - A distinct identifier for each data entry, represented as a string.

### Dataset Types:
- [Chat Dataset](#-chat-dataset)
- [Pair Preferences Dataset](#-pair-preferences-dataset)
- [KTO Dataset](#-kto-dataset)
- [Sampling Dataset](#-sampling-dataset)
- [Multimodal Dataset ](#-multimodal-dataset) (⌛️ Work in progress...)
- [Classification Dataset](#-classification-dataset)
- [DPPO Dataset](#-ddpo-dataset) (⌛️ Work in progress...)



<a name="-chat-dataset"></a>
### Chat Dataset

- `messages`: `list[ChatMessage]` — This is a sequence of messages that make up the chat history. Each `ChatMessage` includes:
  - `role` - The participant's role in the conversation (e.g., `user` or `bot`).
  - `content` -  The textual content of the message.

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

- `context`: `list[ChatMessage]` — This is a sequence of messages that make up the chat history.
- `answer_w`: `ChatMessage` — The more preferable response.
- `answer_l`: `ChatMessage` — The less preferable response.

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
### KTO Dataset
- `context`: `list[ChatMessage]` — This is a sequence of messages that make up the chat history.
- `answer`: `ChatMessage` — The given response.
- `is_desirable`: `bool` —  Indicator if the provided response is considered as desirable or no.

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
## Sampling Dataset
- `messages`: `list[ChatMessage]` — This is a sequence of messages that make up the chat history.
- `answers`: `list[ChatInferenceOutput]` - A list of generated responses. Each `ChatInferenceOutput` is structured as:
  - `id`: `str` -  A unique identifier for the generated response.
  - `content`: `str` - The content of generated completion

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
## Multimodal Dataset
⌛️ in progress..




<a name="-classification-dataset"></a>
### Classification Dataset
- `messages`: `list[ChatMessage]` — This is a sequence of messages that make up the chat history.
- `label`: `int` — Label of provided chat history.

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



<a name="-ddpo-dataset"></a>
## For DDPO Dataset
⌛️ in progress..