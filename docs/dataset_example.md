## Dataset Descriptions

[All datasets](tests/fixtures/datasets) are in **JSONL format**, where:



### Common Attributes:
- **id** — A unique integer identifier.
- **source** — A convenient string identifier for users.

### Dataset Types:
- chat
- pair_preferences
- sampling
- classification
- multimodal


### For Chat:

- **messages** — A list of `chatmessage` where `chatmessage` is structured as:
  - **role**: The role of the speaker (e.g., user, bot).
  - **content**: The message content.

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

### For DPO/KTO (paired) — the format is the same everywhere:

- context: list[chatmessage] — chat history
- answer_winning: chatmessage — the good response
- answer_losing: chatmessage — the poor response
Example:
```json
{"id": 0, "source": "example", 
"context": [{"role": "user", "content": "Can you play chess?"}],
"answer_winning": {"role": "bot", "content": "Yes, of course"},
"answer_losing":{"role": "bot", "content": "Get out, I don't want to talk to you!"}
  }
```



### For KTO (unpaired):
-*context: list[chatmessage] — chat history
- answer: chatmessage — chatmessage
-  is_desirable: bool — whether the response is desirable

    context: list[chatmessage]
    answer: chatmessage
    is_desirable: bool — whether the response is desirable
```json
{
  "id": 0,
  "source": "example",
  "context": [{"role": "user", "content": "Can you play chess?"}],
  "answer": {"role": "bot", "content": "Yes, of course"},
  "is_desirable": true
}
{
  "id": 1,
  "source": "example",
  "context": [{"role": "user", "content": "Can you play chess?"}],
  "answer": {"role": "bot", "content": "Get out, I don't want to talk to you!"},
  "is_desirable": false
}

```

## For SAMPLING:
- **messages** — A list of `chatmessage` where `chatmessage` is structured as:
  - **role**: The role of the speaker (e.g., user, bot).
  - **content**: The message content.

- **answers** - A list of `ChatInferenceOutput` where `ChatInferenceOutput` is structured as:
  - **id** : The number or id of generated completion
  - **content**: The content of generated completion


```json
{"id": "0", "messages": [{"role": "user", "content": "hi", "disable_loss": false}, {"role": "bot", "content": "hi", "disable_loss": false}, {"role": "user", "content": "how are you", "disable_loss": false}], "label": null, "dataset_name": "chat_test", "answers": [{"content": "content", "id": "0"}, {"content": "lol", "id": "1"}]}
{"id": "1", "messages": [{"role": "user", "content": "hi", "disable_loss": false}, {"role": "bot", "content": "hi", "disable_loss": false}, {"role": "user", "content": "how are you", "disable_loss": false}, {"role": "bot", "content": "bad", "disable_loss": false}], "label": null, "dataset_name": "chat_test", "answers": [{"content": "content", "id": "0"}, {"content": "lol", "id": "1"}]}
```


## For MULTIMODAL:
examples [here](tests/fixtures/datasets/multimodal)
