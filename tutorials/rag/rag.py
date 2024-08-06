import os
import random

import dotenv
import torch

assert dotenv.load_dotenv()

GENERATOR_MODEL = 'NousResearch/Hermes-2-Theta-Llama-3-8B'
ENCODER_MODEL = 'abacaj/llama-161M-100B'

from peft import get_peft_config
from peft.peft_model import PeftModelForCausalLM
from transformers import (
    AdamW,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    TrainingArguments,
)

from turbo_alignment.cherry_picks.rag import RagCherryPickCallback
from turbo_alignment.dataset.chat.chat import ChatDataset
from turbo_alignment.settings.datasets.base import (
    DatasetSourceSettings,
    DatasetStrategy,
)
from turbo_alignment.settings.datasets.chat import ChatDatasetSettings
from turbo_alignment.settings.pipelines.train.rag import RAGTrainExperimentSettings

tokenizer = AutoTokenizer.from_pretrained(GENERATOR_MODEL)

from datasets import Value, load_dataset

from turbo_alignment import pipelines
from turbo_alignment.dataset.registry import DatasetRegistry
from turbo_alignment.settings import pipelines as pipeline_settings

ds_qap = load_dataset('rag-datasets/rag-mini-bioasq', 'question-answer-passages')
ds_corpus = load_dataset('rag-datasets/rag-mini-bioasq', 'text-corpus')

from transformers import AutoModelForCausalLM

question_encoder = AutoModelForCausalLM.from_pretrained(ENCODER_MODEL)
question_encoder.eval()
question_tokenizer = AutoTokenizer.from_pretrained(ENCODER_MODEL)

from turbo_alignment.modeling.rag.utils import get_question_embeddings


def save_index(num_passages=1000):
    def embed(sentence):
        with torch.no_grad():
            input_ids = question_tokenizer.encode(sentence, return_tensors='pt')
            attention_mask = torch.ones_like(input_ids)
            encoder_output = question_encoder(input_ids, attention_mask=attention_mask, output_hidden_states=True)
            embedding = get_question_embeddings(encoder_output, attention_mask)
            embedding = embedding.reshape(-1).numpy()

        return embedding

    if num_passages is not None:
        passages_with_embeddings = ds_corpus['passages'].select(range(num_passages))
    else:
        passages_with_embeddings = ds_corpus['passages']

    passages_with_embeddings = passages_with_embeddings.map(lambda example: {'embeddings': embed(example['passage'])})

    passages_with_embeddings.add_faiss_index(column='embeddings')

    print(passages_with_embeddings.get_nearest_examples('embeddings', embed('I am happy.'), k=10))

    passages_with_embeddings.save_faiss_index('embeddings', 'my_index.faiss')

    passages_with_embeddings.drop_index('embeddings')
    passages_with_embeddings = passages_with_embeddings.rename_column('id', 'title').rename_column('passage', 'text')

    features = passages_with_embeddings.features.copy()
    features['title'] = Value('string')
    passages_with_embeddings = passages_with_embeddings.cast(features)
    passages_with_embeddings.save_to_disk('passages')


# save_index()

model_settings_json = {
    'generator_settings': {
        'model_path': GENERATOR_MODEL,
        'model_type': 'causal',
        'transformers_settings': {},
        'embeddings_initialization_strategy': {
            '<|begin_of_text|>': '<s>',
            '<|end_of_text|>': '</s>',
            '<bot>': 'bot',
            '<user>': 'user',
            '<system>': 'system',
        },
        'peft_settings': {
            'r': 4,
            'lora_alpha': 16,
            'lora_dropout': 0.05,
            'target_modules': ['q_proj', 'v_proj', 'k_proj', 'o_proj'],
            'task_type': 'CAUSAL_LM',
            'modules_to_save': [],
            'name': 'LORA',
        },
    },
    'question_encoder_settings': {
        'model_path': ENCODER_MODEL,
        'model_type': 'encoder',
        'transformers_settings': {},
        'embeddings_initialization_strategy': {},
    },
    'index_settings': {'index_path': 'my_index.faiss', 'passages_path': 'passages'},
    'retrieval_settings': {'n_docs': 2, 'max_doc_length': 256, 'query_encoder_max_length': 128},
}

from turbo_alignment.modeling.rag.rag_model import RagSequenceForGeneration
from turbo_alignment.modeling.rag.rag_tokenizer import RagTokenizer
from turbo_alignment.settings.rag_model import RAGPreTrainedModelSettings

model_settings = RAGPreTrainedModelSettings.model_validate(model_settings_json)

quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
generator = AutoModelForCausalLM.from_pretrained(GENERATOR_MODEL, quantization_config=quantization_config)

is_train = True

if is_train:
    peft_config = model_settings.generator_settings.peft_settings.dict()
    peft_config['peft_type'] = peft_config['name']
    del peft_config['name']

    peft_config = get_peft_config(peft_config)
    generator = PeftModelForCausalLM(generator, peft_config)

rag_tokenizer = RagTokenizer(model_settings, tokenizer_path=None)

rag_model = RagSequenceForGeneration(model_settings, generator, question_encoder, rag_tokenizer)

records = []
for example in ds_qap['test']:
    messages = [{'role': 'user', 'content': example['question']}, {'role': 'bot', 'content': example['answer']}]
    record = {
        'messages': messages,
        'meta': {'relevant_passage_ids': example['relevant_passage_ids']},
        'id': example['id'],
    }

    records.append(record)

random.seed(0)
random.shuffle(records)
train_num_samples = int(0.8 * len(records))
train_records = records[:train_num_samples]
val_records = records[train_num_samples:]
val_records = val_records[:10]


def records_to_dataset(records: list, strategy: str, tokenizer: PreTrainedTokenizerBase):
    dataset_cls = DatasetRegistry.by_name('chat').by_name(strategy)

    source = DatasetSourceSettings(name='rag-mini-bioasq', records_data=records, num_samples=len(records))

    chat_dataset_settings_dict = {
        'prompt_template': {
            'role_tag_mapping': {'bot': '<bot>', 'user': '<user>', 'system': '<system>'},
            'prefix_template': '<|im_start|>{role}\n',
            'suffix_template': '<|im_end|>',
        },
        'dataset_type': 'chat',
        'max_tokens_count': None,
        'only_answer_loss': True,
    }
    chat_dataset_settings = ChatDatasetSettings(**chat_dataset_settings_dict)

    dataset = dataset_cls(source=source, settings=chat_dataset_settings, tokenizer=tokenizer, read=True)

    return dataset


def evaluate(val_records, rag_model):
    from turbo_alignment.cherry_picks.rag import RagCherryPickCallback

    inference_chat_dataset = records_to_dataset(val_records, 'inference', tokenizer)

    cherry_pick_settings_dict = {
        'generator_transformers_settings': {
            'num_beams': 1,
            'max_new_tokens': 256,
            'do_sample': False,
        },
        'custom_generation_settings': {'generation_eos_token': '<|im_end|>', 'skip_special_tokens': True},
        'dataset_settings': {
            'sources': [
                {
                    'name': 'support',
                    'records_path': 'tests/fixtures/datasets/chat/train_chat_rag.jsonl',
                    'num_samples': 1,
                }
            ],
            'prompt_template': {
                'role_tag_mapping': {'bot': '<bot>', 'user': '<user>', 'system': '<system>'},
                'prefix_template': '<|im_start|>{role}',
                'suffix_template': '<|im_end|>',
            },
            'dataset_type': 'chat',
            'max_tokens_count': 20000,
            'random_cut': True,
            'only_answer_loss': False,
        },
        'metric_settings': [],
    }

    from turbo_alignment.settings.cherry_pick import ChatCherryPickSettings

    cherry_pick_settings = ChatCherryPickSettings.model_validate(cherry_pick_settings_dict)

    from turbo_alignment.metrics.metric import Metric

    rag_cherry_pick = RagCherryPickCallback(cherry_pick_settings, datasets=[inference_chat_dataset], metrics=[])

    output = rag_cherry_pick.on_evaluate(None, None, None, tokenizer=tokenizer, model=rag_model)


if not is_train:
    evaluate(val_records, rag_model)


def train(train_records, val_records):
    # Training Args
    kwargs = {
        'output_dir': 'train_rag_output',
        'evaluation_strategy': 'epoch',
        'save_strategy': 'epoch',
        'per_device_train_batch_size': 2,
        'per_device_eval_batch_size': 1,
        'gradient_accumulation_steps': 1,
        'eval_steps': 150,
        'save_steps': 150,
        'logging_steps': 1,
        'learning_rate': 0.0004,
        'num_train_epochs': 1,
        'max_steps': 200,
        'lr_scheduler_type': 'linear',
        'lr_scheduler_kwargs': {},
        'warmup_steps': 0,
        'warmup_ratio': 0.1,
        'fp16': True,
        'bf16': False,
        'tf32': False,
        'torch_compile': False,
        'optim': 'adamw_torch',
        'adam_beta1': 0.9,
        'adam_beta2': 0.98,
        'adam_epsilon': 1e-06,
        'weight_decay': 0.01,
        'max_grad_norm': 0.11,
        'deepspeed': None,
        'save_total_limit': 1,
        'save_only_model': False,
        'no_cuda': False,
        'prediction_loss_only': False,
        'load_best_model_at_end': True,
        'logging_first_step': True,
        'fsdp_config': None,
        'fsdp': '',
        'dataloader_num_workers': 8,
        'dataloader_prefetch_factor': None,
        'dataloader_persistent_workers': False,
        'dataloader_pin_memory': True,
        'gradient_checkpointing': False,
        'gradient_checkpointing_kwargs': {},
        'neftune_noise_alpha': None,
        'report_to': [],
    }
    training_args = TrainingArguments(**kwargs)

    training_args._n_gpu = 1

    # create train dataset and val dataset

    train_dataset = records_to_dataset(train_records, 'train', tokenizer)
    val_dataset = records_to_dataset(val_records, 'inference', tokenizer)

    # create trainer
    from transformers.data.data_collator import (
        DataCollatorForSeq2Seq,
        DataCollatorForTokenClassification,
    )

    data_collator = DataCollatorForTokenClassification(tokenizer, pad_to_multiple_of=8)

    from turbo_alignment.trainers.multigpu import MultiGPUCherryPicksTrainer

    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    trainer = MultiGPUCherryPicksTrainer(
        model=rag_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        callbacks=[],
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    generator_parameters = rag_model.rag.generator.parameters()
    question_encoder_parameters = rag_model.rag.question_encoder.parameters()
    optimizer = AdamW(
        [
            {'params': generator_parameters, 'lr': training_args.learning_rate / 1000},
            {
                'params': question_encoder_parameters,
                'lr': training_args.learning_rate * 10,
            },
        ]
    )
    trainer.optimizer = optimizer

    trainer.train()

    return trainer.model


if is_train:
    rag_model = train(train_records, val_records)

    evaluate(val_records, rag_model)
