from tempfile import TemporaryDirectory

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from turbo_alignment.dataset.pair_preferences import PairPreferenceDataCollator
from turbo_alignment.settings.pipelines.train.dpo import (
    DPOLossesType,
    SigmoidLossSettings,
    SyncRefModelSettings,
)
from turbo_alignment.trainers.dpo import DPOTrainer, DPOTrainingArguments


def test_dpo_trainer(dpo_dataset):
    model_path = 'tests/fixtures/models/llama2_tiny'

    model = AutoModelForCausalLM.from_pretrained(model_path)

    ref_model = AutoModelForCausalLM.from_pretrained(model_path)

    with TemporaryDirectory() as tmp_dir:
        args = DPOTrainingArguments(
            do_train=True,
            loss_settings=SigmoidLossSettings(loss_type=DPOLossesType.SIGMOID).dict(),
            sync_ref_settings=SyncRefModelSettings().dict(),
            do_eval=False,
            learning_rate=1.0e-4,
            use_cpu=True,
            num_train_epochs=10,
            report_to=[],
            remove_unused_columns=False,
            output_dir=tmp_dir,
        )

        tokenizer = AutoTokenizer.from_pretrained(model_path)

        data_collator = PairPreferenceDataCollator(tokenizer=tokenizer)

        trainer = DPOTrainer(
            model=model,
            ref_model=ref_model,
            args=args,
            train_dataset=dpo_dataset,
            eval_dataset=dpo_dataset,
            data_collator=data_collator,
        )

        batch = data_collator(list(dpo_dataset))

        loss_before, _ = trainer.get_batch_metrics(model, batch, 'train')
        trainer.train()
        loss_after, _ = trainer.get_batch_metrics(trainer.model, batch, 'train')

        assert torch.greater(loss_before, loss_after)

        initial_model = AutoModelForCausalLM.from_pretrained(model_path)

        trainer.save_model(tmp_dir)
        trained_model = AutoModelForCausalLM.from_pretrained(tmp_dir)

        initial_state_dict = initial_model.state_dict()
        trained_state_dict = trained_model.state_dict()
        for k, v in trained_state_dict.items():
            assert any(torch.not_equal(v, initial_state_dict[k]).tolist())

        ref_model_state_dict = trainer.ref_model.state_dict()
        for k, v in ref_model_state_dict.items():
            assert all(torch.eq(v, initial_state_dict[k]).tolist())
