import argparse
import os
import pickle
import threading
import time
import warnings

import numpy as np
import torch
import vllm
from flask import Flask, Response, request
from transformers import AutoTokenizer

from turbo_alignment.common.tf.loaders.model.model import load_model
from turbo_alignment.common.vllm_utils import vllm_generations_postprocess
from turbo_alignment.settings import pipelines as pipeline_settings

warnings.filterwarnings("ignore")

app = Flask(__name__)
step = 0


def compare_sampling_params(param1, param2):
    return vars(param1) == vars(param2)


@app.route("/generate", methods=["POST"])
def generate():
    try:
        data = pickle.loads(request.data)
    except pickle.UnpicklingError:
        return Response("Invalid data format", status=400)
    response_event = threading.Event()
    request_entry = {"data": data, "response_event": response_event, "response": None}

    with condition:
        request_queue.append(request_entry)
        if len(request_queue) >= N:
            condition.notify_all()
        else:
            condition.wait()

    response_event.wait()
    return Response(request_entry["response"], content_type="application/octet-stream")


def save_rewards(rewards, step):
    np.save(
        os.path.join(rewards_savedir, f"rewards_{step}_on_port_{args.port}.npy"),
        rewards.to(torch.float32).cpu().numpy(),
    )


@torch.inference_mode()
def batch_process():
    global request_queue, step  # Add this line
    while True:
        with condition:
            while len(request_queue) < N:
                condition.wait()

            batch_data = request_queue[:N]
            request_queue = request_queue[N:]
            condition.notify_all()

        # Validate input data
        if not all("prompt_token_ids" in entry["data"] for entry in batch_data):
            continue  # Skip this batch if validation fails

        prompt_token_ids_list = []
        for entry in batch_data:
            prompt_token_ids_list.extend(entry["data"].get("prompt_token_ids", []))

        sampling_params_list = [
            entry["data"].get("sampling_params", "") for entry in batch_data
        ]
        lora_dir_list = [entry["data"].get("lora_dir", []) for entry in batch_data]
        max_length_list = [
            entry["data"].get("max_length", None) for entry in batch_data
        ]
        lora_id_list = [entry["data"].get("lora_id", None) for entry in batch_data]
        train_eval_list = [
            entry["data"].get("train_eval", None) for entry in batch_data
        ]

        assert all(
            vars(param) == vars(sampling_params_list[0])
            for param in sampling_params_list
        ), f"Not all sampling_params are the same: {sampling_params_list}"
        assert all(
            dir == lora_dir_list[0] for dir in lora_dir_list
        ), f"Not all lora_dirs are the same: {lora_dir_list}"
        assert all(
            max_length == max_length_list[0] for max_length in max_length_list
        ), f"Not all max_lengths are the same: {max_length_list}"
        assert all(
            lora_id == lora_id_list[0] for lora_id in lora_id_list
        ), f"Not all lora_ids are the same: {lora_id_list}"
        assert all(
            train_eval == train_eval_list[0] for train_eval in train_eval_list
        ), f"Not all train_evals are the same: {train_eval_list}"

        # no need to generate until the 'max_tokens' is reached
        # what important is that the sum length of prompt and answer is less than 'max_tokens_count'
        sampling_params = sampling_params_list[0]
        sampling_params.max_tokens = min(
            sampling_params.max_tokens,
            max_length_list[0] - min([len(x) for x in prompt_token_ids_list]),
        )
        print("num_prompts", len(prompt_token_ids_list))
        t0 = time.time()
        generations = generating_policy.generate(
            prompts=None,
            prompt_token_ids=prompt_token_ids_list,
            sampling_params=sampling_params,
            lora_request=vllm.lora.request.LoRARequest(
                lora_dir_list[0], lora_id_list[0], lora_dir_list[0]
            ),
        )
        t1 = time.time()
        gen_time = t1 - t0
        # pad_to_max_len because I use torch.compile and I don't want
        query_responses, attention_mask, response_tokens_mask, position_ids = (
            vllm_generations_postprocess(tokenizer, generations, max_length_list[0])
        )
        query_responses, attention_mask, response_tokens_mask, position_ids = (
            query_responses.cuda(),
            attention_mask.cuda(),
            response_tokens_mask.cuda(),
            position_ids.cuda(),
        )

        t2 = time.time()
        postprocess_time = t2 - t1

        # the whole batch doesn't fit into memory
        rewards = []
        mini_batch_size = len(query_responses) // 2
        for i in range(0, len(query_responses), mini_batch_size):
            mini_batch_query_responses = query_responses[i : i + mini_batch_size]
            mini_batch_attention_mask = attention_mask[i : i + mini_batch_size]
            mini_batch_response_mask = response_tokens_mask[i : i + mini_batch_size]
            mini_batch_position_ids = position_ids[i : i + mini_batch_size]

            mini_batch_rewards = reward_model(
                input_ids=mini_batch_query_responses,
                attention_mask=mini_batch_attention_mask,
                response_mask=mini_batch_response_mask,
                position_ids=mini_batch_position_ids,
            )["logits"]

            rewards.append(mini_batch_rewards.cpu())

        rewards = torch.cat(rewards, dim=0)

        if train_eval_list[0] == "train":
            save_rewards(rewards, step)
            step += 1

        t3 = time.time()
        reward_time = t3 - t2

        # Group generations correctly
        response_index = 0
        for entry in batch_data:
            subbatch_size = (
                len(entry["data"].get("prompt_token_ids", []))
                * entry["data"].get("sampling_params", 1).n
            )

            entry["response"] = pickle.dumps(
                (
                    query_responses[
                        response_index : response_index + subbatch_size
                    ].cpu(),
                    attention_mask[
                        response_index : response_index + subbatch_size
                    ].cpu(),
                    response_tokens_mask[
                        response_index : response_index + subbatch_size
                    ].cpu(),
                    position_ids[response_index : response_index + subbatch_size].cpu(),
                    rewards[response_index : response_index + subbatch_size].cpu(),
                    gen_time,
                    postprocess_time,
                    reward_time,
                )
            )
            response_index += subbatch_size
            entry["response_event"].set()

        with condition:
            condition.notify_all()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VLLM Server Batched")
    parser.add_argument(
        "--experiment_settings_path",
        type=str,
        help="Path to the experiment settings file",
    )
    parser.add_argument(
        "--total_num_gpus",
        type=int,
        help="Number of gpus used for training",
    )
    parser.add_argument(
        "--num_servers",
        type=int,
        help="Number of gpus dedicated for generation server",
    )
    parser.add_argument(
        "--port",
        type=int,
        help="Server Port",
    )
    args = parser.parse_args()

    # num available gpus
    num_gpus = torch.cuda.device_count()
    print(f"Number of available GPUs: {num_gpus}")

    N = (args.total_num_gpus - args.num_servers) // args.num_servers
    assert (args.total_num_gpus - args.num_servers) % args.num_servers == 0, (
        args.total_num_gpus,
        args.num_servers,
    )
    print("NUM_GPUS for policy training", N)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    policy_path = "/from_s3/models/policy"
    generating_policy = vllm.LLM(
        model=policy_path,
        dtype="bfloat16",
        enable_lora=True,
        max_lora_rank=64,
        skip_tokenizer_init=True,
        max_model_len=2048,
        max_seq_len_to_capture=2048,
        gpu_memory_utilization=0.75,
    )
    tokenizer = AutoTokenizer.from_pretrained(policy_path)

    experiment_settings = pipeline_settings.REINFORCETrainExperimentSettings.parse_file(
        args.experiment_settings_path
    )
    rewards_savedir = os.path.join(str(experiment_settings.log_path), "rewards")
    os.makedirs(rewards_savedir, exist_ok=True)

    reward_model = (
        load_model(
            model_settings=experiment_settings.reward_model_settings,
            tokenizer=tokenizer,
        )
        .cuda()
        .eval()
    )
    # reward_model.model = torch.compile(reward_model.model)

    request_queue = []
    condition = threading.Condition()

    threading.Thread(target=batch_process, daemon=True).start()
    app.run(host="0.0.0.0", port=args.port, threaded=True)
