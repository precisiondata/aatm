import os
import torch
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    TrainerCallback,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, TaskType
from torch.profiler import profile, ProfilerActivity
import torch.nn.functional as F
from dvclive import Live
from transformers.integrations import DVCLiveCallback


# Setup model and tokenizer
def setup_model_and_tokenizer(
    model_id, use_lora=True, lora_rank=16, pretrain=False, debug=0, seed=None
):
    """Setup model and tokenizer with optional LoRA"""

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    assert (
        tokenizer.chat_template is not None
    ), f"{model_id} does not have chat template."

    # Add special tokens if not present
    special_tokens = [
        "<|predictor_1|>",
        "<|predictor_2|>",
        "<|predictor_3|>",
        "<|predictor_4|>",
        "<|predictor_5|>",
        "<|predictor_6|>",
        "<|predictor_7|>",
        "<|predictor_8|>",
        "<|predictor_9|>",
        "<|predictor_10|>",
        "<|start_header_id|>",
        "<|end_header_id|>",
        "<|eot_id|>",
        "<|perception|>",
    ]
    new_tokens = [token for token in special_tokens if token not in tokenizer.vocab]

    if new_tokens:
        tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})

    # Set pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model with better device mapping for multi-GPU
    device_map = None
    if torch.cuda.is_available():
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        if world_size == 1:
            device_map = "auto"
        else:
            # For multi-GPU with torchrun, don't use device_map
            device_map = None

    if pretrain:
        if seed is not None:
            torch.manual_seed(seed)
        config = AutoConfig.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_config(
            config,
            torch_dtype=torch.bfloat16,
        )
        rank = torch.distributed.get_rank()
        device = torch.device(f"cuda:{rank}")
        model.to(device)
        for p in model.parameters():
            torch.distributed.broadcast(p.data, src=0)
        for b in model.buffers():
            torch.distributed.broadcast(b.data, src=0)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=torch.bfloat16,
            device_map=device_map,
            trust_remote_code=True,
            # Add these for better multi-GPU stability
            low_cpu_mem_usage=True,
            use_cache=False,  # Disable KV cache for training
        )

    # Resize embeddings if we added new tokens
    if new_tokens:
        model.resize_token_embeddings(len(tokenizer))

    # Setup LoRA if requested
    if use_lora:
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=lora_rank,
            lora_alpha=lora_rank * 2,
            lora_dropout=0.1,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        )
        model = get_peft_model(model, lora_config)
        model.enable_input_require_grads()
        if torch.cuda.current_device() == 0:
            model.print_trainable_parameters()

    return model, tokenizer


class ProfilerFLOPCallback(TrainerCallback):
    def __init__(self, profile_steps=10):
        self.profile_steps = profile_steps
        self.total_flops = 0

    def on_step_begin(self, args, state, control, **kwargs):
        if state.global_step < self.profile_steps:
            self.profiler = profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                record_shapes=True,
                with_flops=True,  # This enables FLOP counting if available
            )
            self.profiler.__enter__()

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step < self.profile_steps:
            self.profiler.__exit__(None, None, None)

            # Extract FLOP information
            events = self.profiler.key_averages()
            step_flops = sum(event.flops for event in events if event.flops > 0)
            self.total_flops += step_flops

            if (
                torch.cuda.current_device() == 0
            ):  # and (state.global_step == 63 or state.global_step % 10 == 0):
                print(f"Step {state.global_step}: FLOPs: {step_flops:,.0f}")


class RepresentationTrainer(Trainer):
    """
    Trainer to regularize representations.
    """

    def __init__(self, *args, **kwargs):
        # Extract custom loss parameters
        self.lbd = kwargs.pop("lbd", 1.0)
        self.gamma = kwargs.pop("gamma", 1.0)
        self.last_token = kwargs.pop("last_token", -2)
        self.debug = kwargs.pop("debug", 0)
        self.additive_mask = kwargs.pop("additive_mask", False)
        self.jepa_l2 = kwargs.pop("jepa_l2", False)
        self.jepa_mse = kwargs.pop("jepa_mse", False)
        self.infonce = kwargs.pop("infonce", False)
        self.jepa_ratio = kwargs.pop("jepa_ratio", -1.0)
        assert (
            self.jepa_l2 + self.jepa_mse <= 1
        ), "Only one of jepa_l2 and jepa_mse can be True."
        super().__init__(*args, **kwargs)

    def _last_token_index(self, input_ids, labels, attention_mask):
        index = []

        def unpad(input_ids, attention_mask):
            result = []
            can_break = False
            for id, mask in zip(input_ids, attention_mask):
                if mask != 0:
                    can_break = True
                if mask == 0 and can_break:
                    break
                result.append(id)
            return result

        for i in range(input_ids.shape[0]):
            uii = unpad(input_ids[i], attention_mask[i])

            index.append(len(uii) + self.last_token)

        index_tensor = torch.tensor(index).to(input_ids.device)

        return index_tensor

    def _build_additive_mask(self, k: int):
        mask = torch.zeros((k, k), dtype=torch.float32)
        mask[torch.triu(torch.ones(k, k), diagonal=1) == 1] = -torch.inf
        return mask

    def build_with_additive_mask(self, inputs):
        if self.jepa_ratio > 0.0:
            if torch.rand(1).item() > self.jepa_ratio:
                return {
                    "input_ids": inputs["input_ids"],
                    "labels": inputs["labels"],
                    "attention_mask": inputs["attention_mask"],
                }, True
        batch_size = inputs["input_ids"].shape[0]
        seq_length = inputs["input_ids"].shape[-1]
        device = inputs["input_ids"].device

        # mask shape: [batch_size * 2, 1, seq_length, seq_length]
        mask = torch.full((batch_size * 2, 1, seq_length, seq_length), -torch.inf).to(
            device
        )

        # last_token shape: [batch_size * 2]
        last_token = self._last_token_index(
            inputs["input_ids"], inputs["labels"], inputs["attention_mask"]
        )
        last_token_user = self._last_token_index(
            inputs["input_ids_user"],
            inputs["labels_user"],
            inputs["attention_mask_user"],
        )
        last_token_assistant = self._last_token_index(
            inputs["input_ids_assistant"],
            inputs["labels_assistant"],
            inputs["attention_mask_assistant"],
        )

        for i in range(inputs["input_ids_user"].shape[0]):
            length = last_token[i] + 1
            length_user = last_token_user[i] + 1
            length_assistant = last_token_assistant[i] + 1

            inputs["input_ids_user"][
                i, length_user : length_user + length_assistant
            ] = inputs["input_ids_assistant"][i, :length_assistant]
            inputs["labels_user"][i, length_user : length_user + length_assistant] = (
                inputs["labels_assistant"][i, :length_assistant]
            )

            mask[i, :, 0:length, 0:length] = self._build_additive_mask(length)
            mask[i + batch_size, :, 0:length_user, 0:length_user] = (
                self._build_additive_mask(length_user)
            )
            mask[
                i + batch_size,
                :,
                length_user : length_user + length_assistant,
                length_user : length_user + length_assistant,
            ] = self._build_additive_mask(length_assistant)
        self._last_token_user = last_token_user
        self._last_token_assistant = last_token_assistant + last_token_user + 1
        return {
            "input_ids": torch.cat(
                [inputs["input_ids"], inputs["input_ids_user"]], dim=0
            ),
            "labels": torch.cat([inputs["labels"], inputs["labels_user"]], dim=0),
            "attention_mask": mask,
        }, False

    def forward(self, model, inputs):
        """
        Custom forward pass that handles all model calls.
        """
        # Main forward pass for language modeling
        if self.additive_mask:
            llm_inputs, skip_jepa = self.build_with_additive_mask(inputs)
        else:
            llm_inputs = {
                "input_ids": torch.cat(
                    [
                        inputs["input_ids"],
                        inputs["input_ids_user"],
                        inputs["input_ids_assistant"],
                    ],
                    dim=0,
                ),
                "labels": torch.cat(
                    [
                        inputs["labels"],
                        inputs["labels_user"],
                        inputs["labels_assistant"],
                    ],
                    dim=0,
                ),
                "attention_mask": torch.cat(
                    [
                        inputs["attention_mask"],
                        inputs["attention_mask_user"],
                        inputs["attention_mask_assistant"],
                    ],
                    dim=0,
                ),
            }

        with torch.set_grad_enabled(True):
            outputs = model(**llm_inputs, output_hidden_states=True)

        if self.additive_mask:
            if skip_jepa:
                user_hidden_states = None
                assistant_hidden_states = None
            else:
                batch_size = llm_inputs["input_ids"].shape[0] // 2
                user_hidden_states = outputs.hidden_states[-1][
                    batch_size : batch_size * 2
                ]
                assistant_hidden_states = user_hidden_states
        else:
            batch_size = llm_inputs["input_ids"].shape[0] // 3
            user_hidden_states = outputs.hidden_states[-1][batch_size : batch_size * 2]
            assistant_hidden_states = outputs.hidden_states[-1][batch_size * 2 :]

        # Return all outputs needed for loss computation
        return {
            "main_outputs": outputs,
            "user_hidden_states": user_hidden_states,
            "assistant_hidden_states": assistant_hidden_states,
        }

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        """
        Compute loss with additional regularization terms.
        """
        # Get indices
        if not self.additive_mask:
            index_user = self._last_token_index(
                inputs["input_ids_user"],
                inputs["labels_user"],
                inputs["attention_mask_user"],
            )
            index_assistant = self._last_token_index(
                inputs["input_ids_assistant"],
                inputs["labels_assistant"],
                inputs["attention_mask_assistant"],
            )
        first_dim = inputs["input_ids_user"].shape[0]

        # Get all forward pass results
        forward_results = self.forward(model, inputs)

        # Extract main language modeling loss
        main_outputs = forward_results["main_outputs"]
        lm_loss = main_outputs.loss

        # Compute representation similarity loss
        user_hidden_states = forward_results["user_hidden_states"]
        assistant_hidden_states = forward_results["assistant_hidden_states"]

        # Get embeddings (using last token of each sequence)
        if user_hidden_states is not None:
            if self.additive_mask:
                index_user = self._last_token_user
                index_assistant = self._last_token_assistant
            user_embedding = user_hidden_states[range(first_dim), index_user, :]
            assistant_embedding = assistant_hidden_states[
                range(first_dim), index_assistant, :
            ]

            # Compute cosine similarity
            cosine_similarity = F.cosine_similarity(
                user_embedding, assistant_embedding, dim=-1
            )
            if self.debug == 1 and torch.cuda.current_device() == 0:
                print(user_embedding.shape, assistant_embedding.shape)
                print(cosine_similarity.shape)

            # Compute total loss
            if self.jepa_l2:
                jepa_loss = torch.linalg.norm(
                    user_embedding - assistant_embedding, ord=2, dim=-1
                ).mean()
            elif self.jepa_mse:
                jepa_loss = torch.mean((user_embedding - assistant_embedding) ** 2)
            elif self.infonce:
                ue_norm = F.normalize(user_embedding, p=2, dim=1)
                ae_norm = F.normalize(assistant_embedding, p=2, dim=1)
                cosine_sim = torch.mm(ue_norm, ae_norm.T)
                infonce_logit = cosine_sim / 0.07  # temperature
                infonce_label = torch.arange(
                    cosine_sim.size(0), device=cosine_sim.device
                )
                jepa_loss = F.cross_entropy(infonce_logit, infonce_label)
                if self.debug == 8:
                    print(
                        cosine_sim.shape,
                        infonce_logit.shape,
                        infonce_label.shape,
                        jepa_loss.shape,
                    )
                    exit(0)
            else:
                jepa_loss = 1.0 - torch.mean(cosine_similarity)
        else:
            jepa_loss = 0.0

        total_loss = self.gamma * lm_loss + self.lbd * jepa_loss

        if self.debug == 2 and torch.cuda.current_device() == 0:
            print(lm_loss, self.lbd, torch.mean(cosine_similarity))

        if self.debug == 1 or self.debug == 2:
            exit(0)

        if self.debug == 5 and torch.cuda.current_device() == 0:
            print(f"llm_loss: {lm_loss.float()}, jepa_loss: {jepa_loss.float()}")

        return (total_loss, main_outputs) if return_outputs else total_loss


if __name__ == "__main__":
    import yaml
    from datasets import load_from_disk
    from pathlib import Path
    import time
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--params", type=str)
    args = parser.parse_args()

    with open(args.params, "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)

    model, tokenizer = setup_model_and_tokenizer(
        model_id=params["model_id"],
        use_lora=params["use_lora"],
        lora_rank=params["lora_rank"],
        seed=params["seed"],
        debug=params["debug"],
    )

    train_dataset = load_from_disk("datasets/splits-tokenized/train")
    val_dataset = load_from_disk("datasets/splits-tokenized/val")

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # We're doing causal LM, not masked LM
        pad_to_multiple_of=None,  # Datasets already padded to max length
    )

    output_dir = Path(f'experiments/{params["exp_name"]}/outputs')

    training_args = TrainingArguments(
        output_dir=output_dir / "checkpoints",
        overwrite_output_dir=True,
        # Training parameters
        per_device_train_batch_size=params["batch_size"],
        per_device_eval_batch_size=params["batch_size"],
        gradient_accumulation_steps=params["gradient_accumulation_steps"],
        learning_rate=params["learning_rate"],
        num_train_epochs=params["num_train_epochs"],
        # Evaluation
        eval_strategy=params["eval_strategy"],  # "steps" if eval_dataset else "no",
        eval_steps=params["eval_steps"],
        # Saving
        save_strategy=params["save_strategy"],
        save_steps=params["save_steps"],
        save_total_limit=params["save_total_limit"],
        # Logging
        logging_dir=output_dir / "logs",
        logging_steps=params["eval_steps"],
        # Optimization - key changes for stability
        fp16=False,
        bf16=True,
        gradient_checkpointing=True,  # Enable for memory efficiency
        dataloader_drop_last=True,  # Drop last incomplete batch
        # Memory optimization
        dataloader_num_workers=0,  # Avoid multiprocessing issues
        # Other
        report_to="dvclive",
        remove_unused_columns=False,
        load_best_model_at_end=True,
        # Disable problematic optimizations
        tf32=False,  # May help with stability
        # Set seed for reproducibility
        seed=params["seed"],
        data_seed=params["seed"],
    )

    flop_callback = ProfilerFLOPCallback()

    if params["regular"]:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            callbacks=[flop_callback] if params["track_flop"] else [],
        )
    else:
        trainer = RepresentationTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            callbacks=[flop_callback] if params["track_flop"] else [],
            lbd=params["lbd"],  # Lambda for similarity loss
            gamma=params["gamma"],  # Gamma for LLM loss
            last_token=params["last_token"],  # Index of last token, -1 is '<|eot|>'
            debug=params["debug"],
            additive_mask=params[
                "additive_mask"
            ],  # When set, Use an additive mask to compute both user and assistant in 1 forward pass.
            jepa_l2=params["jepa_l2"],  # When set, Use l2 norm as JEPA loss.
            jepa_mse=params[
                "jepa_mse"
            ],  # When set, Use Mean Squared Error as JEPA loss.
            infonce=params["infonce"],  # When set, Use InfoNCE loss.
            jepa_ratio=params[
                "jepa_ratio"
            ],  # When >0, randomly select this ratio of batches to apply JEPA. This implements Random JEPA-Loss Dropout (LD). If LD = alpha, jepa_ratio = 1 - alpha
        )

    trainer.add_callback(
        DVCLiveCallback(Live(dir=Path(f'experiments/{params["exp_name"]}/dvclive')))
    )
    trainer.train()

    def save_model(model):
        if params["use_lora"]:
            model = model.merge_and_unload()
            model.save_pretrained(output_dir / "best_model")
            tokenizer.save_pretrained(output_dir / "best_model")
        else:
            trainer.save_model(output_dir / "best_model")
            trainer.save_state()
            tokenizer.save_pretrained(output_dir / "best_model")

    retry = 3
    while retry > 0:
        try:
            save_model(model)
            break
        except Exception as e:
            print(f"Success Rate: Saving model encounter error: {e}")
            retry -= 1
            if retry <= 0:
                raise
            time.sleep(10)
