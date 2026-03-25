import json
import logging
import os
import random
import sys

import librosa
from unsloth import FastLanguageModel
from unsloth.kernels import fast_cross_entropy_loss
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm
from trl import SFTTrainer
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset, DatasetDict, VerificationMode


from transformers import (
    HfArgumentParser,
    Trainer,
    set_seed,
    TrainerCallback,
    AutoConfig,
    AutoModelForCausalLM,
    AutoProcessor,
)

from transformers.models.auto.tokenization_auto import TOKENIZER_MAPPING
from transformers.models.auto.processing_auto import PROCESSOR_MAPPING
from transformers.models.qwen2.tokenization_qwen2_fast import Qwen2TokenizerFast
from transformers.models.qwen2.tokenization_qwen2 import Qwen2Tokenizer
from transformers import TrainingArguments as HfTrainingArguments
from peft import LoraConfig, get_peft_model, TaskType

from vibevoice.modular.modeling_vibevoice import VibeVoiceForConditionalGeneration
from vibevoice.modular.configuration_vibevoice import (
    VibeVoiceConfig,
    VibeVoiceAcousticTokenizerConfig,
    VibeVoiceSemanticTokenizerConfig,
    VibeVoiceDiffusionHeadConfig,
)

from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor



# Ensure push_to_hub are registered
def _disable_processor_push_to_hub() -> None:
    def _push_to_hub(cls, *args, **kwargs):
        """Dummy method to disable push_to_hub."""
        return ""

    VibeVoiceProcessor.push_to_hub = classmethod(_push_to_hub)  # type: ignore[misc]

_disable_processor_push_to_hub()
# Register the custom VibeVoice configurations and model with transformers.
AutoConfig.register("vibevoice", VibeVoiceConfig)
AutoConfig.register("vibevoice_acoustic_tokenizer", VibeVoiceAcousticTokenizerConfig)
AutoConfig.register("vibevoice_semantic_tokenizer", VibeVoiceSemanticTokenizerConfig)
AutoConfig.register("vibevoice_diffusion_head", VibeVoiceDiffusionHeadConfig)
AutoModelForCausalLM.register(VibeVoiceConfig, VibeVoiceForConditionalGeneration)

# Register tokenizer mapping (VibeVoice uses Qwen2 tokenizer)
TOKENIZER_MAPPING.register(VibeVoiceConfig, (Qwen2Tokenizer, Qwen2TokenizerFast))

# Register processor mapping (VibeVoice uses VibeVoiceProcessor)
PROCESSOR_MAPPING.register(VibeVoiceConfig, VibeVoiceProcessor)

from vibevoice.finetune.data_vibevoice import VibeVoiceDataset, VibeVoiceCollator

logger = logging.getLogger(__name__)

# ================== SAMPLE CALLBACK UTILS ==================

import copy
import torch
from transformers import TrainerCallback

class EmaCallback(TrainerCallback):
    def __init__(self, attr_path="model.prediction_head", decay=0.999, device="cpu"):
        """
        attr_path: where the head lives under self.model (Trainer wraps your VibeVoiceForConditionalGeneration)
        decay:     EMA decay (0.999 ~ stable, 0.9999 ~ very smooth, slower to adapt)
        """
        self.attr_path = attr_path
        self.decay = float(decay)
        self.device = torch.device(device)
        self.shadow = None
        self._orig = None  # store non-EMA weights when we swap

    def _get_module(self, model):
        # Resolve dotted path like "model.prediction_head"
        mod = model
        for name in self.attr_path.split('.'):
            mod = getattr(mod, name)
        return mod

    def save_shadow(self, path: str) -> None:
        """Persist EMA shadow state to *path* so it can be restored on resume."""
        if self.shadow is not None:
            torch.save(self.shadow, path)

    def load_shadow(self, path: str) -> bool:
        """Restore EMA shadow state from *path*. Returns True on success."""
        if os.path.isfile(path):
            self.shadow = torch.load(path, map_location=self.device, weights_only=True)
            return True
        return False

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        if self.shadow is not None:
            return
        head = self._get_module(model)
        self.shadow = {k: p.detach().to(self.device).clone()
                       for k, p in head.state_dict().items()}

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if self.shadow is None: return
        head = self._get_module(model)
        with torch.no_grad():
            for k, v in head.state_dict().items():
                self.shadow[k].mul_(self.decay).add_(v.detach().to(self.device), alpha=(1.0 - self.decay))

    # ---- Swap helpers ----
    def _swap_in_ema(self, model):
        head = self._get_module(model)
        self._orig = copy.deepcopy(head.state_dict())
        head.load_state_dict(self.shadow, strict=False)

    def _swap_back(self, model):
        if self._orig is None: return
        head = self._get_module(model)
        head.load_state_dict(self._orig, strict=False)
        self._orig = None

    def on_train_end(self, args, state, control, model=None, **kwargs):
        # final checkpoint: persist EMA
        self._swap_in_ema(model)

class InferenceEvalCallback(TrainerCallback):
    """Run TTS inference on a provided conversation after each evaluation,
    save the resulting .wav to the output directory, and log audio to TensorBoard."""

    SAMPLE_RATE = 24000

    def __init__(self, processor, voice_prompt_paths: List[str],
                 cfg_scale: float = 1.3, eval_file: Path | str | None = None, eval_text: str | None = None,
                 eval_on_start: bool = True, ema_callback: Optional["EmaCallback"] = None):
        self.processor = processor
        self.eval_text = eval_text
        self.voice_prompt_paths = voice_prompt_paths
        self.eval_file = eval_file
        self.cfg_scale = cfg_scale
        self._inference_model = None
        self._tb_writer = None
        self.eval_on_start = eval_on_start
        self.ema_callback = ema_callback

    def _get_inference_model(self, training_model):
        """Build a zero-copy inference wrapper that shares the training model's modules."""
        if self._inference_model is not None:
            return self._inference_model

        from vibevoice.modular.modeling_vibevoice_inference import (
            VibeVoiceForConditionalGenerationInference,
        )
        from transformers import PreTrainedModel, GenerationConfig

        unwrapped = training_model
        while hasattr(unwrapped, "module"):
            unwrapped = unwrapped.module

        inf = object.__new__(VibeVoiceForConditionalGenerationInference)
        PreTrainedModel.__init__(inf, unwrapped.config)
        inf.model = unwrapped.model
        inf.lm_head = unwrapped.lm_head
        inf.ddpm_inference_steps = getattr(
            unwrapped.config.diffusion_head_config, "ddpm_num_inference_steps", 10
        )
        inf.generation_config = GenerationConfig()
        inf.eval()
        self._inference_model = inf
        return inf

    def _get_tb_writer(self, args):
        if self._tb_writer is not None:
            return self._tb_writer
        try:
            from torch.utils.tensorboard import SummaryWriter
            log_dir = args.logging_dir
            if log_dir:
                self._tb_writer = SummaryWriter(log_dir=log_dir)
        except ImportError:
            logger.warning("tensorboard not installed – skipping audio logging")
        return self._tb_writer

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        if model is None or not self.eval_text:
            return
        if args.local_rank not in [-1, 0]:
            return
        try:
            logger.info("Running baseline inference before training (step 0)...")
            if self.eval_on_start:
                self._run_inference(args, state, model)
        except Exception as e:
            logger.warning(f"Baseline inference failed: {e}")
            import traceback; traceback.print_exc()

    def on_evaluate(self, args, state, control, model=None, **kwargs):
        if model is None or not self.eval_text:
            return
        if args.local_rank not in [-1, 0]:
            return
        try:
            self._run_inference(args, state, model)
        except Exception as e:
            logger.warning(f"Eval inference failed: {e}")
            import traceback; traceback.print_exc()

    @torch.no_grad()
    def _run_inference(self, args, state, model):
        inf_model = self._get_inference_model(model)

        prev_use_cache = getattr(inf_model.config, "use_cache", None)
        inf_model.config.use_cache = True

        if self.eval_file:
            samples = []
            if self.eval_text:
                samples.append({"text": self.eval_text, "voice_prompts": self.voice_prompt_paths})

            with open(self.eval_file, "r") as f:
                eval_jsonl = [json.loads(line) for line in f]
            samples.extend(random.sample(eval_jsonl, k=3))
        elif self.eval_text:
            samples = [{"text": self.eval_text, "voice_prompts": self.voice_prompt_paths}]
        else:
            return

        # Determine which variants to generate
        ema_cb = self.ema_callback
        has_ema = ema_cb is not None and ema_cb.shadow is not None
        variants = [("train", False)]  # (label, swap_ema)
        if has_ema:
            variants.append(("ema", True))

        for i, sample in tqdm(enumerate(samples), total=len(samples)):
            eval_text = sample["text"]
            voice_prompt_paths = sample["voice_prompts"]
            inputs = self.processor(
                text=[eval_text],
                voice_samples=voice_prompt_paths,
                padding=True,
                return_tensors="pt",
                return_attention_mask=True,
            )
            device = next(inf_model.parameters()).device
            for k, v in inputs.items():
                if torch.is_tensor(v):
                    inputs[k] = v.to(device)

            for variant_label, swap_ema in variants:
                if swap_ema:
                    ema_cb._swap_in_ema(model)
                try:
                    self._generate_and_log(inf_model, inputs, eval_text, args, state, i, variant_label)
                finally:
                    if swap_ema and ema_cb._orig is not None:
                        ema_cb._swap_back(model)

        if prev_use_cache is not None:
            inf_model.config.use_cache = prev_use_cache

    @torch.no_grad()
    def _generate_and_log(self, inf_model, inputs, eval_text, args, state, sample_idx, variant_label):
        """Run generation for a single variant (train or ema) and save / log the result."""
        # Temporarily restore unpatched encode for inference (.sample() support)
        at = inf_model.model.acoustic_tokenizer
        patched_encode = getattr(at, 'encode', None)
        base_encode = getattr(at, '_base_encode', None)
        if base_encode is not None:
            at.encode = base_encode
        try:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = inf_model.generate(
                    **inputs,
                    max_new_tokens=None,
                    cfg_scale=self.cfg_scale,
                    tokenizer=self.processor.tokenizer,
                    generation_config={"do_sample": False},
                    verbose=False,
                )
        finally:
            if base_encode is not None and patched_encode is not None:
                at.encode = patched_encode

        if not outputs.speech_outputs or outputs.speech_outputs[0] is None:
            logger.warning(f"Eval inference ({variant_label}) produced no audio")
            return

        audio_tensor = outputs.speech_outputs[0]

        step = state.global_step
        audio_dir = os.path.join(args.output_dir, "eval_audio")
        os.makedirs(audio_dir, exist_ok=True)
        wav_path = os.path.join(audio_dir, f"step_{step:06d}_{sample_idx}_{variant_label}.wav")
        self.processor.save_audio(audio_tensor, output_path=wav_path)
        logger.info(f"Eval inference audio ({variant_label}) saved to {wav_path}")

        with Path(wav_path).with_suffix(".txt").open("w") as f:
            f.write(eval_text)

        tb = self._get_tb_writer(args)
        if tb is not None:
            audio_np = audio_tensor.cpu().float().numpy()
            if audio_np.ndim > 1:
                audio_np = audio_np.squeeze()
            tb.add_audio(f"eval/inference_audio_{sample_idx}_{variant_label}", audio_np, global_step=step,
                         sample_rate=self.SAMPLE_RATE)
            tb.flush()
            logger.info(f"Logged eval inference audio ({variant_label}) to TensorBoard (step {step})")


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        default=None, metadata={"help": "Path to VibeVoice base model with config.json"}
    )
    processor_name_or_path: Optional[str] = field(
        default=None, metadata={"help": "Path to processor dir (preprocessor_config.json). Defaults to model path."}
    )
    cache_dir: Optional[str] = field(default=None)
    freeze_acoustic_tokenizer: bool = field(default=True)
    freeze_semantic_tokenizer: bool = field(default=True)
    lora_r: int = field(default=8)
    lora_alpha: int = field(default=32)
    lora_dropout: float = field(default=0.05)
    lora_target_modules: str = field(
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        metadata={"help": "Comma-separated list of target module names in the LLM blocks"},
    )
    lora_wrap_diffusion_head: bool = field(default=False, metadata={"help": "Wrap diffusion head with PEFT LoRA"})
    train_diffusion_head: bool = field(default=False, metadata={"help": "Train diffusion prediction head (full fine-tune)"})
    train_connectors: bool = field(default=False, metadata={"help": "Train acoustic/semantic connectors (full fine-tune)"})
    freeze_language_model: bool = field(default=True, metadata={"help": "Freeze all LLM layers. Set False to fully fine-tune."})
    use_llm_lora: bool = field(default=False, metadata={"help": "Apply LoRA adapters to the LLM. Requires freeze_language_model=True."})
    train_embeddings: bool = field(default=False, metadata={"help": "Train input/output embeddings (embed_tokens + lm_head)."})
    llm_learning_rate: Optional[float] = field(default=None, metadata={"help": "Separate learning rate for the LLM (LoRA or full). If None, uses the global learning_rate."})
    connectors_learning_rate: Optional[float] = field(default=None, metadata={"help": "Separate learning rate for the connectors (full fine-tune). If None, uses the global learning_rate."})
    diffusion_learning_rate: Optional[float] = field(default=None, metadata={"help": "Separate learning rate for the diffusion prediction head. If None, uses the global learning_rate."})
    layers_to_freeze: Optional[str] = field(
        default=None, 
        metadata={"help": "Comma-separated indices of diffusion head layers to freeze (e.g., '0,1,5,7,8')."}
    )

@dataclass
class DataArguments:
    dataset_name: Optional[str] = field(default=None, metadata={"help": "HF dataset name or 'json' with --train_jsonl for local files"})
    dataset_config_name: Optional[str] = field(default=None)
    train_split_name: str = field(default="train")
    eval_split_name: Optional[str] = field(default="validation")
    text_column_name: str = field(default="text")
    audio_column_name: str = field(default="audio")
    voice_prompts_column_name: Optional[str] = field(default="voice_prompts")
    eval_split_size: float = field(default=0.0)
    ignore_verifications: bool = field(default=False)
    max_length: Optional[int] = field(default=None)
    train_jsonl: Optional[str] = field(default=None, metadata={"help": "Path to local train JSONL with {text, audio, [voice_prompts]}"})
    validation_jsonl: Optional[str] = field(default=None, metadata={"help": "Optional path to local validation JSONL"})
    voice_prompt_drop_rate: float = field(
        default=0.0,
        metadata={"help": "Probability to drop conditioning voice prompt during training (0.0 keep always, 1.0 drop always)."},
    )
    eval_inference_text: Optional[str] = field(
        default=None,
        metadata={"help": "Text script (or path to .txt file) for TTS inference after each eval. E.g. 'Speaker 1: Hello, this is a test.'"},
    )
    eval_inference_voice_prompt: Optional[str] = field(
        default=None,
        metadata={"help": "Comma-separated path(s) to voice prompt WAV file(s) for eval inference."},
    )
    eval_inference_cfg_scale: float = field(
        default=1.3,
        metadata={"help": "CFG scale for eval inference generation."},
    )
    precomputed_latents_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Path to directory with precomputed latents (.npy files + latents_meta.json). "
                          "When set, the training pipeline loads precomputed acoustic/semantic latents "
                          "instead of running the frozen tokenizer encoders on every step."},
    )


@dataclass
class CustomTrainingArguments(HfTrainingArguments):
    ddpm_batch_mul: int = field(default=1)
    ce_loss_weight: float = field(default=1.0)
    diffusion_loss_weight: float = field(default=1.0)
    debug_ce_details: bool = field(default=False)
    debug_ce_topk: int = field(default=5)
    debug_ce_max_examples: int = field(default=1)
    debug_ce_every_n_steps: int = field(default=200)
    gradient_clipping: bool = field(
        default=False,
        metadata={"help": "Enable gradient clipping using max_grad_norm (set via --max_grad_norm, default 1.0). When False, disables clipping by forcing max_grad_norm=0.0."},
    )
    debug_save: bool = field(
        default=False,
        metadata={"help": "If set, saves model components BEFORE training starts, into output_dir/debug_initial."},
    )
    init_from_checkpoint: Optional[str] = field(
        default=None,
        metadata={
            "help": "Initialize LoRA/extra module weights from a checkpoint directory without restoring optimizer/scheduler/trainer state."
        },
    )
    save_merged: bool = field(
        default=False,
        metadata={"help": "When using LLM LoRA, also save a merged model (LoRA folded into base weights) at each checkpoint under merged/."},
    )


def build_lora_config(args: ModelArguments) -> LoraConfig:
    target_modules = [s.strip() for s in args.lora_target_modules.split(",") if s.strip()]
    return LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=target_modules,
    )

def build_head_lora_config(args: ModelArguments) -> LoraConfig:
    target_modules = ["noisy_images_proj","cond_proj","gate_proj","up_proj","down_proj","linear"]
    return LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.FEATURE_EXTRACTION,
        target_modules=target_modules,
    )

def mask_for_ce(labels: torch.Tensor, attention_mask: torch.Tensor, acoustic_input_mask: torch.Tensor, pad_id: int = -100) -> torch.Tensor:
    shifted = labels[:, 1:].contiguous()
    base_mask = attention_mask[:, 1:].contiguous().eq(1) if (attention_mask is not None and attention_mask.numel() > 0) else torch.ones_like(shifted, dtype=torch.bool)
    # Only mask interior acoustic positions (both source AND target are acoustic).
    # This preserves CE loss at boundary transitions:
    #   speech_start -> first speech_diffusion  (entry into speech)
    #   last speech_diffusion -> speech_end     (exit from speech)
    source_is_acoustic = acoustic_input_mask[:, :-1].contiguous()
    target_is_acoustic = acoustic_input_mask[:, 1:].contiguous()
    is_interior_acoustic = source_is_acoustic & target_is_acoustic
    final_mask = base_mask & (~is_interior_acoustic)
    out = shifted.clone()
    out[~final_mask] = pad_id
    return out

def _patch_acoustic_encode_for_legacy_indexing(model_obj, logger_):
    try:
        acoustic = getattr(getattr(model_obj, "model", model_obj), "acoustic_tokenizer", None)
        if acoustic is None or not hasattr(acoustic, "encode"):
            logger_.warning("No acoustic_tokenizer.encode() found to patch.")
            return
        base_encode = acoustic.encode
        def encode_wrapped(*args, **kwargs):
            out = base_encode(*args, **kwargs)
            try:
                _ = out[0][0]
                return out
            except Exception:
                pass
            if isinstance(out, dict):
                for k in ("frames", "codes", "tokens", "latents", "hidden_states"):
                    if k in out:
                        return [[out[k]]]
                if len(out) > 0:
                    return [[next(iter(out.values()))]]
            for attr in ("frames", "codes", "tokens", "latents", "hidden_states"):
                if hasattr(out, attr):
                    return [[getattr(out, attr)]]
            try:
                if isinstance(out, torch.Tensor):
                    return [[out]]
            except Exception:
                pass
            return [[out]]
        acoustic.encode = encode_wrapped
        logger_.info("Patched acoustic_tokenizer.encode() to return [[...]] for legacy indexing.")
    except Exception as e:
        logger_.warning(f"Failed to patch acoustic_tokenizer.encode(): {e}")

def main() -> None:
    parser = HfArgumentParser((ModelArguments, DataArguments, CustomTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".yaml"):
        model_args, data_args, training_args = parser.parse_yaml_file(
            yaml_file=sys.argv[1]
        )
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    log_level = logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN
    logger.setLevel(log_level)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s", datefmt="%m/%d/%Y %H:%M:%S"))
        logger.addHandler(handler)
    logger.info("Training/evaluation parameters %s", training_args)
    set_seed(training_args.seed)

    # Configure gradient clipping
    if not getattr(training_args, "gradient_clipping", False):
        if hasattr(training_args, "max_grad_norm"):
            training_args.max_grad_norm = 0.0
            logger.info("Gradient clipping disabled (set max_grad_norm=0.0). Use --gradient_clipping to enable.")
    else:
        if (not hasattr(training_args, "max_grad_norm")) or training_args.max_grad_norm is None or training_args.max_grad_norm <= 0:
            training_args.max_grad_norm = 1.0
        logger.info(f"Gradient clipping enabled: max_grad_norm={training_args.max_grad_norm}")

    # Load processor
    processor_path = model_args.processor_name_or_path or model_args.model_name_or_path
    if processor_path is None:
        raise ValueError("--model_name_or_path (or --processor_name_or_path) must be provided")
    processor: VibeVoiceProcessor = VibeVoiceProcessor.from_pretrained(processor_path)

    # Required special tokens
    tok = processor.tokenizer
    for required in ["speech_start_id", "speech_diffusion_id", "speech_end_id"]:
        if not hasattr(tok, required) or getattr(tok, required) is None:
            raise RuntimeError(f"Tokenizer missing required special id: {required}")

    # Load model
    if model_args.model_name_or_path is None:
        raise ValueError("--model_name_or_path is required to load VibeVoice base model")
    
    logger.info(f"Loading model from: {model_args.model_name_or_path}")
    
    # Ensure registrations are active (sometimes they need to be re-registered)
    logger.info("Re-registering VibeVoice configurations with transformers...")
    AutoConfig.register("vibevoice", VibeVoiceConfig)
    AutoConfig.register("vibevoice_acoustic_tokenizer", VibeVoiceAcousticTokenizerConfig)
    AutoConfig.register("vibevoice_semantic_tokenizer", VibeVoiceSemanticTokenizerConfig)
    AutoConfig.register("vibevoice_diffusion_head", VibeVoiceDiffusionHeadConfig)
    AutoModelForCausalLM.register(VibeVoiceConfig, VibeVoiceForConditionalGeneration)
    TOKENIZER_MAPPING.register(VibeVoiceConfig, (Qwen2Tokenizer, Qwen2TokenizerFast))
    PROCESSOR_MAPPING.register(VibeVoiceConfig, VibeVoiceProcessor)
    
    dtype = torch.float32
    if training_args.bf16:
        dtype = torch.bfloat16
    elif getattr(training_args, "fp16", False):
        dtype = torch.float16


    model,tokenizer = FastLanguageModel.from_pretrained(
            model_args.model_name_or_path,
            auto_model=VibeVoiceForConditionalGeneration,
            dtype=dtype,
            whisper_language="none",
            whisper_task="none",
            use_gradient_checkpointing = "unsloth" if training_args.gradient_checkpointing else False,
            device_map={'': torch.cuda.current_device()},
            load_in_4bit = False
    )

    _patch_acoustic_encode_for_legacy_indexing(model, logger)
    processor.semantic_tokenizer = getattr(model.model, "semantic_tokenizer", None)

    # Diagnostics: LM head tie
    try:
        in_emb_mod = model.get_input_embeddings()
        out_emb_mod = model.get_output_embeddings()
        in_w = getattr(in_emb_mod, "weight", None)
        out_w = getattr(out_emb_mod, "weight", None)
        shared_ptr = bool(in_w is not None and out_w is not None and in_w.data_ptr() == out_w.data_ptr())
        values_equal = False
        if in_w is not None and out_w is not None and in_w.shape == out_w.shape:
            try:
                values_equal = bool(torch.allclose(in_w, out_w))
            except Exception:
                values_equal = False
        try:
            tie_cfg = getattr(getattr(model.config, "decoder_config", model.config), "tie_word_embeddings", None)
        except Exception:
            tie_cfg = getattr(model.config, "tie_word_embeddings", None)
        logger.info(f"LM head diagnostics -> shared_params={shared_ptr}, values_equal={values_equal}, tie_word_embeddings={tie_cfg}")
        if out_w is not None:
            logger.info(f"LM head requires_grad before freeze: {bool(out_w.requires_grad)}")
    except Exception as e:
        logger.warning(f"LM head tie diagnostics failed: {e}")

    # Hard-tie LM head
    try:
        emb_module = model.get_input_embeddings()
        head_module = model.get_output_embeddings()
        if hasattr(emb_module, "weight") and hasattr(head_module, "weight"):
            if emb_module.weight.shape == head_module.weight.shape and emb_module.weight.data_ptr() != head_module.weight.data_ptr():
                with torch.no_grad():
                    head_module.weight = emb_module.weight
                logger.info("Force-tied LM head weight to input embeddings (pointer share).")
    except Exception as e:
        logger.warning(f"Force-tie of LM head failed: {e}")

    # Validate special IDs (info logs only)
    try:
        special_names = ["speech_start_id", "speech_diffusion_id", "speech_end_id"]
        try:
            vocab_size = int(getattr(model.config.decoder_config, "vocab_size", 0))
        except Exception:
            vocab_size = 0
        in_emb_mod = model.get_input_embeddings()
        out_emb_mod = model.get_output_embeddings()
        in_w = getattr(in_emb_mod, "weight", None)
        out_w = getattr(out_emb_mod, "weight", None)
        for name in special_names:
            val = getattr(tok, name, None)
            exists = (val is not None)
            in_range = (exists and isinstance(val, int) and 0 <= val < vocab_size)
            equal_row = None
            if in_range and in_w is not None and out_w is not None and in_w.shape == out_w.shape and in_w.size(0) > val:
                try:
                    equal_row = bool(torch.allclose(in_w[val], out_w[val]))
                except Exception:
                    equal_row = False
            decoded_str = None
            if exists and isinstance(val, int):
                try:
                    decoded_str = tok.decode([val])
                except Exception:
                    try:
                        decoded_str = tok.convert_ids_to_tokens(val)
                    except Exception:
                        decoded_str = "<decode_failed>"
            logger.info(f"Special token check -> {name}={val}, decoded='{decoded_str}', exists={exists}, in_vocab_range={in_range}, emb_vs_head_row_equal={equal_row}")
    except Exception as e:
        logger.warning(f"Special token ID/row validation failed: {e}")

    # Quick tokenizer diagnostics (optional)
    try:
        logger.info("=== TOKENIZER DIAGNOSTICS ===")
        logger.info(f"Tokenizer class: {type(tok).__name__}")
        logger.info(f"Tokenizer vocab_size: {tok.vocab_size}")
        # tiny CE smoke test
        with torch.no_grad():
            simple_text = "The cat sat on the mat."
            simple_ids = torch.tensor([tok.encode(simple_text, add_special_tokens=True)], device=model.device)
            simple_mask = torch.ones_like(simple_ids, dtype=torch.bool)
            x = model.get_input_embeddings()(simple_ids)
            outputs = model.model(inputs_embeds=x, attention_mask=simple_mask, return_dict=True)
            logits = model.lm_head(outputs.last_hidden_state)
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = simple_ids[:, 1:].contiguous()
            ce_loss = fast_cross_entropy_loss(shift_logits, shift_labels)
            logger.info(f"Simple text CE loss: {ce_loss.item():.4f}")
    except Exception as e:
        logger.warning(f"Tokenizer diagnostics failed: {e}")

    # Disable cache during training
    if hasattr(model.config, "use_cache") and training_args.do_train:
        model.config.use_cache = False

    # Freeze tokenizers
    if model_args.freeze_acoustic_tokenizer and hasattr(model.model, "acoustic_tokenizer"):
        for p in model.model.acoustic_tokenizer.parameters():
            p.requires_grad = False
    if model_args.freeze_semantic_tokenizer and hasattr(model.model, "semantic_tokenizer"):
        for p in model.model.semantic_tokenizer.parameters():
            p.requires_grad = False

    # Validate flag combinations
    if not model_args.freeze_language_model and model_args.use_llm_lora:
        raise ValueError(
            "Cannot use use_llm_lora=True with freeze_language_model=False. "
            "LoRA is only meaningful when the base LLM weights are frozen.")

    # LoRA wrap LLM (optional)
    if model_args.use_llm_lora:
        lora_cfg = build_lora_config(model_args)
        model.model.language_model = get_peft_model(model.model.language_model, lora_cfg)
        logger.info("LLM wrapped with LoRA.")
    else:
        logger.info("No LoRA on LLM (use_llm_lora=False).")

    try:
        model.tie_weights()
    except Exception:
        pass

    # Freeze all then enable trainable subsets
    for _, p in model.named_parameters():
        p.requires_grad = False

    if model_args.use_llm_lora:
        for n, p in model.model.language_model.named_parameters():
            if "lora_A" in n or "lora_B" in n:
                p.requires_grad = True
    elif not model_args.freeze_language_model:
        for p in model.model.language_model.parameters():
            p.requires_grad = True
        logger.info("LLM unfrozen (full fine-tune).")

    # Diffusion head LoRA wrapping (optional)
    if getattr(model_args, "lora_wrap_diffusion_head", False) and hasattr(model.model, "prediction_head"):
        class _HeadForwardShim(nn.Module):
            def __init__(self, base: nn.Module):
                super().__init__(); self.base = base

            def forward(self, *args, **kwargs):
                if len(args) >= 3:
                    noisy_images, timesteps, condition = args[:3]
                else:
                    noisy_images = kwargs.get("noisy_images")
                    timesteps = kwargs.get("timesteps")
                    condition = kwargs.get("condition")
                return self.base(noisy_images, timesteps, condition)
        try:
            shim = _HeadForwardShim(model.model.prediction_head)
            model.model.prediction_head = get_peft_model(shim, build_head_lora_config(model_args))
            for n, p in model.model.prediction_head.named_parameters():
                if "lora_A" in n or "lora_B" in n:
                    p.requires_grad = True
        except Exception as e:
            logger.warning(f"Could not LoRA-wrap diffusion head: {e}")

    # Train full diffusion head (optional)
    if getattr(model_args, "train_diffusion_head", False) and hasattr(model.model, "prediction_head"):
        for p in model.model.prediction_head.parameters():
            p.requires_grad = True

    # Freeze diffusion head layers (optional)
    if model_args.layers_to_freeze is not None and hasattr(model.model, "prediction_head"):
        head_params = list(model.model.prediction_head.named_parameters())
        try:
            indices_to_freeze = {int(x.strip()) for x in model_args.layers_to_freeze.split(',') if x.strip()}
            frozen_count = 0
            for i, (name, param) in enumerate(head_params):
                if i in indices_to_freeze:
                    param.requires_grad = False
                    frozen_count += 1
                    logger.info(f"Froze layer [{i}]: {name}")
            logger.info(f"Successfully froze {frozen_count} parameter groups in the diffusion head.")
        except Exception as e:
            logger.error(f"Could not parse --layers_to_freeze: {e}")
            raise
    
    # Connectors
    if getattr(model_args, "train_connectors", False):
        if hasattr(model.model, "acoustic_connector"):
            for p in model.model.acoustic_connector.parameters():
                p.requires_grad = True
        if hasattr(model.model, "semantic_connector"):
            for p in model.model.semantic_connector.parameters():
                p.requires_grad = True
    else:
        if hasattr(model.model, "acoustic_connector"):
            for p in model.model.acoustic_connector.parameters():
                p.requires_grad = False
        if hasattr(model.model, "semantic_connector"):
            for p in model.model.semantic_connector.parameters():
                p.requires_grad = False

    # Freeze embedding + head (unless train_embeddings is set)
    if model_args.train_embeddings:
        try:
            emb = model.get_input_embeddings()
            if hasattr(emb, "weight"):
                emb.weight.requires_grad_(True)
            head = model.get_output_embeddings()
            if head is not None and hasattr(head, "weight"):
                head.weight.requires_grad_(True)
        except Exception:
            pass
        logger.info("Embeddings and LM head will be trained.")
    else:
        try:
            emb = model.get_input_embeddings()
            if hasattr(emb, "weight"):
                emb.weight.requires_grad_(False)
            head = model.get_output_embeddings()
            if head is not None and hasattr(head, "weight"):
                head.weight.requires_grad_(False)
        except Exception:
            pass

    # Diagnostics
    def _sum_params(named_iter):
        return sum(p.numel() for _, p in named_iter if p.requires_grad)
    try:
        lm_lora = _sum_params(model.model.language_model.named_parameters()) if hasattr(model.model, "language_model") else 0
        pred_head_train = _sum_params(model.model.prediction_head.named_parameters()) if hasattr(model.model, "prediction_head") else 0
        ac_conn_train = _sum_params(model.model.acoustic_connector.named_parameters()) if hasattr(model.model, "acoustic_connector") else 0
        se_conn_train = _sum_params(model.model.semantic_connector.named_parameters()) if hasattr(model.model, "semantic_connector") else 0
        total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        emb_train = model.get_input_embeddings().weight.numel() if model.get_input_embeddings().weight.requires_grad else 0
        head_out = model.get_output_embeddings()
        lm_head_train = head_out.weight.numel() if (head_out is not None and head_out.weight.requires_grad) else 0
        lm_label = "LLM-frozen" if model_args.freeze_language_model and not model_args.use_llm_lora else ("LLM-LoRA" if model_args.use_llm_lora else "LLM-full")
        logger.info(f"Trainable by block -> {lm_label}: {lm_lora:,} | diff_head: {pred_head_train:,} | ac_conn: {ac_conn_train:,} | se_conn: {se_conn_train:,} | embeddings: {emb_train:,} | lm_head: {lm_head_train:,}")
        logger.info("TOTAL trainable: %s", f"{total_trainable:,}")
    except Exception:
        pass

    # Datasets
    verification_mode = VerificationMode.NO_CHECKS if data_args.ignore_verifications else VerificationMode.BASIC_CHECKS
    if data_args.train_jsonl is not None:
        data_files: Dict[str, str] = {"train": data_args.train_jsonl}
        if data_args.validation_jsonl is not None:
            data_files["validation"] = data_args.validation_jsonl
        raw = load_dataset("json", data_files=data_files, verification_mode=verification_mode, cache_dir=model_args.cache_dir, keep_in_memory=True)
    else:
        if data_args.dataset_name is None:
            raise ValueError("Provide --dataset_name (HF datasets) or use --train_jsonl/--validation_jsonl for local files.")
        raw = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            verification_mode=verification_mode,
            cache_dir=model_args.cache_dir,
        )
    train_ds = raw[data_args.train_split_name]
    eval_ds = None
    if training_args.do_eval:
        if data_args.eval_split_name and data_args.eval_split_name in raw:
            eval_ds = raw[data_args.eval_split_name]
        elif data_args.eval_split_size and data_args.eval_split_size > 0 and len(train_ds) > 1:
            split = train_ds.train_test_split(test_size=data_args.eval_split_size, seed=training_args.seed)
            train_ds, eval_ds = split["train"], split["test"]

    train_base_dir = str(Path(data_args.train_jsonl).parent) if data_args.train_jsonl else None
    precomputed_dir = data_args.precomputed_latents_dir

    # If a precomputed latents directory is set, load the HF Dataset from disk
    # and use it instead of the JSONL-loaded dataset.
    if precomputed_dir is not None:
        from datasets import load_from_disk
        ds_path = os.path.join(precomputed_dir, "dataset")
        logger.info(f"Loading precomputed HF Dataset from {ds_path}")
        precomputed_full = load_from_disk(ds_path)

        if training_args.do_eval and data_args.eval_split_size and data_args.eval_split_size > 0 and len(precomputed_full) > 1:
            split = precomputed_full.train_test_split(test_size=data_args.eval_split_size, seed=training_args.seed)
            train_ds, eval_ds = split["train"], split["test"]
        else:
            train_ds = precomputed_full
            # Keep eval_ds from earlier JSONL-based loading if it exists

    # Ratios/dims from processor+model
    speech_compress_ratio = getattr(processor, "speech_tok_compress_ratio", 3200)
    semantic_dim = getattr(model.config, "semantic_vae_dim", None)
    if semantic_dim is None:
        try:
            semantic_dim = int(getattr(model.config.semantic_tokenizer_config, "vae_dim", 128))
        except Exception:
            semantic_dim = 128
    acoustic_dim = getattr(model.config, "acoustic_vae_dim", 64)

    train_dataset = VibeVoiceDataset(
        train_ds,
        text_column=data_args.text_column_name,
        audio_column=data_args.audio_column_name,
        voice_prompts_column=data_args.voice_prompts_column_name,
        base_dir=train_base_dir,
        speech_compress_ratio=speech_compress_ratio,
    )

    eval_dataset = None
    if eval_ds is not None:
        eval_base_dir = str(Path(data_args.validation_jsonl).parent) if data_args.validation_jsonl else train_base_dir
        eval_dataset = VibeVoiceDataset(
            eval_ds,
            text_column=data_args.text_column_name,
            audio_column=data_args.audio_column_name,
            voice_prompts_column=data_args.voice_prompts_column_name,
            base_dir=eval_base_dir,
            speech_compress_ratio=speech_compress_ratio,
        )

    # If precomputed latents are available, load scaling factors and set them on the model
    if precomputed_dir is not None:
        meta_path = os.path.join(precomputed_dir, "latents_meta.json")
        if os.path.isfile(meta_path):
            with open(meta_path, "r") as f:
                latents_meta = json.load(f)
            sf = latents_meta["scaling_factor"]
            bf = latents_meta["bias_factor"]
            model.model.speech_scaling_factor.copy_(torch.tensor(sf))
            model.model.speech_bias_factor.copy_(torch.tensor(bf))
            logger.info(f"Loaded precomputed scaling_factor={sf:.6f}, bias_factor={bf:.6f} from {meta_path}")
        else:
            logger.warning(
                f"precomputed_latents_dir={precomputed_dir} set but no latents_meta.json found. "
                "Scaling factors will be computed from the first batch (slower first step)."
            )

    compute_semantics_flag = hasattr(processor, "semantic_tokenizer") and processor.semantic_tokenizer is not None

    data_collator = VibeVoiceCollator(
        processor=processor,
        max_length=data_args.max_length,
        speech_compress_ratio=speech_compress_ratio,
        semantic_vae_dim=semantic_dim,
        acoustic_vae_dim=acoustic_dim,
        compute_semantics=compute_semantics_flag,
        debug_checks=False,
        voice_prompt_drop_rate=data_args.voice_prompt_drop_rate,
    )

    class LoRADebugCallback(TrainerCallback):
        def __init__(self, log_every_n_steps: int = 50):
            self.log_every_n_steps = max(1, int(log_every_n_steps))
            self.prev_param_norms: Dict[str, float] = {}
            self.lora_param_names: List[str] = []

        def on_train_begin(self, args, state, control, model=None, **kwargs):
            try:
                if model is None:
                    return
                named: Dict[str, torch.nn.Parameter] = dict(model.named_parameters())
                self.lora_param_names = [n for n in named.keys() if ("lora_A" in n or "lora_B" in n)]
                for n in self.lora_param_names:
                    p = named[n]
                    self.prev_param_norms[n] = float(p.data.norm().item())
                total = len(self.lora_param_names)
                req_grad = sum(1 for n in self.lora_param_names if named[n].requires_grad)
                num_A = sum(1 for n in self.lora_param_names if "lora_A" in n)
                num_B = sum(1 for n in self.lora_param_names if "lora_B" in n)
                zero_B = sum(1 for n in self.lora_param_names if ("lora_B" in n and float(named[n].data.norm().item()) == 0.0))
                logger.info(f"LoRA debug: found {total} LoRA params (A={num_A}, B={num_B}); trainable={req_grad}. Initial lora_B_zero={zero_B}.")
                if total == 0:
                    logger.warning("LoRA debug: No LoRA parameters found. Check lora_target_modules.")
                if req_grad != total:
                    logger.warning("LoRA debug: Some LoRA params are frozen. They should be trainable.")
            except Exception as e:
                logger.warning(f"LoRA debug (on_train_begin) failed: {e}")

        def on_step_end(self, args, state, control, model=None, **kwargs):
            try:
                if model is None or len(self.lora_param_names) == 0:
                    return
                step = int(getattr(state, "global_step", 0) or 0)
                if step % self.log_every_n_steps != 0 and step != 1:
                    return
                named: Dict[str, torch.nn.Parameter] = dict(model.named_parameters())
                changed_A = 0
                changed_B = 0
                zero_B = 0
                eps = 1e-12
                for n in self.lora_param_names:
                    p = named.get(n, None)
                    if p is None:
                        continue
                    prev = self.prev_param_norms.get(n, 0.0)
                    curr = float(p.data.norm().item())
                    if "lora_A" in n and abs(curr - prev) > eps:
                        changed_A += 1
                    if "lora_B" in n:
                        if abs(curr - prev) > eps:
                            changed_B += 1
                        if curr == 0.0:
                            zero_B += 1
                    self.prev_param_norms[n] = curr
                total_A = sum(1 for n in self.lora_param_names if "lora_A" in n)
                total_B = sum(1 for n in self.lora_param_names if "lora_B" in n)
                logger.info(f"LoRA debug step {step}: changed A {changed_A}/{total_A}, changed B {changed_B}/{total_B}, lora_B_zero_now={zero_B}.")
            except Exception as e:
                logger.warning(f"LoRA debug (on_step_end) failed: {e}")

    class VibeVoiceTrainer(Trainer):

        _ce_loss_acc: float = 0.0
        _diffusion_loss_acc: float = 0.0
        _loss_count: int = 0
        _eval_ce_loss_acc: float = 0.0
        _eval_diffusion_loss_acc: float = 0.0
        _eval_loss_count: int = 0

        def log(self, logs, *args, **kwargs):
            if "loss" in logs and self._loss_count > 0:
                logs["train/ce_loss"] = self._ce_loss_acc / self._loss_count
                logs["train/diffusion_loss"] = self._diffusion_loss_acc / self._loss_count
                self._ce_loss_acc = 0.0
                self._diffusion_loss_acc = 0.0
                self._loss_count = 0
                if hasattr(self, "optimizer") and self.optimizer is not None and len(self.optimizer.param_groups) > 0:
                    lr_val = self.optimizer.param_groups[0].get("lr", None)
                    if lr_val is not None:
                        logs["train/learning_rate_real"] = float(lr_val)
            if "eval_loss" in logs and self._eval_loss_count > 0:
                logs["eval/ce_loss"] = self._eval_ce_loss_acc / self._eval_loss_count
                logs["eval/diffusion_loss"] = self._eval_diffusion_loss_acc / self._eval_loss_count
                self._eval_ce_loss_acc = 0.0
                self._eval_diffusion_loss_acc = 0.0
                self._eval_loss_count = 0
            super().log(logs, *args, **kwargs)

        def _get_ema_callback(self) -> Optional[EmaCallback]:
            for cb in self.callback_handler.callbacks:
                if isinstance(cb, EmaCallback):
                    return cb
            return None

        def evaluate(self, *args, **kwargs):
            # NOTE: We intentionally do NOT swap in EMA weights here.
            # Eval loss tracks the actual training weights, which is more
            # useful for monitoring training progress.  EMA quality is
            # compared via the inference audio samples generated by
            # InferenceEvalCallback (which handles its own EMA swap).
            return super().evaluate(*args, **kwargs)

        def create_optimizer(self):
            if self.optimizer is not None:
                return self.optimizer

            llm_lr = model_args.llm_learning_rate
            connectors_lr = model_args.connectors_learning_rate
            diffusion_lr = model_args.diffusion_learning_rate
            base_lr = self.args.learning_rate

            need_multi_group = (llm_lr is not None and llm_lr != base_lr) or \
                               (connectors_lr is not None and connectors_lr != base_lr) or \
                               (diffusion_lr is not None and diffusion_lr != base_lr)

            if need_multi_group:
                effective_llm_lr = llm_lr if llm_lr is not None else base_lr
                effective_conn_lr = connectors_lr if connectors_lr is not None else base_lr
                effective_diff_lr = diffusion_lr if diffusion_lr is not None else base_lr

                llm_params = []
                connectors_params = []
                diffusion_params = []
                other_params = []
                for name, param in self.model.named_parameters():
                    if not param.requires_grad:
                        continue
                    if "language_model" in name or "lm_head" in name:
                        llm_params.append(param)
                    elif "acoustic_connector" in name or "semantic_connector" in name:
                        connectors_params.append(param)
                    elif "prediction_head" in name:
                        diffusion_params.append(param)
                    else:
                        other_params.append(param)

                param_groups = []
                if other_params:
                    param_groups.append({"params": other_params, "lr": base_lr})
                if llm_params:
                    param_groups.append({"params": llm_params, "lr": effective_llm_lr})
                if connectors_params:
                    param_groups.append({"params": connectors_params, "lr": effective_conn_lr})
                if diffusion_params:
                    param_groups.append({"params": diffusion_params, "lr": effective_diff_lr})

                logger.info(
                    f"Optimizer param groups: "
                    f"LLM ({len(llm_params)} tensors, lr={effective_llm_lr}) | "
                    f"Connectors ({len(connectors_params)} tensors, lr={effective_conn_lr}) | "
                    f"Diffusion ({len(diffusion_params)} tensors, lr={effective_diff_lr}) | "
                    f"Other ({len(other_params)} tensors, lr={base_lr})")

                opt_cls, opt_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
                opt_kwargs.pop("lr", None)
                self.optimizer = opt_cls(param_groups, **opt_kwargs)
            else:
                # Default single-group optimizer
                super().create_optimizer()

            return self.optimizer

        def training_forward(self, model: VibeVoiceForConditionalGeneration, inputs: Dict[str, Any]):
            """Custom forward pass for training with new diffusion loss calculation."""
            # Extract inputs
            input_ids = inputs.get("input_ids")
            attention_mask = inputs.get("attention_mask")
            position_ids = inputs.get("position_ids")
            past_key_values = inputs.get("past_key_values")
            inputs_embeds = inputs.get("inputs_embeds")
            use_cache = inputs.get("use_cache", False)
            output_attentions = inputs.get("output_attentions")
            output_hidden_states = inputs.get("output_hidden_states")
            return_dict = inputs.get("return_dict", True)
            cache_position = inputs.get("cache_position")
            
            # Speech-related inputs
            speech_tensors = inputs.get("speech_tensors")
            speech_masks = inputs.get("speech_masks")
            speeches_loss_input = inputs.get("speeches_loss_input")
            speech_semantic_tensors = inputs.get("speech_semantic_tensors")
            acoustic_input_mask = inputs.get("acoustic_input_mask")
            acoustic_loss_mask = inputs.get("acoustic_loss_mask")
            precomputed_latents = inputs.get("precomputed_latents", False)
            ddmp_batch_mul = training_args.ddpm_batch_mul
            kwargs = {}
            
            # --- START: Copy of model forward logic with new diffusion loss ---
            x = model.get_input_embeddings()(input_ids)
            
            x = x.clone()
            if getattr(training_args, "bf16", False):
                model_dtype = torch.bfloat16
            elif getattr(training_args, "fp16", False):
                model_dtype = torch.float16
            else:
                model_dtype = getattr(getattr(emb_module, "weight", None), "dtype", x.dtype)
            if x.dtype != model_dtype:
                x = x.to(dtype=model_dtype)
            semantic_speech_all_connect_features = model.model.semantic_connector(speech_semantic_tensors)
            if precomputed_latents and speech_tensors is not None:
                # --- Precomputed path: speech_tensors is [segments, T, D] (acoustic means) ---
                with torch.no_grad():
                    acoustic_means = speech_tensors.type_as(x)
                    fix_std = model.model.acoustic_tokenizer.fix_std
                    std_dist_type = model.model.acoustic_tokenizer.std_dist_type
                    if std_dist_type == 'gaussian':
                        batch_size = acoustic_means.size(0)
                        value = fix_std / 0.8
                        std = torch.randn(batch_size, dtype=acoustic_means.dtype, device=acoustic_means.device) * value
                        std = std.view(-1, 1, 1)
                        audio_tokens = acoustic_means + std * torch.randn_like(acoustic_means)
                    elif std_dist_type == 'fix':
                        audio_tokens = acoustic_means + fix_std * torch.randn_like(acoustic_means)
                    else:
                        audio_tokens = acoustic_means

                    # Apply scaling/bias (already set from latents_meta.json or first-batch init)
                    if torch.isnan(model.model.speech_scaling_factor) or torch.isnan(model.model.speech_bias_factor):
                        scaling_factor = 1. / audio_tokens[speech_masks].flatten().std()
                        bias_factor = -audio_tokens[speech_masks].flatten().mean()
                        if torch.distributed.is_available() and torch.distributed.is_initialized():
                            torch.distributed.all_reduce(scaling_factor, op=torch.distributed.ReduceOp.SUM)
                            torch.distributed.all_reduce(bias_factor, op=torch.distributed.ReduceOp.SUM)
                            world_size = torch.distributed.get_world_size()
                            model.model.speech_scaling_factor.copy_(scaling_factor / world_size)
                            model.model.speech_bias_factor.copy_(bias_factor / world_size)
                        else:
                            model.model.speech_scaling_factor.copy_(scaling_factor)
                            model.model.speech_bias_factor.copy_(bias_factor)
                        logger.info(f"Precomputed path: initialized speech_scaling_factor={model.model.speech_scaling_factor.item():.6f}, "
                                    f"speech_bias_factor={model.model.speech_bias_factor.item():.6f}")

                    speech_all_features = (audio_tokens + model.model.speech_bias_factor) * model.model.speech_scaling_factor

                speech_all_connect_features = model.model.acoustic_connector(speech_all_features)

                if semantic_speech_all_connect_features is not None:
                    x[acoustic_input_mask] = speech_all_connect_features[speech_masks] + semantic_speech_all_connect_features[speech_masks]
                else:
                    x[acoustic_input_mask] = speech_all_connect_features[speech_masks]
                speech_features = speech_all_features[speeches_loss_input & speech_masks]
                speech_connect_features = speech_all_connect_features[speeches_loss_input & speech_masks]
            elif speeches_loss_input is not None:
                # --- Original path: encode raw waveforms ---
                speech_all_features, speech_all_connect_features = model.forward_speech_features(
                        speech_tensors=speech_tensors.type_as(x) if speech_tensors is not None else None,
                        speech_masks=speech_masks,
                        speech_type=kwargs.get("speech_type", "audio"),
                        return_unmask=True
                    )
                if speech_tensors is not None:
                    if semantic_speech_all_connect_features is not None:
                        x[acoustic_input_mask] = speech_all_connect_features[speech_masks] + semantic_speech_all_connect_features[speech_masks]
                    else:
                        x[acoustic_input_mask] = speech_all_connect_features[speech_masks]
                    speech_features = speech_all_features[speeches_loss_input & speech_masks] # only part audio need diffuse
                    speech_connect_features = speech_all_connect_features[speeches_loss_input & speech_masks]
                    # Forward-time consistency check: selected latent count should match number of acoustic placeholders
                    try:
                        if acoustic_input_mask is not None:
                            assert speech_connect_features.shape[0] == int(acoustic_input_mask.sum().item()), (
                                f"Mismatch between selected speech connectors ({speech_connect_features.shape[0]}) and acoustic_input_mask sum ({int(acoustic_input_mask.sum().item())})"
                            )
                    except Exception:
                        pass
            else:
                speech_features, speech_connect_features = model.forward_speech_features(
                        speech_tensors=speech_tensors.type_as(x) if speech_tensors is not None else None,
                        speech_masks=speech_masks,
                        speech_type=kwargs.get("speech_type", "audio"),
                    )
                if speech_tensors is not None:
                    x[acoustic_input_mask] = speech_connect_features


            autocast_dtype = None
            if model_dtype == torch.bfloat16:
                autocast_dtype = torch.bfloat16
            elif model_dtype == torch.float16:
                autocast_dtype = torch.float16

            def _forward_model():
                return model.model(
                    input_ids=None,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    inputs_embeds=x,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    output_hidden_states=False,
                    return_dict=return_dict,
                    cache_position=cache_position,
                )

            if autocast_dtype is not None and x.device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                    outputs = _forward_model()
            else:
                outputs = _forward_model()

            hidden_states = outputs.last_hidden_state
            logits = model.lm_head(hidden_states)

            if isinstance(speech_features, torch.Tensor) and speech_features.dtype != hidden_states.dtype:
                speech_features = speech_features.to(dtype=hidden_states.dtype)
            if isinstance(speech_connect_features, torch.Tensor) and speech_connect_features.dtype != hidden_states.dtype:
                speech_connect_features = speech_connect_features.to(dtype=hidden_states.dtype)

            loss = None

            # --- NEW Diffusion Loss Calculation ---
            diffusion_loss = None
            # This block is executed only if we are in a context that involves speech.
            if speech_tensors is not None and acoustic_loss_mask.sum().item() > 0:
                condition_features = hidden_states[acoustic_loss_mask]
                
                speech_len, latent_size = speech_features.shape
                # Sanity check: ensure 1:1 alignment between selected conditions and latents
                try:
                    assert condition_features.shape[0] == speech_len, (
                        f"Mismatch: condition_features={condition_features.shape[0]} vs speech_features={speech_len}"
                    )
                except Exception:
                    pass
                
                noise = torch.randn(
                    (speech_len * ddmp_batch_mul, latent_size),
                    device=hidden_states.device,
                    dtype=hidden_states.dtype
                )
                
                timesteps = torch.multinomial(
                    torch.ones(model.config.diffusion_head_config.ddpm_num_steps),
                    speech_len * ddmp_batch_mul,
                    replacement=True,
                ).to(hidden_states.device)

                speech_features_repeated = speech_features.repeat_interleave(ddmp_batch_mul, dim=0)
                condition_features_repeated = condition_features.repeat_interleave(ddmp_batch_mul, dim=0)

                noisy_speech_features = model.model.noise_scheduler.add_noise(
                    speech_features_repeated, noise, timesteps
                )
                
                model_output = model.model.prediction_head(
                    noisy_speech_features, 
                    timesteps.type_as(x), 
                    condition_features_repeated
                )

                prediction_type = model.config.diffusion_head_config.prediction_type
                if prediction_type == "epsilon":
                    target_for_loss = noise
                elif prediction_type == "v_prediction":
                    target_for_loss = model.model.noise_scheduler.get_velocity(
                        speech_features_repeated, noise, timesteps
                    )
                else:
                    raise NotImplementedError(f"Prediction type {prediction_type} not implemented")

                diffusion_loss = F.mse_loss(model_output.float(), target_for_loss.float(), reduction='sum')
                if latent_size > 0 and ddmp_batch_mul > 0 and speech_len > 0:
                    diffusion_loss = diffusion_loss / latent_size / ddmp_batch_mul / speech_len
                else:
                    diffusion_loss = torch.tensor(0.0, device=diffusion_loss.device)
            
            else:
                # Dummy loss for DDP to work when there are no speech samples in a batch,
                # but we are in a speech context.
                diffusion_loss = sum(p.sum() for p in model.model.prediction_head.parameters()) * 0.0
                diffusion_loss += sum(p.sum() for p in model.model.acoustic_connector.parameters()) * 0.0
                diffusion_loss += sum(p.sum() for p in model.model.semantic_connector.parameters()) * 0.0
            # --- End NEW Diffusion Loss Calculation ---

            from vibevoice.modular.modeling_vibevoice import VibeVoiceCausalLMOutputWithPast
            return VibeVoiceCausalLMOutputWithPast(
                loss=loss,
                diffusion_loss=diffusion_loss,
                speech_token_num=speech_len if speech_tensors is not None else 0,
                logits=logits,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )

        def compute_loss(self, model: VibeVoiceForConditionalGeneration, inputs: Dict[str, Any], return_outputs=False, num_items_in_batch: Optional[int] = None):
            labels = inputs.get("input_ids")
            attention_mask = inputs.get("attention_mask")
            acoustic_input_mask = inputs.get("acoustic_input_mask")

            # Ensure semantic tensors exist and have correct dtype/device
            sem = inputs.get("speech_semantic_tensors", None)
            if hasattr(model, "module"):
                model = model.module
            try:
                target_dtype = next(model.model.semantic_connector.parameters()).dtype
            except Exception:
                target_dtype = model.get_input_embeddings().weight.dtype

            if sem is None:
                sm = inputs.get("speech_masks")
                if sm is not None:
                    zeros = torch.zeros(
                        sm.size(0), sm.size(1),
                        getattr(model.config, "semantic_vae_dim", 128),
                        dtype=target_dtype,
                        device=sm.device,
                    )
                    inputs["speech_semantic_tensors"] = zeros
            else:
                if isinstance(sem, torch.Tensor):
                    inputs["speech_semantic_tensors"] = sem.to(dtype=target_dtype)

            # Use custom training forward pass with new diffusion loss
            outputs = self.training_forward(model, inputs)

            # Invariants: token/latent selection equality across views (warn, don't assert)
            # try:
            #     al_mask = inputs.get("acoustic_loss_mask")
            #     sp_masks = inputs.get("speech_masks")
            #     sp_loss_sel = inputs.get("speeches_loss_input")
            #     num_tok_total = int(acoustic_input_mask.sum().item()) if acoustic_input_mask is not None else 0
            #     num_tok_loss = int(al_mask.sum().item()) if al_mask is not None else 0
            #     num_lat_total = int(sp_masks.sum().item()) if sp_masks is not None else 0
            #     num_lat_loss = int(((sp_loss_sel & sp_masks).sum().item())) if (sp_loss_sel is not None and sp_masks is not None) else 0
            #     self.log({
            #         "debug/num_tok_total": float(num_tok_total),
            #         "debug/num_tok_loss": float(num_tok_loss),
            #         "debug/num_lat_total": float(num_lat_total),
            #         "debug/num_lat_loss": float(num_lat_loss),
            #     })
            #     if sp_loss_sel is not None and sp_masks is not None and al_mask is not None:
            #         if num_tok_loss != num_lat_loss:
            #             logger.warning(f"Loss selection mismatch: acoustic_loss_mask={num_tok_loss} vs speeches_loss_input={num_lat_loss}")
            # except Exception:
            #     pass

            # CE Loss
            logits = outputs.logits
            ce_labels = mask_for_ce(labels, attention_mask, acoustic_input_mask, pad_id=-100)
            shift_logits = logits[:, :-1, :].contiguous()
            ce_loss = fast_cross_entropy_loss(shift_logits, ce_labels)

            # Optional CE diagnostics
            # try:
            #     self._debug_ce(shift_logits, ce_labels, attention_mask, acoustic_input_mask)
            # except Exception as e:
            #     logger.warning(f"Failed invoking CE debug: {e}")

            # Diffusion loss
            diffusion_loss = outputs.diffusion_loss if outputs.diffusion_loss is not None else torch.tensor(0.0, device=ce_loss.device)
            total = training_args.ce_loss_weight * ce_loss + training_args.diffusion_loss_weight * diffusion_loss

            # Accumulate component losses for averaged logging
            _ce = ce_loss.detach().item()
            _diff = diffusion_loss.detach().item() if isinstance(diffusion_loss, torch.Tensor) else float(diffusion_loss)
            if model.training:
                self._ce_loss_acc += _ce
                self._diffusion_loss_acc += _diff
                self._loss_count += 1
            else:
                self._eval_ce_loss_acc += _ce
                self._eval_diffusion_loss_acc += _diff
                self._eval_loss_count += 1

            return (total, outputs) if return_outputs else total

        def _debug_ce(self, shift_logits: torch.Tensor, ce_labels: torch.Tensor, attention_mask: Optional[torch.Tensor], acoustic_input_mask: Optional[torch.Tensor]):
            try:
                if not getattr(training_args, "debug_ce_details", False):
                    return
                step = int(getattr(self.state, "global_step", 0) or 0)
                every_n = max(1, int(getattr(training_args, "debug_ce_every_n_steps", 200) or 200))
                if not (step <= 1 or (step % every_n == 0)):
                    return

                with torch.no_grad():
                    vocab = shift_logits.size(-1)
                    per_token_loss = F.cross_entropy(
                        shift_logits.view(-1, vocab),
                        ce_labels.view(-1),
                        reduction="none",
                        ignore_index=-100,
                    ).view_as(ce_labels)

                    valid_mask = ce_labels.ne(-100)
                    num_valid = int(valid_mask.sum().item())
                    avg_loss = float((per_token_loss[valid_mask].mean().item())) if num_valid > 0 else float("nan")

                    per_ex_avgs = []
                    max_examples = max(1, int(getattr(training_args, "debug_ce_max_examples", 1) or 1))
                    B = ce_labels.size(0)
                    for b in range(min(B, max_examples)):
                        vb = valid_mask[b]
                        if int(vb.sum().item()) > 0:
                            per_ex_avgs.append(float(per_token_loss[b][vb].mean().item()))
                        else:
                            per_ex_avgs.append(float("nan"))
                    logger.info(f"CE debug: tokens_in_loss={num_valid}, avg_loss={avg_loss:.4f}, per_example_avgs={[round(x,4) if x==x else None for x in per_ex_avgs]}")
            except Exception as e:
                logger.warning(f"CE detailed debug failed: {e}")

        # --------- CHECKPOINT SAVE/LOAD ---------

        def _save(self, output_dir: Optional[str] = None, state_dict=None) -> None:
            try:
                target_dir = output_dir or self.args.output_dir
                os.makedirs(target_dir, exist_ok=True)

                unwrapped = self.model
                while hasattr(unwrapped, "module"):
                    unwrapped = unwrapped.module

                # Always save the full model. Use nn.Module.state_dict() to
                # bypass any PEFT filtering so all weights are included.
                # We save the actual training weights (NOT EMA) so that
                # the optimizer state remains consistent on resume.
                save_sd = state_dict if state_dict is not None else nn.Module.state_dict(unwrapped)
                unwrapped.save_pretrained(target_dir, state_dict=save_sd, safe_serialization=True)
                logger.info(f"Full model saved to {target_dir}")

                # Save EMA shadow separately so it can be restored on resume
                ema_cb = self._get_ema_callback()
                if ema_cb and ema_cb.shadow is not None:
                    ema_path = os.path.join(target_dir, "ema_shadow.pt")
                    ema_cb.save_shadow(ema_path)
                    logger.info(f"EMA shadow saved to {ema_path}")

                # Additionally save LoRA adapters if any component is PEFT-wrapped
                lm = getattr(unwrapped.model, "language_model", None)
                ph = getattr(unwrapped.model, "prediction_head", None)
                has_lm_lora = lm is not None and hasattr(lm, "peft_config")
                has_ph_lora = ph is not None and hasattr(ph, "peft_config")
                if has_lm_lora:
                    lora_out = os.path.join(target_dir, "lora")
                    os.makedirs(lora_out, exist_ok=True)
                    lm.save_pretrained(lora_out)
                    logger.info(f"LLM LoRA adapters saved to {lora_out}")
                if has_ph_lora:
                    ph_dir = os.path.join(target_dir, "lora", "diffusion_head")
                    os.makedirs(ph_dir, exist_ok=True)
                    ph.save_pretrained(ph_dir)
                    logger.info(f"Diffusion head LoRA adapters saved to {ph_dir}")

                # Save merged model (LoRA folded into base weights) for direct inference
                if getattr(self.args, "save_merged", False) and (has_lm_lora or has_ph_lora):
                    try:
                        merged_dir = os.path.join(target_dir, "merged")
                        os.makedirs(merged_dir, exist_ok=True)

                        # Merge LoRA deltas into base weights (in-place, reversible)
                        if has_lm_lora:
                            lm.merge_adapter()
                        if has_ph_lora:
                            ph.merge_adapter()

                        # Build clean state dict: drop LoRA params, clean PEFT key prefixes
                        merged_sd = {}
                        for name, param in unwrapped.named_parameters():
                            if any(k in name for k in ("lora_A", "lora_B", "lora_embedding", "lora_magnitude", "ranknum")):
                                continue
                            clean = name.replace(".base_model.model.", ".").replace(".base_layer", "")
                            merged_sd[clean] = param.data
                        for name, buf in unwrapped.named_buffers():
                            clean = name.replace(".base_model.model.", ".").replace(".base_layer", "")
                            merged_sd[clean] = buf

                        unwrapped.save_pretrained(merged_dir, state_dict=merged_sd, safe_serialization=True)
                        logger.info(f"Merged model saved to {merged_dir}")

                        # Unmerge to restore original weights for continued training
                        if has_lm_lora:
                            lm.unmerge_adapter()
                        if has_ph_lora:
                            ph.unmerge_adapter()
                    except Exception as e:
                        logger.warning(f"Failed to save merged model: {e}")
                        try:
                            if has_lm_lora:
                                lm.unmerge_adapter()
                            if has_ph_lora:
                                ph.unmerge_adapter()
                        except Exception:
                            pass

            except Exception as e:
                logger.warning(f"Failed to save model: {e}")

        def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
            target_model = model if model is not None else self.model

            # New format: full model at checkpoint root
            has_full_model = (
                os.path.isfile(os.path.join(resume_from_checkpoint, "model.safetensors")) or
                os.path.isfile(os.path.join(resume_from_checkpoint, "pytorch_model.bin"))
            )
            if has_full_model:
                super()._load_from_checkpoint(resume_from_checkpoint, model=target_model)
                logger.info(f"Loaded model checkpoint from {resume_from_checkpoint}")

                # Restore EMA shadow if saved alongside the checkpoint
                ema_cb = self._get_ema_callback()
                if ema_cb is not None:
                    ema_path = os.path.join(resume_from_checkpoint, "ema_shadow.pt")
                    if ema_cb.load_shadow(ema_path):
                        logger.info(f"Restored EMA shadow from {ema_path}")
                    else:
                        logger.info("No ema_shadow.pt in checkpoint; EMA will initialize from loaded weights on train_begin")
                return

            # Legacy format: component-level saves under lora/ directory
            lora_dir = os.path.join(resume_from_checkpoint, "lora")
            if os.path.isdir(lora_dir):
                self._load_legacy_checkpoint(resume_from_checkpoint, target_model)
                return

            raise ValueError(f"Can't find a valid checkpoint at {resume_from_checkpoint}")

        def _load_legacy_checkpoint(self, checkpoint_dir: str, model=None) -> None:
            """Load from pre-simplification checkpoint format (lora/ directory with separate component files)."""
            target_model = model if model is not None else self.model
            target_model = getattr(target_model, "module", target_model)
            lora_dir = os.path.join(checkpoint_dir, "lora")

            def _load_file(path):
                try:
                    return torch.load(path, map_location="cpu", weights_only=True)
                except TypeError:
                    return torch.load(path, map_location="cpu")

            loaded = []
            errors = []

            # LLM LoRA adapters
            lm = getattr(target_model.model, "language_model", None)
            if os.path.isfile(os.path.join(lora_dir, "adapter_config.json")) and hasattr(lm, "peft_config"):
                try:
                    from peft import load_peft_weights, set_peft_model_state_dict
                    adapter_name = lm.active_adapters[0] if hasattr(lm, "active_adapters") and lm.active_adapters else "default"
                    set_peft_model_state_dict(lm, load_peft_weights(lora_dir, device="cpu"), adapter_name=adapter_name)
                    loaded.append("LLM LoRA")
                except Exception as e:
                    errors.append(f"LLM LoRA: {e}")

            # Diffusion head
            ph = getattr(target_model.model, "prediction_head", None)
            for ph_path in [os.path.join(lora_dir, "diffusion_head_full.bin"),
                            os.path.join(lora_dir, "diffusion_head", "diffusion_head_full.bin")]:
                if os.path.isfile(ph_path) and ph is not None:
                    try:
                        ph.load_state_dict(_load_file(ph_path), strict=False)
                        loaded.append("diffusion_head")
                    except Exception as e:
                        errors.append(f"diffusion_head: {e}")
                    break

            # Connectors
            for name, attr in [("acoustic_connector", "acoustic_connector"), ("semantic_connector", "semantic_connector")]:
                mod = getattr(target_model.model, attr, None)
                path = os.path.join(lora_dir, name, "pytorch_model.bin")
                if os.path.isfile(path) and mod is not None:
                    try:
                        mod.load_state_dict(_load_file(path), strict=False)
                        loaded.append(name)
                    except Exception as e:
                        errors.append(f"{name}: {e}")

            # Embeddings / LM head
            emb_dir = os.path.join(lora_dir, "embeddings")
            for fname, getter in [("input_embeddings.bin", "get_input_embeddings"), ("lm_head.bin", "get_output_embeddings")]:
                path = os.path.join(emb_dir, fname)
                if os.path.isfile(path):
                    try:
                        sd = _load_file(path)
                        getattr(target_model, getter)().weight.data.copy_(sd["weight"])
                        loaded.append(fname)
                    except Exception as e:
                        errors.append(f"{fname}: {e}")

            if errors and not loaded:
                raise ValueError(f"Failed to load legacy checkpoint from {lora_dir}: {'; '.join(errors)}")
            if errors:
                logger.warning(f"Partial legacy checkpoint load: {'; '.join(errors)}")
            logger.info(f"Loaded legacy checkpoint components: {', '.join(loaded)} from {lora_dir}")


    # ------------- Build the Trainer -------------

    # Resolve which adapters to apply in samples

    callbacks = [EmaCallback(attr_path="model.prediction_head", decay=0.999, device="cpu")]

    if data_args.validation_jsonl or (data_args.eval_inference_text and data_args.eval_inference_voice_prompt):
        eval_text = data_args.eval_inference_text
        if eval_text and os.path.isfile(eval_text):
            with open(eval_text, "r", encoding="utf-8") as f:
                eval_text = f.read().strip()
        voice_prompt_str = data_args.eval_inference_voice_prompt or ""
        voice_paths = [p.strip() for p in voice_prompt_str.split(",") if p.strip()]
        inference_cb = InferenceEvalCallback(
            processor=processor,
            eval_text=eval_text,
            eval_file=data_args.validation_jsonl,
            voice_prompt_paths=voice_paths,
            cfg_scale=data_args.eval_inference_cfg_scale,
            eval_on_start=training_args.eval_on_start,
            ema_callback=callbacks[0],
        )
        callbacks.append(inference_cb)
        logger.info(f"Eval inference enabled: {len(voice_paths)} voice prompt(s), cfg_scale={data_args.eval_inference_cfg_scale}")

    if model_args.use_llm_lora:
        callbacks.append(LoRADebugCallback(log_every_n_steps=(int(getattr(training_args, "logging_steps", 50) or 50))))

    trainer = VibeVoiceTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=callbacks,
    )

    # Load weights from a checkpoint without restoring optimizer/scheduler/step state
    if getattr(training_args, "init_from_checkpoint", None) is not None and getattr(training_args, "resume_from_checkpoint", None) is not None:
        raise ValueError(
            "--init_from_checkpoint cannot be used together with --resume_from_checkpoint. "
            "Use init_from_checkpoint to load only model weights, or resume_from_checkpoint to fully resume training state.")
    if getattr(training_args, "init_from_checkpoint", None):
        trainer._load_from_checkpoint(training_args.init_from_checkpoint)

    # Optional debug pre-training save
    if getattr(training_args, "debug_save", False):
        debug_dir = os.path.join(training_args.output_dir, "debug_initial")
        logger.info(f"[debug_save] Saving initial model to: {debug_dir}")
        trainer._save(debug_dir)

    if training_args.do_train:
        trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        final_dir = os.path.join(training_args.output_dir, "final")
        logger.info(f"Training complete. Saving final model to {final_dir}")
        trainer._save(final_dir)

    if training_args.do_eval and eval_dataset is not None:
        trainer.evaluate()


if __name__ == "__main__":
    main()