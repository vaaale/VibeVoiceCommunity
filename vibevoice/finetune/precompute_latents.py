"""
Pre-compute acoustic and semantic latents for VibeVoice training.

This script encodes all audio (target + voice prompts) through the frozen
acoustic and semantic tokenizers and saves the results as a Hugging Face
Dataset (Arrow format) for efficient memory-mapped loading during training.

Usage:
    python -m vibevoice.finetune.precompute_latents \
        --model_name_or_path /path/to/vibevoice-model \
        --input_jsonl /path/to/dataset.jsonl \
        --output_dir /path/to/precomputed_dataset \
        [--processor_name_or_path /path/to/processor] \
        [--device cuda]
"""

import argparse
import json
import logging
import math
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch

try:
    import librosa
except ImportError:
    librosa = None

from tqdm import tqdm

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def _load_audio_to_24k(audio_path: str, target_sr: int = 24000) -> np.ndarray:
    if librosa is None:
        raise RuntimeError("librosa is required. pip install librosa")
    wav, sr = librosa.load(audio_path, sr=None, mono=True)
    if int(sr) != target_sr:
        wav = librosa.resample(y=wav.astype(np.float32), orig_sr=int(sr), target_sr=target_sr)
    return wav.astype(np.float32)


def _encode_acoustic(acoustic_tokenizer, wav: np.ndarray, device: torch.device) -> np.ndarray:
    """Encode waveform -> acoustic mean latents [T, vae_dim]."""
    wav_t = torch.from_numpy(wav).unsqueeze(0).unsqueeze(0).to(device=device, dtype=torch.float32)
    with torch.no_grad():
        enc_out = acoustic_tokenizer.encode(wav_t)
    # enc_out is VibeVoiceTokenizerEncoderOutput with .mean [1, T, D]
    mean = enc_out.mean
    if mean.dim() == 3:
        mean = mean[0]  # [T, D]
    return mean.cpu().float().numpy()


def _encode_semantic(semantic_tokenizer, wav: np.ndarray, device: torch.device) -> np.ndarray:
    """Encode waveform -> semantic latents [T, sem_dim]."""
    wav_t = torch.from_numpy(wav).unsqueeze(0).unsqueeze(0).to(device=device, dtype=torch.float32)
    with torch.no_grad():
        enc_out = semantic_tokenizer.encode(wav_t)
    mean = enc_out.mean
    if mean.dim() == 3:
        mean = mean[0]  # [T, D]
    return mean.cpu().float().numpy()


def main():
    parser = argparse.ArgumentParser(description="Pre-compute latents for VibeVoice training")
    parser.add_argument("--model_name_or_path", type=str, required=True,
                        help="Path to VibeVoice model (contains acoustic/semantic tokenizers)")
    parser.add_argument("--processor_name_or_path", type=str, default=None,
                        help="Path to processor dir. Defaults to model path.")
    parser.add_argument("--input_jsonl", type=str, required=True,
                        help="Path to input JSONL dataset")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory where the HF Dataset with latent columns will be saved")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "bfloat16", "float16"])
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype_map = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}
    dtype = dtype_map[args.dtype]

    # --- Load model to get tokenizers ---
    logger.info(f"Loading model from {args.model_name_or_path}")

    from transformers import AutoConfig, AutoModelForCausalLM
    from vibevoice.modular.configuration_vibevoice import (
        VibeVoiceConfig, VibeVoiceAcousticTokenizerConfig,
        VibeVoiceSemanticTokenizerConfig, VibeVoiceDiffusionHeadConfig,
    )
    from vibevoice.modular.modeling_vibevoice import VibeVoiceForConditionalGeneration

    AutoConfig.register("vibevoice", VibeVoiceConfig)
    AutoConfig.register("vibevoice_acoustic_tokenizer", VibeVoiceAcousticTokenizerConfig)
    AutoConfig.register("vibevoice_semantic_tokenizer", VibeVoiceSemanticTokenizerConfig)
    AutoConfig.register("vibevoice_diffusion_head", VibeVoiceDiffusionHeadConfig)
    AutoModelForCausalLM.register(VibeVoiceConfig, VibeVoiceForConditionalGeneration)

    model = VibeVoiceForConditionalGeneration.from_pretrained(
        args.model_name_or_path, torch_dtype=dtype,
    ).to(device).eval()

    acoustic_tokenizer = model.model.acoustic_tokenizer
    semantic_tokenizer = model.model.semantic_tokenizer
    fix_std = float(acoustic_tokenizer.fix_std.item())
    std_dist_type = acoustic_tokenizer.std_dist_type

    logger.info(f"Acoustic VAE dim: {model.config.acoustic_vae_dim}, fix_std: {fix_std}, dist_type: {std_dist_type}")
    logger.info(f"Semantic VAE dim: {model.config.semantic_vae_dim}")

    # --- Load processor for audio normalization ---
    processor_path = args.processor_name_or_path or args.model_name_or_path
    from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor
    processor = VibeVoiceProcessor.from_pretrained(processor_path)

    # --- Read input JSONL ---
    input_jsonl = Path(args.input_jsonl)
    base_dir = input_jsonl.parent
    with open(input_jsonl, "r") as f:
        samples = [json.loads(line) for line in f if line.strip()]
    logger.info(f"Loaded {len(samples)} samples from {input_jsonl}")

    output_dir = Path(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # --- Pre-compute scaling factors from first batch ---
    # We compute global scaling/bias from a subset of samples, matching the
    # runtime logic in forward_speech_features.
    logger.info("Computing global scaling/bias factors from first 100 samples...")
    all_vals = []
    subset = samples[:min(100, len(samples))]
    for s in tqdm(subset, desc="Scaling stats"):
        audio_path = s.get("audio", "")
        if not os.path.isabs(audio_path):
            audio_path = str(base_dir / audio_path)
        try:
            wav = _load_audio_to_24k(audio_path)
            if processor.db_normalize and processor.audio_normalizer:
                wav = processor.audio_normalizer(wav)
            mean = _encode_acoustic(acoustic_tokenizer, wav, device)
            # Apply sampling (matching forward_speech_features which computes stats
            # from sampled audio_tokens, not raw means)
            mean_t = torch.from_numpy(mean)
            if std_dist_type == 'gaussian':
                batch_size = 1
                value = fix_std / 0.8
                std_val = torch.randn(batch_size) * value
                sampled = mean_t + std_val.view(-1, 1) * torch.randn_like(mean_t)
            elif std_dist_type == 'fix':
                sampled = mean_t + fix_std * torch.randn_like(mean_t)
            else:
                sampled = mean_t
            all_vals.append(sampled.numpy().flatten())
        except Exception as e:
            logger.warning(f"Skipping {audio_path} for stats: {e}")

    all_vals_np = np.concatenate(all_vals)
    bias_factor = float(-all_vals_np.mean())
    scaling_factor = float(1.0 / all_vals_np.std())
    logger.info(f"Computed scaling_factor={scaling_factor:.6f}, bias_factor={bias_factor:.6f}")

    # Save scaling factors alongside dataset
    meta = {
        "scaling_factor": scaling_factor,
        "bias_factor": bias_factor,
        "fix_std": fix_std,
        "std_dist_type": std_dist_type,
        "acoustic_vae_dim": model.config.acoustic_vae_dim,
        "semantic_vae_dim": model.config.semantic_vae_dim,
    }
    meta_path = output_dir / "latents_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    logger.info(f"Saved metadata to {meta_path}")

    # --- Encode all samples ---
    from datasets import Dataset

    output_rows = []
    skipped = 0

    for idx, sample in enumerate(tqdm(samples, desc="Encoding latents")):
        try:
            row = dict(sample)  # preserve all original fields

            # --- Target audio ---
            audio_path = sample.get("audio", "")
            if not os.path.isabs(audio_path):
                audio_path = str(base_dir / audio_path)

            wav_target = _load_audio_to_24k(audio_path)
            if processor.db_normalize and processor.audio_normalizer:
                wav_target = processor.audio_normalizer(wav_target)

            acoustic_mean = _encode_acoustic(acoustic_tokenizer, wav_target, device)
            semantic_feat = _encode_semantic(semantic_tokenizer, wav_target, device)

            row["target_acoustic_mean"] = acoustic_mean.tolist()
            row["target_semantic_feat"] = semantic_feat.tolist()
            row["target_latent_len"] = int(acoustic_mean.shape[0])

            # --- Voice prompts ---
            voice_prompts = sample.get("voice_prompts", None)
            prompt_acs = []
            prompt_ses = []
            prompt_lens = []
            if voice_prompts:
                if not isinstance(voice_prompts, list):
                    voice_prompts = [voice_prompts]

                for pi, vp in enumerate(voice_prompts):
                    vp_path = vp if os.path.isabs(vp) else str(base_dir / vp)
                    wav_vp = _load_audio_to_24k(vp_path)
                    if processor.db_normalize and processor.audio_normalizer:
                        wav_vp = processor.audio_normalizer(wav_vp)

                    vp_ac_mean = _encode_acoustic(acoustic_tokenizer, wav_vp, device)
                    vp_se_feat = _encode_semantic(semantic_tokenizer, wav_vp, device)

                    prompt_acs.append(vp_ac_mean.tolist())
                    prompt_ses.append(vp_se_feat.tolist())
                    prompt_lens.append(int(vp_ac_mean.shape[0]))

            row["prompt_acoustic_latents"] = prompt_acs
            row["prompt_semantic_latents"] = prompt_ses
            row["prompt_latent_lens"] = prompt_lens

            output_rows.append(row)

        except Exception as e:
            logger.warning(f"Failed to process sample {idx}: {e}")
            skipped += 1

    # --- Save as HF Dataset ---
    logger.info(f"Building HF Dataset from {len(output_rows)} rows...")
    ds = Dataset.from_list(output_rows)
    ds.save_to_disk(str(output_dir / "dataset"))

    logger.info(f"Done. Saved dataset to {output_dir / 'dataset'} ({skipped} skipped)")
    logger.info(f"Metadata at {meta_path}")


if __name__ == "__main__":
    main()
