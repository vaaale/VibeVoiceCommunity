"""
Build a VibeVoice model directory using Qwen3-0.6B-Base as the decoder backbone.

This script:
1. Creates a VibeVoiceForConditionalGeneration model from the Qwen3 config
2. Loads pretrained Qwen3 Norwegian weights into the language_model sub-module
3. Copies acoustic/semantic tokenizer + diffusion head weights from the existing VibeVoice-1.5B
4. Saves the complete model + config to the target directory

Usage:
    python scripts/build_vibevoice_qwen3.py
"""

import json
import os
import sys
import shutil

import torch
from safetensors.torch import load_file, save_file

# Ensure the project root is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from vibevoice.modular.configuration_vibevoice import (
    VibeVoiceConfig,
    VibeVoiceAcousticTokenizerConfig,
    VibeVoiceSemanticTokenizerConfig,
    VibeVoiceDiffusionHeadConfig,
)
from vibevoice.modular.modeling_vibevoice import (
    VibeVoiceForConditionalGeneration,
    VibeVoiceModel,
)

# ── Configuration ────────────────────────────────────────────────────────────

# Path to the VibeVoice Qwen3 config JSON we created earlier
VIBEVOICE_QWEN3_CONFIG = os.path.join(
    os.path.dirname(__file__), "..", "vibevoice", "configs", "qwen3_0.6b_32k.json"
)

# Pretrained Qwen3-0.6B Norwegian (merged 16-bit) — provides language_model weights
QWEN3_PRETRAINED_DIR = "/mnt/storage/Models/Qwen3-0.6B-Norwegian/norwegianmerged_16bit"

# Existing VibeVoice-1.5B model — provides acoustic/semantic tokenizer + diffusion head weights
VIBEVOICE_DONOR_MODEL = "vibevoice/VibeVoice-1.5B"

# Output directory
OUTPUT_DIR = "/mnt/storage/Models/VibeVoice-Qwen3/norwegian-qwen3-base"


def load_vibevoice_config() -> VibeVoiceConfig:
    """Load the VibeVoice config from the Qwen3 JSON template."""
    with open(VIBEVOICE_QWEN3_CONFIG, "r") as f:
        cfg_dict = json.load(f)

    # Ensure model_type is vibevoice (the JSON template uses "vibepod" for legacy compat)
    cfg_dict["model_type"] = "vibevoice"
    cfg_dict["acoustic_tokenizer_config"]["model_type"] = "vibevoice_acoustic_tokenizer"
    cfg_dict["semantic_tokenizer_config"]["model_type"] = "vibevoice_semantic_tokenizer"
    cfg_dict["diffusion_head_config"]["model_type"] = "vibevoice_diffusion_head"

    config = VibeVoiceConfig(**cfg_dict)
    return config


def load_qwen3_state_dict(qwen3_dir: str) -> dict:
    """Load the pretrained Qwen3 state dict from safetensors."""
    safetensors_path = os.path.join(qwen3_dir, "model.safetensors")
    if os.path.exists(safetensors_path):
        print(f"  Loading Qwen3 weights from {safetensors_path}")
        return load_file(safetensors_path, device="cpu")
    
    # Try sharded format
    index_path = os.path.join(qwen3_dir, "model.safetensors.index.json")
    if os.path.exists(index_path):
        print(f"  Loading sharded Qwen3 weights from {qwen3_dir}")
        with open(index_path, "r") as f:
            index = json.load(f)
        weight_map = index["weight_map"]
        seen_files = set()
        full_state_dict = {}
        for param_name, filename in weight_map.items():
            if filename not in seen_files:
                seen_files.add(filename)
                shard = load_file(os.path.join(qwen3_dir, filename), device="cpu")
                full_state_dict.update(shard)
        return full_state_dict

    raise FileNotFoundError(f"No model.safetensors found in {qwen3_dir}")


def load_donor_state_dict(donor_id: str) -> dict:
    """Load the donor VibeVoice-1.5B state dict (from HF hub or local)."""
    from huggingface_hub import snapshot_download
    
    if os.path.isdir(donor_id):
        model_dir = donor_id
    else:
        print(f"  Downloading donor model: {donor_id}")
        model_dir = snapshot_download(donor_id)
    
    # Try single file first
    single = os.path.join(model_dir, "model.safetensors")
    if os.path.exists(single):
        return load_file(single, device="cpu")
    
    # Try sharded
    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.exists(index_path):
        print(f"  Loading sharded donor weights from {model_dir}")
        with open(index_path, "r") as f:
            index = json.load(f)
        weight_map = index["weight_map"]
        seen_files = set()
        full_state_dict = {}
        for param_name, filename in weight_map.items():
            if filename not in seen_files:
                seen_files.add(filename)
                shard = load_file(os.path.join(model_dir, filename), device="cpu")
                full_state_dict.update(shard)
        return full_state_dict

    raise FileNotFoundError(f"No safetensors found in {model_dir}")


def remap_qwen3_keys(qwen3_sd: dict) -> dict:
    """
    Remap Qwen3ForCausalLM state dict keys to VibeVoice's model.language_model.* namespace.
    
    Qwen3ForCausalLM uses keys like:
        model.embed_tokens.weight
        model.layers.0.self_attn.q_proj.weight
        ...
        model.norm.weight
    
    VibeVoice stores the LM as:
        model.language_model.embed_tokens.weight
        model.language_model.layers.0.self_attn.q_proj.weight
        ...
        model.language_model.norm.weight
    
    The lm_head is stored directly as:
        lm_head.weight
    """
    remapped = {}
    for key, tensor in qwen3_sd.items():
        if key.startswith("model."):
            # model.X -> model.language_model.X
            new_key = "model.language_model." + key[len("model."):]
            remapped[new_key] = tensor
        elif key == "lm_head.weight":
            remapped["lm_head.weight"] = tensor
        else:
            # Keep as-is (shouldn't happen for standard Qwen3)
            remapped[key] = tensor
    return remapped


def extract_donor_speech_components(donor_sd: dict) -> dict:
    """
    Extract acoustic tokenizer, semantic tokenizer, connectors, diffusion head,
    and speech scaling buffers from the donor VibeVoice-1.5B state dict.
    """
    prefixes = (
        "model.acoustic_tokenizer.",
        "model.semantic_tokenizer.",
        "model.acoustic_connector.",
        "model.semantic_connector.",
        "model.prediction_head.",
        "model.speech_scaling_factor",
        "model.speech_bias_factor",
    )
    speech_sd = {}
    for key, tensor in donor_sd.items():
        if any(key.startswith(p) for p in prefixes):
            speech_sd[key] = tensor
    return speech_sd


def main():
    print("=" * 60)
    print("Building VibeVoice-Qwen3 model directory")
    print("=" * 60)

    # ── 1. Register custom configs ──────────────────────────────────────
    AutoConfig.register("vibevoice", VibeVoiceConfig)
    AutoConfig.register("vibevoice_acoustic_tokenizer", VibeVoiceAcousticTokenizerConfig)
    AutoConfig.register("vibevoice_semantic_tokenizer", VibeVoiceSemanticTokenizerConfig)
    AutoConfig.register("vibevoice_diffusion_head", VibeVoiceDiffusionHeadConfig)
    AutoModelForCausalLM.register(VibeVoiceConfig, VibeVoiceForConditionalGeneration)

    # ── 2. Load config ──────────────────────────────────────────────────
    print("\n[1/6] Loading VibeVoice Qwen3 config...")
    config = load_vibevoice_config()
    print(f"  decoder_config type: {type(config.decoder_config).__name__}")
    print(f"  decoder hidden_size: {config.decoder_config.hidden_size}")
    print(f"  diffusion hidden_size: {config.diffusion_head_config.hidden_size}")

    # ── 3. Create fresh model from config ───────────────────────────────
    print("\n[2/6] Creating VibeVoiceForConditionalGeneration from config...")
    model = VibeVoiceForConditionalGeneration(config)
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ── 4. Load pretrained Qwen3 weights ────────────────────────────────
    print(f"\n[3/6] Loading pretrained Qwen3 weights from:\n  {QWEN3_PRETRAINED_DIR}")
    qwen3_sd = load_qwen3_state_dict(QWEN3_PRETRAINED_DIR)
    print(f"  Loaded {len(qwen3_sd)} tensors from Qwen3")

    # Remap keys to VibeVoice namespace
    remapped_qwen3 = remap_qwen3_keys(qwen3_sd)
    print(f"  Remapped to {len(remapped_qwen3)} VibeVoice keys")

    # Load into model (strict=False since speech components are missing)
    missing, unexpected = model.load_state_dict(remapped_qwen3, strict=False)
    print(f"  Missing keys (speech components, expected): {len(missing)}")
    print(f"  Unexpected keys: {len(unexpected)}")
    if unexpected:
        print(f"  WARNING unexpected keys: {unexpected[:10]}")

    # ── 5. Load donor speech components from VibeVoice-1.5B ─────────────
    print(f"\n[4/6] Loading donor speech components from:\n  {VIBEVOICE_DONOR_MODEL}")
    donor_sd = load_donor_state_dict(VIBEVOICE_DONOR_MODEL)
    speech_sd = extract_donor_speech_components(donor_sd)
    print(f"  Extracted {len(speech_sd)} speech component tensors")

    # Check for dimension mismatches in connectors (1.5B has hidden_size=1536, Qwen3 has 1024)
    # Build a combined lookup of all model parameters and buffers
    model_state = {n: p for n, p in model.named_parameters()}
    model_state.update({n: b for n, b in model.named_buffers()})

    connector_mismatches = []
    for key, tensor in speech_sd.items():
        if "connector" in key:
            model_param = model_state.get(key)
            if model_param is not None and model_param.shape != tensor.shape:
                connector_mismatches.append((key, tensor.shape, model_param.shape))

    if connector_mismatches:
        print(f"\n  ⚠ Connector dimension mismatches (donor vs target):")
        for key, donor_shape, target_shape in connector_mismatches:
            print(f"    {key}: {donor_shape} -> {target_shape}")
        print("  → Connectors will be randomly initialized (they'll be trained)")
        # Remove mismatched connector keys
        for key, _, _ in connector_mismatches:
            del speech_sd[key]

    # Check diffusion head dimension mismatch
    head_mismatches = []
    for key, tensor in list(speech_sd.items()):
        if "prediction_head" in key:
            model_param = model_state.get(key)
            if model_param is not None and model_param.shape != tensor.shape:
                head_mismatches.append((key, tensor.shape, model_param.shape))

    if head_mismatches:
        print(f"\n  ⚠ Diffusion head dimension mismatches (donor vs target):")
        for key, donor_shape, target_shape in head_mismatches:
            print(f"    {key}: {donor_shape} -> {target_shape}")
        print("  → Diffusion head will be randomly initialized (it'll be trained)")
        for key, _, _ in head_mismatches:
            del speech_sd[key]

    # Load compatible speech components
    if speech_sd:
        missing2, unexpected2 = model.load_state_dict(speech_sd, strict=False)
        loaded_count = len(speech_sd) - len(unexpected2)
        print(f"  Loaded {loaded_count} compatible speech component tensors")
        if unexpected2:
            print(f"  Unexpected: {unexpected2[:5]}")

    # ── 6. Save model ───────────────────────────────────────────────────
    print(f"\n[5/6] Saving model to:\n  {OUTPUT_DIR}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Save with safetensors
    model.save_pretrained(OUTPUT_DIR, safe_serialization=True)
    print(f"  Model saved")

    # ── 7. Copy tokenizer + processor files ────────────────────────────
    print(f"\n[6/6] Copying tokenizer and processor files...")

    # Copy preprocessor_config.json from the Qwen3 processor directory
    processor_config_src = os.path.join(
        os.path.dirname(__file__), "..", "vibevoice", "processor_qwen3_0.6b", "preprocessor_config.json"
    )
    processor_config_dst = os.path.join(OUTPUT_DIR, "preprocessor_config.json")
    if os.path.exists(processor_config_src):
        shutil.copy2(processor_config_src, processor_config_dst)
        print(f"  Copied preprocessor_config.json from processor_qwen3_0.6b/")
    else:
        print(f"  WARNING: preprocessor_config.json not found at {processor_config_src}")

    tokenizer_files = [
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
        "special_tokens_map.json",
        "added_tokens.json",
    ]
    for fname in tokenizer_files:
        src = os.path.join(QWEN3_PRETRAINED_DIR, fname)
        dst = os.path.join(OUTPUT_DIR, fname)
        if os.path.exists(src):
            shutil.copy2(src, dst)
            print(f"  Copied {fname}")
        else:
            print(f"  Skipped {fname} (not found in source)")

    # ── Summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Done! Model directory contents:")
    print("=" * 60)
    for f in sorted(os.listdir(OUTPUT_DIR)):
        fpath = os.path.join(OUTPUT_DIR, f)
        if os.path.isfile(fpath):
            size_mb = os.path.getsize(fpath) / (1024 * 1024)
            print(f"  {f:40s} {size_mb:8.1f} MB")

    total_params = sum(p.numel() for p in model.parameters())
    lm_params = sum(p.numel() for n, p in model.named_parameters() if "language_model" in n or "lm_head" in n)
    speech_params = total_params - lm_params
    print(f"\n  Total parameters:    {total_params:>12,}")
    print(f"  LM parameters:       {lm_params:>12,}")
    print(f"  Speech parameters:   {speech_params:>12,}")
    print(f"\n  Output: {OUTPUT_DIR}")
    print(f"  Config model_type: {config.model_type}")
    print(f"  Decoder model_type: {config.decoder_config.model_type}")


if __name__ == "__main__":
    main()
