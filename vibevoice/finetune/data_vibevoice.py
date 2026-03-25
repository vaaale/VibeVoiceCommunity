import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import warnings
import random

try:
    import librosa  # type: ignore
except Exception:  # pragma: no cover
    librosa = None  # Fallback: user must install librosa when using local audio paths

try:
    import resampy  # type: ignore
except Exception:  # pragma: no cover
    resampy = None


def _resample_if_needed(wav: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return wav.astype(np.float32, copy=False)
    if resampy is not None:
        return resampy.resample(wav.astype(np.float32), orig_sr, target_sr)
    if librosa is not None:
        return librosa.resample(y=wav.astype(np.float32), orig_sr=orig_sr, target_sr=target_sr)
    warnings.warn(
        "No resampler available; treating audio as target_sr without resampling. Install resampy or librosa.",
        RuntimeWarning,
    )
    return wav.astype(np.float32, copy=False)


# Lightweight HF-style dataset wrapper (optional). Trainer can also pass raw HF datasets directly.
class VibeVoiceDataset:
    def __init__(
        self,
        dataset: Any,
        text_column: str = "text",
        audio_column: str = "audio",
        voice_prompts_column: Optional[str] = "voice_prompts",
        base_dir: Optional[str] = None,
        speech_compress_ratio: int = 3200,
    ) -> None:
        self.dataset = dataset
        self.text_column = text_column
        self.audio_column = audio_column
        self.voice_prompts_column = voice_prompts_column
        self.base_dir = Path(base_dir) if base_dir else None
        self.speech_compress_ratio = speech_compress_ratio

    def _resolve(self, path_str: str) -> str:
        """Resolve a relative path against *base_dir* (the JSONL parent directory)."""
        if self.base_dir is None or os.path.isabs(path_str):
            return path_str
        return str(self.base_dir / path_str)

    @property
    def has_precomputed_latents(self) -> bool:
        """Check if the dataset contains precomputed latent columns."""
        if len(self.dataset) == 0:
            return False
        first = self.dataset[0]
        return "target_acoustic_mean" in first

    def _random_crop_latent(
        self, acoustic_mean: np.ndarray, semantic_feat: Optional[np.ndarray]
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], int]:
        """Random crop a voice prompt in latent space (analogous to waveform cropping)."""
        T = acoustic_mean.shape[0]
        sec_per_frame = self.speech_compress_ratio / 24000.0
        audio_len_seconds = T * sec_per_frame

        min_len_sec = min(5.0, audio_len_seconds / 4.0)
        max_len_sec = min(15.0, audio_len_seconds / 2.0)
        if min_len_sec > max_len_sec:
            min_len_sec = max_len_sec
        max_len_sec = min(max_len_sec, audio_len_seconds)

        if max_len_sec < 0.1:
            return None, None, 0

        crop_len_sec = random.uniform(min_len_sec, max_len_sec)
        crop_len_frames = max(1, int(crop_len_sec / sec_per_frame))
        crop_len_frames = min(crop_len_frames, T)

        max_start = max(0, T - crop_len_frames)
        start = random.randint(0, max_start)

        ac_crop = acoustic_mean[start : start + crop_len_frames]
        se_crop = (
            semantic_feat[start : start + crop_len_frames]
            if semantic_feat is not None
            else None
        )
        return ac_crop, se_crop, crop_len_frames

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.dataset[idx]
        data: Dict[str, Any] = {}
        data["text"] = item[self.text_column]
        audio = item[self.audio_column]
        data["audio"] = self._resolve(audio) if isinstance(audio, str) else audio

        if self.has_precomputed_latents:
            return self._getitem_precomputed(idx, item, data)

        user_provided_prompt = None
        if self.voice_prompts_column and self.voice_prompts_column in item:
            user_provided_prompt = item[self.voice_prompts_column]

        if user_provided_prompt:
            # A prompt was provided in the dataset, so we use it.
            if not isinstance(user_provided_prompt, list):
                user_provided_prompt = [user_provided_prompt]
            data["voice_prompts"] = [
                self._resolve(p) if isinstance(p, str) else p
                for p in user_provided_prompt
            ]
        else:
            # FALLBACK: No prompt provided, so we auto-generate one from the target audio.
            try:
                target_sr = 24000
                wav_array = _load_audio_to_24k(item[self.audio_column], target_sr=target_sr)
                audio_len_seconds = len(wav_array) / target_sr

                min_len_sec = min(5.0, audio_len_seconds / 4.0)
                max_len_sec = min(15.0, audio_len_seconds / 2.0)
                
                if min_len_sec > max_len_sec:
                    min_len_sec = max_len_sec
                max_len_sec = min(max_len_sec, audio_len_seconds)

                if max_len_sec > 0.1:
                    prompt_len_sec = random.uniform(min_len_sec, max_len_sec)
                    prompt_len_samples = int(prompt_len_sec * target_sr)

                    max_start_sample = len(wav_array) - prompt_len_samples
                    start_sample = random.randint(0, max_start_sample)
                    
                    prompt_crop = wav_array[start_sample : start_sample + prompt_len_samples]
                    
                    data["voice_prompts"] = [prompt_crop]
                else:
                    data["voice_prompts"] = None

            except Exception as e:
                warnings.warn(f"Could not create voice prompt for item {idx}: {e}")
                data["voice_prompts"] = None            
        return data

    def _getitem_precomputed(self, idx: int, item: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
        """Read precomputed latents directly from HF Dataset columns."""
        # Target latents — stored as nested lists in the Arrow dataset
        target_ac = np.asarray(item["target_acoustic_mean"], dtype=np.float32)
        target_se = np.asarray(item["target_semantic_feat"], dtype=np.float32)
        data["target_acoustic_mean"] = target_ac
        data["target_semantic_feat"] = target_se
        data["target_latent_len"] = item.get("target_latent_len", int(target_ac.shape[0]))

        # Voice prompts
        prompt_ac_raw = item.get("prompt_acoustic_latents")
        prompt_se_raw = item.get("prompt_semantic_latents")
        if prompt_ac_raw and len(prompt_ac_raw) > 0:
            prompt_acs: List[np.ndarray] = []
            prompt_ses: List[np.ndarray] = []
            prompt_lens: List[int] = []
            for ac_nested, se_nested in zip(prompt_ac_raw, prompt_se_raw):
                ac = np.asarray(ac_nested, dtype=np.float32)
                se = np.asarray(se_nested, dtype=np.float32)
                ac_crop, se_crop, clen = self._random_crop_latent(ac, se)
                if clen > 0 and ac_crop is not None:
                    prompt_acs.append(ac_crop)
                    prompt_ses.append(se_crop)
                    prompt_lens.append(clen)
            if prompt_acs:
                data["prompt_acoustic_means"] = prompt_acs
                data["prompt_semantic_feats"] = prompt_ses
                data["prompt_latent_lens"] = prompt_lens
            else:
                data["prompt_acoustic_means"] = None
        else:
            # Auto-generate voice prompt by cropping from target latents
            ac_crop, se_crop, clen = self._random_crop_latent(target_ac, target_se)
            if clen > 0 and ac_crop is not None:
                data["prompt_acoustic_means"] = [ac_crop]
                data["prompt_semantic_feats"] = [se_crop]
                data["prompt_latent_lens"] = [clen]
            else:
                data["prompt_acoustic_means"] = None

        # voice_prompts stays None; the collator handles prompt tokens via dummy waveforms
        data["voice_prompts"] = None
        return data


def _load_audio_to_24k(audio: Union[str, np.ndarray, torch.Tensor, Dict[str, Any]], *, target_sr: int = 24000) -> np.ndarray:
    if isinstance(audio, np.ndarray):
        return audio.astype(np.float32)
    if isinstance(audio, torch.Tensor):
        return audio.detach().cpu().float().numpy()
    if isinstance(audio, str):
        if librosa is None:
            raise RuntimeError("librosa is required to load audio file paths. Please pip install librosa.")
        wav, sr = librosa.load(audio, sr=None, mono=True)
        wav = _resample_if_needed(wav, int(sr), target_sr)
        return wav
    if isinstance(audio, dict) and "array" in audio and "sampling_rate" in audio:
        arr = np.asarray(audio["array"], dtype=np.float32)
        sr = int(audio["sampling_rate"])
        arr = _resample_if_needed(arr, sr, target_sr)
        return arr
    raise ValueError(f"Unsupported audio type: {type(audio)}")


@dataclass
class VibeVoiceCollator:
    processor: Any  # VibeVoiceProcessor
    max_length: Optional[int] = None
    speech_compress_ratio: int = 3200
    semantic_vae_dim: int = 128
    acoustic_vae_dim: int = 64
    compute_semantics: bool = False
    debug_checks: bool = False

    text_field: str = "text"
    audio_field: str = "audio"
    voice_prompts_field: str = "voice_prompts"
    voice_prompt_drop_rate: float = 0.0

    def __call__(self, features: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        if features and "target_acoustic_mean" in features[0]:
            return self._collate_precomputed(features)
        batch_size = len(features)

        sample_input_ids: List[List[int]] = []
        sample_attention_masks: List[List[int]] = []
        sample_acoustic_input_masks: List[List[bool]] = []
        sample_acoustic_loss_masks: List[List[bool]] = []

        all_speech_waveforms: List[np.ndarray] = []
        all_speech_latent_lengths: List[int] = []
        per_segment_is_target: List[bool] = []

        for ex in features:
            text: str = ex.get(self.text_field, "")
            voice_prompts: Optional[List[Union[str, np.ndarray, torch.Tensor]]] = ex.get(self.voice_prompts_field)
            target_audio: Union[str, np.ndarray, torch.Tensor, Dict[str, Any]] = ex.get(self.audio_field)

            # Clamp drop rate for safety
            _drop_rate = self.voice_prompt_drop_rate
            if _drop_rate < 0.0:
                _drop_rate = 0.0
            elif _drop_rate > 1.0:
                _drop_rate = 1.0

            proc = self.processor(
                text=[text],
                voice_samples=[voice_prompts] if voice_prompts is not None and random.random() >= _drop_rate else None,
                padding=False,
                truncation=False,
                max_length=self.max_length,
                return_tensors="pt",
            )

            ids = proc["input_ids"][0].tolist()
            attn = proc.get("attention_mask", torch.ones_like(proc["input_ids"]))[0].tolist()
            speech_input_mask = proc.get("speech_input_mask")
            if speech_input_mask is None:
                speech_input_mask = torch.zeros_like(proc["input_ids"], dtype=torch.bool)
            speech_input_mask_list = speech_input_mask[0].tolist()

            wav_target = _load_audio_to_24k(target_audio, target_sr=24000)
            # Prefer exact frame count from acoustic tokenizer if available; fallback to compress ratio
            target_latent_len = None
            try:
                acoustic_tok = getattr(self.processor, "acoustic_tokenizer", None)
                if acoustic_tok is not None and hasattr(acoustic_tok, "encode"):
                    enc_out = acoustic_tok.encode(wav_target)
                    # Normalize various possible return formats to get time dimension
                    T = None
                    try:
                        # Direct array-like with shape (T, D) or (T,)
                        if hasattr(enc_out, "shape") and len(getattr(enc_out, "shape", [])) >= 1:
                            T = int(enc_out.shape[0])
                        else:
                            # Nested lists/tuples or ModelOutput-like
                            cand = enc_out
                            # Drill down a couple of levels safely
                            for _ in range(2):
                                if isinstance(cand, (list, tuple)) and len(cand) > 0:
                                    cand = cand[0]
                            if hasattr(cand, "shape") and len(getattr(cand, "shape", [])) >= 1:
                                T = int(cand.shape[0])
                    except Exception:
                        T = None
                    if T is not None and T > 0:
                        target_latent_len = T
            except Exception:
                target_latent_len = None
            if target_latent_len is None:
                target_latent_len = max(1, int(math.ceil(len(wav_target) / float(self.speech_compress_ratio))))

            speech_diff_id = self.processor.tokenizer.speech_diffusion_id
            target_placeholders = [speech_diff_id] * target_latent_len

            ids_extended = ids + target_placeholders
            attn_extended = attn + [1] * target_latent_len

            acoustic_input_mask = speech_input_mask_list + [True] * target_latent_len
            acoustic_loss_mask = ([False] * len(speech_input_mask_list)) + [True] * target_latent_len

            speech_end_id = self.processor.tokenizer.speech_end_id
            ids_extended.append(speech_end_id)
            attn_extended.append(1)
            acoustic_input_mask.append(False)
            acoustic_loss_mask.append(False)

            if self.max_length is not None and len(ids_extended) > self.max_length:
                cut = len(ids_extended) - int(self.max_length)
                leading_non_acoustic = 0
                for v in acoustic_input_mask:
                    if v:
                        break
                    leading_non_acoustic += 1
                if cut > leading_non_acoustic:
                    raise ValueError(
                        f"--max_length={self.max_length} would truncate into acoustic tokens. "
                        f"Needed cut={cut}, but only {leading_non_acoustic} leading non-acoustic tokens available. "
                        "Increase max_length or shorten text/voice-prompt preamble."
                    )
                ids_extended = ids_extended[cut:]
                attn_extended = attn_extended[cut:]
                acoustic_input_mask = acoustic_input_mask[cut:]
                acoustic_loss_mask = acoustic_loss_mask[cut:]

            sample_input_ids.append(ids_extended)
            sample_attention_masks.append(attn_extended)
            sample_acoustic_input_masks.append(acoustic_input_mask)
            sample_acoustic_loss_masks.append(acoustic_loss_mask)

            voice_speeches = []
            voice_latent_lengths = []
            if proc.get("speech_tensors") is not None:
                voice_np = proc["speech_tensors"].cpu().numpy()
                voice_masks = proc["speech_masks"].cpu().numpy().astype(bool)
                for seg_idx in range(voice_np.shape[0]):
                    voice_speeches.append(voice_np[seg_idx])
                    voice_latent_lengths.append(int(voice_masks[seg_idx].sum()))

            all_speech_waveforms.extend(voice_speeches)
            all_speech_latent_lengths.extend(voice_latent_lengths)
            per_segment_is_target.extend([False] * len(voice_speeches))

            all_speech_waveforms.append(wav_target)
            all_speech_latent_lengths.append(target_latent_len)
            per_segment_is_target.append(True)

        max_seq_len = max(len(x) for x in sample_input_ids)
        padded_input_ids = []
        padded_attention_masks = []
        padded_acoustic_input_masks = []
        padded_acoustic_loss_masks = []
        tok = self.processor.tokenizer
        pad_token_id = getattr(tok, "pad_token_id", None)
        if pad_token_id is None or pad_token_id < 0:
            pad_token_id = getattr(tok, "eos_token_id", None)
            if pad_token_id is None or pad_token_id < 0:
                raise ValueError(
                    "Tokenizer has no pad_token_id or eos_token_id; please set one or pass a valid pad id."
                )
        for ids, attn, ain_mask, aloss_mask in zip(
            sample_input_ids, sample_attention_masks, sample_acoustic_input_masks, sample_acoustic_loss_masks
        ):
            pad_len = max_seq_len - len(ids)
            padded_input_ids.append(ids + [pad_token_id] * pad_len)
            padded_attention_masks.append(attn + [0] * pad_len)
            padded_acoustic_input_masks.append(ain_mask + [False] * pad_len)
            padded_acoustic_loss_masks.append(aloss_mask + [False] * pad_len)

        input_ids_tensor = torch.tensor(padded_input_ids, dtype=torch.long)
        attention_mask_tensor = torch.tensor(padded_attention_masks, dtype=torch.long)
        acoustic_input_mask_tensor = torch.tensor(padded_acoustic_input_masks, dtype=torch.bool)
        acoustic_loss_mask_tensor = torch.tensor(padded_acoustic_loss_masks, dtype=torch.bool)

        if all_speech_waveforms:
            max_wave_len = max(w.shape[0] for w in all_speech_waveforms)
            padded_speeches = np.zeros((len(all_speech_waveforms), max_wave_len), dtype=np.float32)
            for i, w in enumerate(all_speech_waveforms):
                L = w.shape[0]
                padded_speeches[i, :L] = w

            max_latent_len = max(all_speech_latent_lengths) if all_speech_latent_lengths else 1
            speech_masks_np = np.zeros((len(all_speech_waveforms), max_latent_len), dtype=np.bool_)
            for i, L_lat in enumerate(all_speech_latent_lengths):
                speech_masks_np[i, :L_lat] = True

            speech_tensors_tensor = torch.tensor(padded_speeches, dtype=torch.float32)
            speech_masks_tensor = torch.tensor(speech_masks_np, dtype=torch.bool)

            speeches_loss_input_np = np.zeros_like(speech_masks_np, dtype=np.bool_)
            for i, is_target in enumerate(per_segment_is_target):
                if is_target:
                    speeches_loss_input_np[i] = speech_masks_np[i]
            speeches_loss_input_tensor = torch.tensor(speeches_loss_input_np, dtype=torch.bool)

            # Semantic features
            if self.compute_semantics and hasattr(self.processor, "semantic_tokenizer") and self.processor.semantic_tokenizer is not None:
                sem_feats: List[np.ndarray] = []
                for w in all_speech_waveforms:
                    try:
                        # Expect [T, D]  where T ≈ ceil(len(w)/compress_ratio)
                        sem = self.processor.semantic_tokenizer.encode(w)
                        sem = np.asarray(sem, dtype=np.float32)
                    except Exception:
                        sem = np.zeros((0, self.semantic_vae_dim), dtype=np.float32)
                    if sem.ndim != 2:
                        raise RuntimeError(f"Semantic tokenizer returned unexpected shape {sem.shape}. Expect [T, D].")
                    L = sem.shape[0]
                    D = sem.shape[1]
                    if D != self.semantic_vae_dim:
                        if D < self.semantic_vae_dim:
                            pad_d = np.zeros((L, self.semantic_vae_dim - D), dtype=np.float32)
                            sem = np.concatenate([sem, pad_d], axis=1)
                        else:
                            sem = sem[:, : self.semantic_vae_dim]
                    if L < max_latent_len:
                        pad = np.zeros((max_latent_len - L, self.semantic_vae_dim), dtype=np.float32)
                        sem = np.concatenate([sem, pad], axis=0)
                    elif L > max_latent_len:
                        sem = sem[:max_latent_len]
                    sem_feats.append(sem.astype(np.float32))
                speech_semantic_tensors = torch.tensor(np.stack(sem_feats, axis=0), dtype=torch.float32)
            else:
                # Semantic tokenizer unavailable while semantics are required for training.
                # Raise to avoid silently degrading alignment with zeroed features.
                raise RuntimeError(
                    "Semantic features are required but could not be computed. "
                    "Ensure processor.semantic_tokenizer is available or precompute and provide features."
                )
        else:
            speech_tensors_tensor = None
            speech_masks_tensor = None
            speeches_loss_input_tensor = None
            speech_semantic_tensors = None  # No segments in batch

        if self.debug_checks:
            assert (input_ids_tensor >= 0).all(), "input_ids contains negative indices"
            if speech_tensors_tensor is not None:
                assert speech_tensors_tensor.dim() == 2, "Expected speech_tensors 2D [segments, samples]"

        return {
            "input_ids": input_ids_tensor,
            "labels": input_ids_tensor.clone(),
            "attention_mask": attention_mask_tensor,
            "speech_tensors": speech_tensors_tensor,
            "speech_masks": speech_masks_tensor,
            "speech_semantic_tensors": speech_semantic_tensors,
            "acoustic_input_mask": acoustic_input_mask_tensor,
            "acoustic_loss_mask": acoustic_loss_mask_tensor,
            "speeches_loss_input": speeches_loss_input_tensor,
        }

    def _collate_precomputed(self, features: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        """Collate a batch using precomputed acoustic/semantic latents."""

        sample_input_ids: List[List[int]] = []
        sample_attention_masks: List[List[int]] = []
        sample_acoustic_input_masks: List[List[bool]] = []
        sample_acoustic_loss_masks: List[List[bool]] = []

        all_acoustic_means: List[np.ndarray] = []
        all_semantic_feats: List[np.ndarray] = []
        all_latent_lengths: List[int] = []
        per_segment_is_target: List[bool] = []

        for ex in features:
            text: str = ex.get(self.text_field, "")
            target_ac_mean: np.ndarray = ex["target_acoustic_mean"]
            target_se_feat: np.ndarray = ex["target_semantic_feat"]
            target_latent_len: int = ex["target_latent_len"]

            prompt_ac_means = ex.get("prompt_acoustic_means")
            prompt_se_feats = ex.get("prompt_semantic_feats")
            prompt_latent_lens = ex.get("prompt_latent_lens")

            # Voice prompt dropout
            _drop_rate = max(0.0, min(1.0, self.voice_prompt_drop_rate))
            drop_prompt = random.random() < _drop_rate

            # Build voice_samples for the processor using dummy waveforms so that
            # it creates the correct number of placeholder tokens.
            if prompt_ac_means is not None and not drop_prompt:
                dummy_prompts = [
                    np.full(int(clen * self.speech_compress_ratio), 1e-4, dtype=np.float32)
                    for clen in prompt_latent_lens
                ]
                proc = self.processor(
                    text=[text],
                    voice_samples=[dummy_prompts],
                    padding=False,
                    truncation=False,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
            else:
                proc = self.processor(
                    text=[text],
                    voice_samples=None,
                    padding=False,
                    truncation=False,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                prompt_ac_means = None  # dropped

            ids = proc["input_ids"][0].tolist()
            attn = proc.get("attention_mask", torch.ones_like(proc["input_ids"]))[0].tolist()
            speech_input_mask = proc.get("speech_input_mask")
            if speech_input_mask is None:
                speech_input_mask = torch.zeros_like(proc["input_ids"], dtype=torch.bool)
            speech_input_mask_list = speech_input_mask[0].tolist()

            # Target placeholders
            speech_diff_id = self.processor.tokenizer.speech_diffusion_id
            target_placeholders = [speech_diff_id] * target_latent_len

            ids_extended = ids + target_placeholders
            attn_extended = attn + [1] * target_latent_len

            acoustic_input_mask = speech_input_mask_list + [True] * target_latent_len
            acoustic_loss_mask = ([False] * len(speech_input_mask_list)) + [True] * target_latent_len

            speech_end_id = self.processor.tokenizer.speech_end_id
            ids_extended.append(speech_end_id)
            attn_extended.append(1)
            acoustic_input_mask.append(False)
            acoustic_loss_mask.append(False)

            if self.max_length is not None and len(ids_extended) > self.max_length:
                cut = len(ids_extended) - int(self.max_length)
                leading_non_acoustic = 0
                for v in acoustic_input_mask:
                    if v:
                        break
                    leading_non_acoustic += 1
                if cut > leading_non_acoustic:
                    raise ValueError(
                        f"--max_length={self.max_length} would truncate into acoustic tokens. "
                        f"Needed cut={cut}, but only {leading_non_acoustic} leading non-acoustic tokens available. "
                        "Increase max_length or shorten text/voice-prompt preamble."
                    )
                ids_extended = ids_extended[cut:]
                attn_extended = attn_extended[cut:]
                acoustic_input_mask = acoustic_input_mask[cut:]
                acoustic_loss_mask = acoustic_loss_mask[cut:]

            sample_input_ids.append(ids_extended)
            sample_attention_masks.append(attn_extended)
            sample_acoustic_input_masks.append(acoustic_input_mask)
            sample_acoustic_loss_masks.append(acoustic_loss_mask)

            # Collect voice prompt latents
            if prompt_ac_means is not None:
                for pac, pse, plen in zip(prompt_ac_means, prompt_se_feats, prompt_latent_lens):
                    all_acoustic_means.append(pac)
                    all_semantic_feats.append(pse)
                    all_latent_lengths.append(plen)
                    per_segment_is_target.append(False)

            # Collect target latents
            all_acoustic_means.append(target_ac_mean)
            all_semantic_feats.append(target_se_feat)
            all_latent_lengths.append(target_latent_len)
            per_segment_is_target.append(True)

        # Pad token sequences
        max_seq_len = max(len(x) for x in sample_input_ids)
        tok = self.processor.tokenizer
        pad_token_id = getattr(tok, "pad_token_id", None)
        if pad_token_id is None or pad_token_id < 0:
            pad_token_id = getattr(tok, "eos_token_id", None)
            if pad_token_id is None or pad_token_id < 0:
                raise ValueError(
                    "Tokenizer has no pad_token_id or eos_token_id; please set one or pass a valid pad id."
                )

        padded_input_ids = []
        padded_attention_masks = []
        padded_acoustic_input_masks = []
        padded_acoustic_loss_masks = []
        for ids, attn, ain_mask, aloss_mask in zip(
            sample_input_ids, sample_attention_masks, sample_acoustic_input_masks, sample_acoustic_loss_masks
        ):
            pad_len = max_seq_len - len(ids)
            padded_input_ids.append(ids + [pad_token_id] * pad_len)
            padded_attention_masks.append(attn + [0] * pad_len)
            padded_acoustic_input_masks.append(ain_mask + [False] * pad_len)
            padded_acoustic_loss_masks.append(aloss_mask + [False] * pad_len)

        input_ids_tensor = torch.tensor(padded_input_ids, dtype=torch.long)
        attention_mask_tensor = torch.tensor(padded_attention_masks, dtype=torch.long)
        acoustic_input_mask_tensor = torch.tensor(padded_acoustic_input_masks, dtype=torch.bool)
        acoustic_loss_mask_tensor = torch.tensor(padded_acoustic_loss_masks, dtype=torch.bool)

        # Pack latent segments
        if all_acoustic_means:
            vae_dim = all_acoustic_means[0].shape[1]
            max_latent_len = max(all_latent_lengths) if all_latent_lengths else 1

            num_segments = len(all_acoustic_means)
            padded_ac = np.zeros((num_segments, max_latent_len, vae_dim), dtype=np.float32)
            speech_masks_np = np.zeros((num_segments, max_latent_len), dtype=np.bool_)

            for i, (ac, length) in enumerate(zip(all_acoustic_means, all_latent_lengths)):
                L = min(ac.shape[0], max_latent_len)
                padded_ac[i, :L] = ac[:L]
                speech_masks_np[i, :length] = True

            speech_tensors_tensor = torch.tensor(padded_ac, dtype=torch.float32)
            speech_masks_tensor = torch.tensor(speech_masks_np, dtype=torch.bool)

            # Loss input mask
            speeches_loss_input_np = np.zeros_like(speech_masks_np, dtype=np.bool_)
            for i, is_target in enumerate(per_segment_is_target):
                if is_target:
                    speeches_loss_input_np[i] = speech_masks_np[i]
            speeches_loss_input_tensor = torch.tensor(speeches_loss_input_np, dtype=torch.bool)

            # Semantic features
            sem_dim = self.semantic_vae_dim
            if all_semantic_feats and all_semantic_feats[0] is not None:
                sem_dim = all_semantic_feats[0].shape[1]
            padded_sem = np.zeros((num_segments, max_latent_len, sem_dim), dtype=np.float32)
            for i, (se, length) in enumerate(zip(all_semantic_feats, all_latent_lengths)):
                if se is not None:
                    L = min(se.shape[0], max_latent_len)
                    D = se.shape[1]
                    if D != sem_dim:
                        if D < sem_dim:
                            pad_d = np.zeros((L, sem_dim - D), dtype=np.float32)
                            se = np.concatenate([se[:L], pad_d], axis=1)
                        else:
                            se = se[:L, :sem_dim]
                    else:
                        se = se[:L]
                    padded_sem[i, :L] = se
            speech_semantic_tensors = torch.tensor(padded_sem, dtype=torch.float32)
        else:
            speech_tensors_tensor = None
            speech_masks_tensor = None
            speeches_loss_input_tensor = None
            speech_semantic_tensors = None

        if self.debug_checks:
            assert (input_ids_tensor >= 0).all(), "input_ids contains negative indices"
            if speech_tensors_tensor is not None:
                assert speech_tensors_tensor.dim() == 3, "Expected precomputed speech_tensors 3D [segments, T, D]"

        return {
            "input_ids": input_ids_tensor,
            "labels": input_ids_tensor.clone(),
            "attention_mask": attention_mask_tensor,
            "speech_tensors": speech_tensors_tensor,
            "speech_masks": speech_masks_tensor,
            "speech_semantic_tensors": speech_semantic_tensors,
            "acoustic_input_mask": acoustic_input_mask_tensor,
            "acoustic_loss_mask": acoustic_loss_mask_tensor,
            "speeches_loss_input": speeches_loss_input_tensor,
            "precomputed_latents": True,
        }