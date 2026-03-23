import json
import shutil
from pathlib import Path

from tqdm import tqdm

src = Path("./exp3_clean.jsonl")
dst = Path("/mnt/Datasets/VibeVoice/Podcast/vibevoice_dataset.jsonl")

def main():
    dst_dir = dst.parent
    samples_dir = dst_dir / "samples"
    voice_prompts_dir = dst_dir / "voice_prompts"
    dst_dir.mkdir(parents=True, exist_ok=True)
    samples_dir.mkdir(parents=True, exist_ok=True)
    voice_prompts_dir.mkdir(parents=True, exist_ok=True)

    with src.open("r", encoding="utf-8") as inp, dst.open("w", encoding="utf-8") as out:
        for line in tqdm(inp):
            sample = json.loads(line)
            audio = Path(sample["audio"])
            voice_prompts = sample["voice_prompts"]

            new_audio_path = Path("samples") / audio.name
            sample["audio"] = str(new_audio_path)
            dst_audio_file = dst_dir / new_audio_path
            if not dst_audio_file.exists():
                shutil.copyfile(audio, dst_audio_file)

            new_voice_prompts = []
            for voice_prompt in voice_prompts:
                new_voice_prompt = Path("voice_prompts") / Path(voice_prompt).name
                new_voice_prompts.append(str(Path("voice_prompts") / new_voice_prompt))
                dst_voice_prompt_file = voice_prompts_dir / Path(voice_prompt).name
                if not Path(dst_voice_prompt_file).exists():
                    shutil.copyfile(voice_prompt, dst_voice_prompt_file)

            out.write(json.dumps(sample, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
