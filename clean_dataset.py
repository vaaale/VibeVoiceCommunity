from pathlib import Path
import json
import librosa


def filter_duration(x):
    voice_samples = x.get("voice_prompts", [])
    # Check if duration is in voice_samples is longer than 5 seconds
    for vc in voice_samples:
        wav, sr = librosa.load(vc)
        if len(wav) / sr < 3:
            return False
    wav, sr = librosa.load(x["audio"])
    if len(wav) / sr < 5:
        return False
    if len(x["text"].split()) < 10:
        return False
    if "duration" not in x:
        x["duration"] = len(wav) / sr
    return x["duration"] > 5


def main():
    dataset_path = Path("./exp3.jsonl")
    output_path = Path("./exp3_clean.jsonl")

    with open(dataset_path, "r") as inp, open(output_path, "w") as out:
        for line in inp:
            data = json.loads(line)
            if not filter_duration(data):
                continue
            out.write(json.dumps(data, ensure_ascii=False) + "\n")


if __name__ == '__main__':
    main()

