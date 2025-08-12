#!/usr/bin/env python3
import csv
import math
import subprocess
from pathlib import Path
from typing import List, Tuple
import argparse

# ----------------------------- CONFIG ---------------------------------

TARGET_SR = 44100
TARGET_BITS = 16
TARGET_CH = 2
NORMALIZE_DB = -1
SECONDS_PER_CLIP = 4.0  # DCASE Task 7 clips are 4 s

# Augmentations: (suffix, sox_effects...)
AUGMENTATIONS: List[Tuple[str, List[str]]] = [
    ("reverb", ["reverb", "50"]),          # spacious tail
    ("pitchup", ["pitch", "100"]),         # +100 cents
    ("pitchdown", ["pitch", "-100"]),      # -100 cents
    ("tempo95", ["tempo", "-m", "0.95"]),  # -5% tempo
    ("tempo105", ["tempo", "-m", "1.05"]), # +5% tempo
    ("lowpass4k", ["lowpass", "4000"]),    # darker tone
    ("highpass200", ["highpass", "200"]),  # thinner tone
]

# Caption postfixes for augmented files (optional, nice for clarity)
CAPTION_TAGS = {
    "reverb": "(with room reverb)",
    "pitchup": "(slightly higher pitch)",
    "pitchdown": "(slightly lower pitch)",
    "tempo95": "(slightly slower tempo)",
    "tempo105": "(slightly faster tempo)",
    "lowpass4k": "(darker tone)",
    "highpass200": "(thinner tone)",
}

# Safety cap to avoid exploding data
MAX_VARIANTS_PER_FILE = 8

# ----------------------------------------------------------------------

def run(cmd, logf):
    logf.write(" ".join(cmd) + "\n")
    logf.flush()
    subprocess.run(cmd, check=True)

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def sox_convert(in_path: Path, out_path: Path, logf):
    """Convert to 44.1kHz, stereo, 16-bit and normalize peaks to -1 dBFS."""
    cmd = [
        "sox", str(in_path),
        "-r", str(TARGET_SR),
        "-c", str(TARGET_CH),
        "-b", str(TARGET_BITS),
        str(out_path),
        "gain", "-n", str(NORMALIZE_DB)
    ]
    run(cmd, logf)

def sox_augment(in_path: Path, out_path: Path, effect: List[str], logf):
    """Apply augmentation + normalize to -1 dBFS."""
    cmd = ["sox", str(in_path), str(out_path)] + effect + ["gain", "-n", str(NORMALIZE_DB)]
    run(cmd, logf)

def minutes_from_count(n_files: int, seconds_per_clip: float = SECONDS_PER_CLIP) -> float:
    return (n_files * seconds_per_clip) / 60.0

def main():
    ap = argparse.ArgumentParser(description="Auto-extend DCASE clips to target minutes for Stable Audio Open fine-tuning.")
    ap.add_argument("--raw_dir", type=Path, required=True, help="Folder with original WAVs (e.g., dev_001.wav)")
    ap.add_argument("--caption_csv", type=Path, required=True, help="CSV with columns: file,caption")
    ap.add_argument("--out_root", type=Path, required=True, help="Output root folder")
    ap.add_argument("--target_minutes", type=float, default=120.0, help="Approximate target duration in minutes")
    ap.add_argument("--max_variants_per_file", type=int, default=MAX_VARIANTS_PER_FILE, help="Hard cap of variants per source file")
    args = ap.parse_args()

    raw_dir: Path = args.raw_dir
    caption_csv: Path = args.caption_csv
    out_root: Path = args.out_root
    target_minutes: float = args.target_minutes
    max_variants: int = args.max_variants_per_file

    audio_out = out_root / "audio"
    logs_out = out_root / "logs"
    ensure_dir(audio_out)
    ensure_dir(logs_out)

    with open(logs_out / "sox_commands.txt", "w") as logf:
        # Load captions
        rows = []
        with open(caption_csv, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            assert "file" in reader.fieldnames and "caption" in reader.fieldnames, \
                "caption_csv must have columns: file, caption"
            for r in reader:
                rows.append({"file": r["file"], "caption": r["caption"]})

        # Match raw wavs with CSV entries
        src_to_caption = {r["file"]: r["caption"] for r in rows}
        raw_wavs = {p.name: p for p in raw_dir.glob("*.wav")}
        existing = [raw_wavs[name] for name in src_to_caption.keys() if name in raw_wavs]

        if not existing:
            raise FileNotFoundError(f"No matching .wav files in {raw_dir} for entries in {caption_csv}")

        # Convert originals
        final_rows = []
        for p in existing:
            base_name = p.stem  # dev_001
            out_name = f"{base_name}_44k16.wav"
            out_path = audio_out / out_name
            sox_convert(p, out_path, logf)
            final_rows.append({"file": out_name, "caption": src_to_caption[p.name]})

        # Compute how many minutes we have vs target
        base_minutes = minutes_from_count(len(final_rows))
        need_minutes = max(0.0, target_minutes - base_minutes)

        # Compute per-file variants needed (uniform)
        if need_minutes > 0:
            files_needed = math.ceil((need_minutes * 60.0) / SECONDS_PER_CLIP)
            per_file_variants = math.ceil(files_needed / len(final_rows))
            per_file_variants = min(per_file_variants, max_variants)
        else:
            per_file_variants = 0

        print(f"Target: {target_minutes:.2f} min | Base: {base_minutes:.2f} min")
        print(f"Generating {per_file_variants} variants per file (cap {max_variants})...")

        # Augment
        for row in list(final_rows):  # iterate over the originals
            src_name = row["file"]
            src_caption = row["caption"]
            src_path = audio_out / src_name

            for i in range(per_file_variants):
                suffix, effect = AUGMENTATIONS[i % len(AUGMENTATIONS)]
                out_name = f"{Path(src_name).stem}_{suffix}.wav"
                out_path = audio_out / out_name
                sox_augment(src_path, out_path, effect, logf)

                cap_tag = CAPTION_TAGS.get(suffix, f"(aug: {suffix})")
                aug_caption = f"{src_caption} {cap_tag}"
                final_rows.append({"file": out_name, "caption": aug_caption})

        # Write final metadata
        out_csv = out_root / "metadata.csv"
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["file", "caption"])
            writer.writeheader()
            for r in final_rows:
                writer.writerow(r)

        total_minutes = minutes_from_count(len(final_rows))
        print(f"Total clips: {len(final_rows)} | {total_minutes:.2f} min ({total_minutes/60:.2f} h)")
        print(f"Output audio: {audio_out}")
        print(f"Output CSV:   {out_csv}")
        print(f"SoX log:      {logs_out / 'sox_commands.txt'}")

if __name__ == "__main__":
    main()

