#!/usr/bin/env python3
"""
Enhanced make_ds_preset.py with Decent Sampler ADSR UI controls.

Original description:

Usage (run from the folder containing this script plus input_files/output_files):
    python make_ds_preset.py my_sample.wav --root C4
    python make_ds_preset.py my_sample.mp3 --root 60

Workflow:
  - This script expects:
        ./make_ds_preset.py
        ./input_files/
        ./output_files/
  - You pass ONLY the file name (or relative path) of the sample,
    and it assumes it's inside ./input_files.
  - It creates an output folder INSIDE ./output_files, named after the
    input file (without extension), e.g. "my_sample".
  - It then puts:
        - a .wav version of the sample
        - the .dspreset
    in that output folder.

Special handling:
  - input_files is treated as read-only. It is never modified or cleaned.
  - For .mp3 (or any non-.wav) inputs:
      * Copy the file from input_files into the output folder,
      * Convert that copy to .wav using ffmpeg,
      * Delete the copied non-wav file, leaving only the .wav + .dspreset.

Enhancement:
  - Adds four ADSR controls (Attack, Decay, Sustain, Release) to the UI
    and hooks them up to the amplitude envelope.
"""

import argparse
import math
import os
import re
import shutil
import subprocess
import xml.etree.ElementTree as ET

from termcolor import colored


# ---------------------------
# Helper: parse root note
# ---------------------------

NOTE_TO_SEMITONE = {
    "C": 0,
    "C#": 1, "Db": 1,
    "D": 2,
    "D#": 3, "Eb": 3,
    "E": 4, "Fb": 4,
    "F": 5, "E#": 5,
    "F#": 6, "Gb": 6,
    "G": 7,
    "G#": 8, "Ab": 8,
    "A": 9,
    "A#": 10, "Bb": 10,
    "B": 11, "Cb": 11, "B#": 0,
}

NOTE_REGEX = re.compile(r"^([A-Ga-g])([#b]?)(-?\d+)$")


def parse_root_note(root: str) -> int:
    """Parse a root note argument which can be either:
      - an integer MIDI note number, e.g. "60"
      - a note name like "C4", "A#3", "Db2"

    Returns:
        MIDI note number (0-127).

    Raises:
        ValueError if the format is invalid or out of range.
    """
    root = root.strip()

    # First, try plain integer MIDI note
    try:
        midi = int(root)
        if not (0 <= midi <= 127):
            raise ValueError(f"MIDI note {midi} out of range (0-127)")
        return midi
    except ValueError:
        pass  # Not a plain int; fall through to note-name parsing

    # Parse as note name
    m = NOTE_REGEX.match(root)
    if not m:
        raise ValueError(
            f"Invalid root note format: '{root}'. "
            "Use MIDI number (e.g. 60) or note name (e.g. C4, A#3, Db2)."
        )

    letter = m.group(1).upper()
    accidental = m.group(2)
    octave = int(m.group(3))

    note_name = letter + accidental

    if note_name not in NOTE_TO_SEMITONE:
        raise ValueError(f"Unrecognized note name: '{note_name}'")

    semitone = NOTE_TO_SEMITONE[note_name]

    # MIDI note formula: C-1 = 0, C4 = 60, etc.
    midi = (octave + 1) * 12 + semitone

    if not (0 <= midi <= 127):
        raise ValueError(
            f"Resulting MIDI note {midi} from '{root}' out of range (0-127)"
        )

    return midi


# ---------------------------
# Pitch estimation
# ---------------------------

SEMITONE_TO_NOTE = [
    "C", "C#", "D", "D#", "E", "F",
    "F#", "G", "G#", "A", "A#", "B",
]


def midi_to_note_name(midi: int) -> str:
    """Return a note name like 'C4' for a MIDI number."""
    octave = (midi // 12) - 1
    name = SEMITONE_TO_NOTE[midi % 12]
    return f"{name}{octave}"


def hz_to_midi_float(hz: float) -> float:
    """Convert Hz to a (possibly fractional) MIDI note number."""
    if hz <= 0:
        raise ValueError("Frequency must be positive")
    return 12.0 * math.log2(hz / 440.0) + 69.0


def estimate_fundamental_hz(wav_path: str) -> float | None:
    """
    Estimate the fundamental frequency of a WAV file using librosa's
    probabilistic YIN (pyin) algorithm.

    Returns the median voiced-frame fundamental in Hz, or None if no
    voiced frames were detected.
    """
    try:
        import librosa
        import numpy as np
    except ImportError:
        raise ImportError(
            "librosa is required for pitch estimation. "
            "Run: pip install librosa"
        )

    y, sr = librosa.load(wav_path, sr=None, mono=True)

    f0, voiced_flag, _ = librosa.pyin(
        y,
        fmin=float(librosa.note_to_hz("C2")),
        fmax=float(librosa.note_to_hz("C8")),
        sr=sr,
    )

    voiced_f0 = f0[voiced_flag]
    if len(voiced_f0) == 0:
        return None

    return float(np.median(voiced_f0))


def print_pitch_estimate(hz: float, mode: str) -> None:
    """Print a coloured pitch-estimate report to stdout."""
    midi_float = hz_to_midi_float(hz)
    midi_int = round(midi_float)
    midi_int = max(0, min(127, midi_int))
    cents_off = (midi_float - midi_int) * 100.0
    note_name = midi_to_note_name(midi_int)

    print(colored("-" * 50, "cyan"))
    print(colored("  Pitch Estimation Results", "cyan", attrs=["bold"]))
    print(colored("-" * 50, "cyan"))
    print(
        "  Estimated fundamental: "
        + colored(f"{hz:.3f} Hz", "yellow", attrs=["bold"])
    )

    if mode == "12edo":
        print(
            "  Nearest 12-EDO note:   "
            + colored(f"{note_name}  (MIDI {midi_int})", "green", attrs=["bold"])
            + colored(f"  [{cents_off:+.1f} cents]", "white")
        )
        print(colored("-" * 50, "cyan"))
    else:  # pure_hz
        print(
            "  Nearest 12-EDO note:   "
            + colored(f"{note_name}  (MIDI {midi_int})", "white")
            + colored(f"  [{cents_off:+.1f} cents]", "white")
        )
        print(
            "  rootNote (Hz):         "
            + colored(f"{hz:.4f}", "magenta", attrs=["bold"])
            + colored(
                "  ! DecentSampler rootNote expects an integer MIDI value "
                "(0-127); writing Hz here is non-standard and may be ignored "
                "or misinterpreted by the player.",
                "red",
            )
        )
        print(colored("-" * 50, "cyan"))


# ---------------------------
# Core preset generator
# ---------------------------


def create_decent_sampler_preset(
    sample_path: str,
    root_note: int = 60,
    root_note_raw: str | None = None,
) -> str:
    """
    Create a Decent Sampler .dspreset file for a single sample with
    visible ADSR knobs in the UI.

    :param sample_path: Path to the WAV sample file (in the output folder).
    :param root_note: MIDI note number that represents the sample's original
                      pitch.
    :param root_note_raw: If provided, written verbatim as the rootNote
                          attribute (e.g. a Hz string for --estimate-pure-hz).
    :return: Path to the created .dspreset file.
    """
    # Ensure sample exists
    if not os.path.isfile(sample_path):
        raise FileNotFoundError(f"Sample not found: {sample_path}")

    # Normalize paths
    sample_abs = os.path.abspath(sample_path)
    sample_dir = os.path.dirname(sample_abs)
    sample_filename = os.path.basename(sample_abs)

    # Use sample name (without extension) as instrument/preset name
    base_name, _ = os.path.splitext(sample_filename)
    preset_filename = base_name + ".dspreset"
    preset_path = os.path.join(sample_dir, preset_filename)

    # --- ADSR defaults (seconds, sustain 0-1) -----------------------
    attack_default = 0.005
    decay_default = 0.5
    sustain_default = 1.0
    release_default = 0.25
    # ---------------------------------------------------------------

    # Build XML structure
    root = ET.Element("DecentSampler")
    # (optional but nice) - require a reasonably recent version:
    # root.set("minVersion", "1.13.3")

    # UI with a single tab
    ui = ET.SubElement(root, "ui")
    ui.set("width", "812")
    ui.set("height", "375")

    tab = ET.SubElement(ui, "tab")
    tab.set("name", "main")

    # Title label
    label = ET.SubElement(tab, "label")
    label.set("x", "20")
    label.set("y", "20")
    label.set("text", base_name)

    # ------- Attack knob -------------------------------------------
    atk = ET.SubElement(tab, "labeled-knob")
    atk.set("x", "50")
    atk.set("y", "80")
    atk.set("width", "90")
    atk.set("label", "Attack")
    atk.set("type", "float")
    atk.set("minValue", "0.0")
    atk.set("maxValue", "5.0")
    atk.set("value", str(attack_default))

    atk_bind = ET.SubElement(atk, "binding")
    atk_bind.set("type", "amp")
    atk_bind.set("level", "group")
    atk_bind.set("groupIndex", "0")      # first (and only) group
    atk_bind.set("parameter", "ENV_ATTACK")
    atk_bind.set("translation", "linear")

    # ------- Decay knob --------------------------------------------
    dec = ET.SubElement(tab, "labeled-knob")
    dec.set("x", "170")
    dec.set("y", "80")
    dec.set("width", "90")
    dec.set("label", "Decay")
    dec.set("type", "float")
    dec.set("minValue", "0.0")
    dec.set("maxValue", "10.0")
    dec.set("value", str(decay_default))

    dec_bind = ET.SubElement(dec, "binding")
    dec_bind.set("type", "amp")
    dec_bind.set("level", "group")
    dec_bind.set("groupIndex", "0")
    dec_bind.set("parameter", "ENV_DECAY")
    dec_bind.set("translation", "linear")

    # ------- Sustain knob ------------------------------------------
    sus = ET.SubElement(tab, "labeled-knob")
    sus.set("x", "290")
    sus.set("y", "80")
    sus.set("width", "90")
    sus.set("label", "Sustain")
    sus.set("type", "float")
    sus.set("minValue", "0.0")
    sus.set("maxValue", "1.0")
    sus.set("value", str(sustain_default))

    sus_bind = ET.SubElement(sus, "binding")
    sus_bind.set("type", "amp")
    sus_bind.set("level", "group")
    sus_bind.set("groupIndex", "0")
    sus_bind.set("parameter", "ENV_SUSTAIN")
    sus_bind.set("translation", "linear")

    # ------- Release knob ------------------------------------------
    rel = ET.SubElement(tab, "labeled-knob")
    rel.set("x", "410")
    rel.set("y", "80")
    rel.set("width", "90")
    rel.set("label", "Release")
    rel.set("type", "float")
    rel.set("minValue", "0.0")
    rel.set("maxValue", "10.0")
    rel.set("value", str(release_default))

    rel_bind = ET.SubElement(rel, "binding")
    rel_bind.set("type", "amp")
    rel_bind.set("level", "group")
    rel_bind.set("groupIndex", "0")
    rel_bind.set("parameter", "ENV_RELEASE")
    rel_bind.set("translation", "linear")

    # -------- Groups + sample --------------------------------------
    groups = ET.SubElement(root, "groups")
    group = ET.SubElement(groups, "group")

    # Set group-level ADSR defaults so they match the knob values
    group.set("attack", str(attack_default))
    group.set("decay", str(decay_default))
    group.set("sustain", str(sustain_default))
    group.set("release", str(release_default))

    sample_elem = ET.SubElement(group, "sample")
    sample_elem.set("path", sample_filename)  # relative
    sample_elem.set("rootNote", root_note_raw if root_note_raw is not None else str(root_note))
    sample_elem.set("loNote", "0")
    sample_elem.set("hiNote", "127")

    # Pretty-print XML (Python 3.9+)
    tree = ET.ElementTree(root)
    try:
        ET.indent(tree, space="  ", level=0)
    except AttributeError:
        pass

    tree.write(preset_path, encoding="UTF-8", xml_declaration=True)
    return preset_path

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Create a Decent Sampler .dspreset for a single sample, with ADSR "
            "controls in the UI.\n"
            "The script expects 'input_files' and 'output_files' folders next "
            "to it.\n\n"
            "Usage:\n"
            "  python make_ds_preset.py my_sample.wav --root C4\n\n"
            "The --root argument may be a MIDI note (e.g. 60) or a note name "
            "(e.g. C4, A#3, Db2).\n\n"
            "Workflow:\n"
            "  - Reads from ./input_files (never modifies/deletes anything there)\n"
            "  - Writes to ./output_files/<sample_name_without_extension>/\n"
            "  - For non-wav inputs, copies to output, converts to wav via ffmpeg,\n"
            "    then deletes the copied non-wav file to keep things tidy."
        )
    )
    parser.add_argument(
        "sample",
        help=(
            "Name (or relative path) of the audio file inside 'input_files', "
            "e.g. 'my_sample.wav' or 'subdir/snare.mp3'."
        ),
    )
    parser.add_argument(
        "--root",
        default="60",
        help="Root note as MIDI number or note name (default: 60 / C4).",
    )
    parser.add_argument(
        "--estimate-root-note-12-edo",
        action="store_true",
        help=(
            "Estimate the root note from the audio and round to the nearest "
            "12-EDO MIDI note. Overrides --root."
        ),
    )
    parser.add_argument(
        "--estimate-pure-hz",
        action="store_true",
        help=(
            "Estimate the root note from the audio and write the raw Hz value "
            "as rootNote in the preset. "
            "Note: DecentSampler rootNote expects an integer MIDI value; "
            "this is non-standard and may not be supported by the player."
        ),
    )

    args = parser.parse_args()

    if args.estimate_root_note_12_edo and args.estimate_pure_hz:
        print(colored("Error: --estimate-root-note-12-edo and --estimate-pure-hz are mutually exclusive.", "red"))
        return

    try:
        root_midi = parse_root_note(args.root)

        # Determine script directory and the fixed input/output roots
        script_dir = os.path.dirname(os.path.abspath(__file__))
        input_root = os.path.join(script_dir, "input_files")
        output_root = os.path.join(script_dir, "output_files")

        if not os.path.isdir(input_root):
            raise FileNotFoundError(
                f"Expected input folder not found: {input_root}"
            )

        os.makedirs(output_root, exist_ok=True)

        # Resolve input sample path (inside input_files)
        input_sample_abs = os.path.abspath(os.path.join(input_root, args.sample))
        if not os.path.isfile(input_sample_abs):
            raise FileNotFoundError(f"Input sample not found: {input_sample_abs}")

        input_basename = os.path.basename(input_sample_abs)
        name_without_ext, ext = os.path.splitext(input_basename)
        ext = ext.lower()

        # Create an output folder named after the input file (without extension).
        # Sanitize slightly to avoid problematic characters in folder names.
        folder_name_raw = name_without_ext
        folder_name = re.sub(r"[^A-Za-z0-9_#\-]+", "_", folder_name_raw)
        output_dir = os.path.join(output_root, folder_name)
        os.makedirs(output_dir, exist_ok=True)

        # Target WAV path inside the output folder.
        wav_path = os.path.join(output_dir, name_without_ext + ".wav")

        # Helper: convert any non-wav file via ffmpeg, using a temporary
        # copied input in the output folder, then removing that temporary file.
        def convert_to_wav_via_ffmpeg(input_path: str, out_wav_path: str) -> None:
            tmp_input_path = os.path.join(
                output_dir, os.path.basename(input_path)
            )
            # Copy from input_files to output_files (never touch original)
            shutil.copy2(input_path, tmp_input_path)
            print(f"Copied input to output folder for conversion: {tmp_input_path}")

            cmd = [
                "ffmpeg",
                "-y",  # overwrite without asking
                "-i",
                tmp_input_path,
                out_wav_path,
            ]
            print("Running ffmpeg to convert input to WAV:")
            print("  ", " ".join(cmd))
            subprocess.run(cmd, check=True)
            print(f"Converted '{tmp_input_path}' -> '{out_wav_path}' using ffmpeg")

            # Remove the temporary non-wav file to keep the output folder clean
            try:
                os.remove(tmp_input_path)
                print(f"Removed temporary file: {tmp_input_path}")
            except OSError as e:
                print(f"Warning: could not remove temporary file {tmp_input_path}: {e}")

        # If the input isn't already a WAV, convert it with ffmpeg
        # in the output folder, using a copied temp input.
        if ext != ".wav":
            convert_to_wav_via_ffmpeg(input_sample_abs, wav_path)
        else:
            # Already WAV: copy into the output folder so preset and sample
            # live together. Never delete/modify input.
            if os.path.abspath(wav_path) != os.path.abspath(input_sample_abs):
                shutil.copy2(input_sample_abs, wav_path)
                print(f"Copied WAV into output folder: {wav_path}")
            else:
                # This would only happen if someone ran the script directly
                # on a file already inside output_files, but we keep it safe.
                print("Input WAV already in the desired output location.")

        # Optionally estimate root note from the audio.
        root_note_raw = None
        if args.estimate_root_note_12_edo or args.estimate_pure_hz:
            print(colored("Estimating fundamental frequency...", "cyan"))
            hz = estimate_fundamental_hz(wav_path)
            if hz is None:
                print(colored(
                    "Warning: no voiced frames detected; falling back to --root value.",
                    "yellow",
                ))
            else:
                mode = "12edo" if args.estimate_root_note_12_edo else "pure_hz"
                print_pitch_estimate(hz, mode)
                if args.estimate_root_note_12_edo:
                    midi_float = hz_to_midi_float(hz)
                    root_midi = max(0, min(127, round(midi_float)))
                else:  # pure_hz
                    root_note_raw = f"{hz:.4f}"

        # Now create the preset using the WAV that lives in the output folder.
        preset_path = create_decent_sampler_preset(wav_path, root_midi, root_note_raw)

        if args.estimate_root_note_12_edo:
            note_str = midi_to_note_name(root_midi)
            print(colored(
                f"Using estimated root note: {note_str} (MIDI {root_midi})",
                "green", attrs=["bold"],
            ))
        elif args.estimate_pure_hz:
            print(colored(
                f"Using estimated root note (Hz): {root_note_raw}  "
                "[non-standard - see warning above]",
                "magenta", attrs=["bold"],
            ))
        else:
            print(f"Parsed root note '{args.root}' -> MIDI {root_midi}")
        print(f"Created Decent Sampler preset with ADSR UI controls:\n  {preset_path}")
        print("Associated WAV sample:\n  " + wav_path)
        print("\nNow in Decent Sampler:")
        print("  1) Load this .dspreset from the '" + output_dir + "' folder")
        print("  2) Adjust Attack, Decay, Sustain, Release using the on-screen knobs")

    except subprocess.CalledProcessError as exc:
        print(f"Error running ffmpeg: {exc}")
    except Exception as exc:
        print(f"Error: {exc}")


if __name__ == "__main__":
    main()
