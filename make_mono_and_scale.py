import os
import subprocess
import tempfile
import tkinter as tk
from tkinter import filedialog, messagebox

import numpy as np
import soundfile as sf


SUPPORTED_EXTS = {".wav", ".mp3"}


def load_audio(path):
    """
    Returns (audio, samplerate)
    audio shape:
      mono   -> (N,)
      stereo -> (N, 2)
    """
    try:
        data, sr = sf.read(path, always_2d=False)
        return data, sr
    except RuntimeError:
        # MP3 fallback using ffmpeg
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name

        subprocess.run(
            ["ffmpeg", "-y", "-i", path, tmp_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )

        data, sr = sf.read(tmp_path, always_2d=False)
        os.remove(tmp_path)
        return data, sr


def save_audio(path, data, sr):
    ext = os.path.splitext(path)[1].lower()
    sf.write(path, data, sr)


def make_output_path(in_path):
    folder, base = os.path.split(in_path)
    stem, ext = os.path.splitext(base)
    return os.path.join(folder, f"{stem}_mono_scaled{ext}")


def process_file(path):
    data, sr = load_audio(path)

    # Ensure float64 for safe headroom
    data = data.astype(np.float64)

    if data.ndim == 2:
        # Stereo → mono by summation (NO averaging, NO clipping)
        mono = data[:, 0] + data[:, 1]
    else:
        mono = data.copy()

    peak = np.max(np.abs(mono)) if mono.size else 0.0
    if peak > 1.0:
        mono /= peak

    out_path = make_output_path(path)
    save_audio(out_path, mono, sr)
    return out_path


# ---------------- GUI ---------------- #

def run_gui():
    try:
        from tkinterdnd2 import TkinterDnD, DND_FILES
        root = TkinterDnD.Tk()
        dnd = True
    except Exception:
        root = tk.Tk()
        dnd = False

    root.title("Mono Scaler (wav/mp3)")
    root.geometry("520x220")
    root.resizable(False, False)

    status = tk.StringVar(value="Ready")

    label = tk.Label(
        root,
        text="Drag & drop WAV or MP3 here\n(or click 'Choose File')",
        font=("Arial", 14),
        justify="center",
    )
    label.pack(expand=True, fill="both", padx=12, pady=12)

    tk.Label(root, textvariable=status, anchor="w").pack(fill="x", padx=12)

    btns = tk.Frame(root)
    btns.pack(fill="x", padx=12, pady=(6, 12))

    def handle_path(path):
        try:
            status.set("Processing...")
            root.update_idletasks()
            out = process_file(path)
            status.set(f"Saved: {out}")
            messagebox.showinfo("Done", f"Saved:\n{out}")
        except Exception as e:
            status.set("Error")
            messagebox.showerror("Error", str(e))

    def choose_file():
        path = filedialog.askopenfilename(
            filetypes=[("Audio", "*.wav *.mp3")]
        )
        if path:
            handle_path(path)

    if dnd:
        def on_drop(event):
            path = event.data.strip("{}")
            handle_path(path)

        label.drop_target_register(DND_FILES)
        label.dnd_bind("<<Drop>>", on_drop)

    tk.Button(btns, text="Choose File", command=choose_file).pack(side="left")
    tk.Button(btns, text="Quit", command=root.destroy).pack(side="right")

    root.mainloop()


if __name__ == "__main__":
    run_gui()
