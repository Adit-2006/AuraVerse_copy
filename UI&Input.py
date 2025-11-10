
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
import json, tempfile, os

from ProcessingMediaAndJSON import process_media, process_json

root = tk.Tk()
root.title("Intelligent Storage System")
root.geometry("780x640")

selected_json_files = []

media_frame = tk.Frame(root, pady=10); media_frame.pack(fill="x")
json_frame  = tk.Frame(root, pady=10); json_frame.pack(fill="both", expand=True)
log_frame   = tk.Frame(root, pady=6);  log_frame.pack(fill="x")

tk.Label(media_frame, text="Media File Uploader", font=("Arial", 14)).pack(anchor="w")

def upload_media_files():
    filenames = filedialog.askopenfilenames(
        title="Select Media Files",
        filetypes=(("Media & Image Files", "*.png *.jpg *.jpeg *.bmp *.gif *.webp *.tif *.tiff *.mp4 *.mov *.avi *.mkv *.heic *.avif"), ("All Files", "*.*"))
    )
    if not filenames:
        return
    valid_ext = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp", ".tif", ".tiff", ".mp4", ".mov", ".avi", ".mkv", ".heic", ".avif"}
    media_files = [f for f in filenames if os.path.splitext(f)[1].lower() in valid_ext]
    if not media_files:
        messagebox.showwarning("No valid media files", "No supported image/video files were selected.")
        return
    try:
        process_media(media_files)
        messagebox.showinfo("Done", f"Processed {len(media_files)} media file(s).")
    except Exception as e:
        messagebox.showerror("Error", f"Media processing failed:\n{e}")

tk.Button(media_frame, text="Select Media Files...", command=upload_media_files).pack(anchor="w")

tk.Label(json_frame, text="Structured Data (JSON)", font=("Arial", 14)).pack(anchor="w")
tk.Label(json_frame, text="Paste your JSON data or upload file(s):").pack(anchor="w")

json_input = scrolledtext.ScrolledText(json_frame, height=14, width=90)
json_input.pack(fill="both", expand=True, padx=2, pady=4)

btn_row = tk.Frame(json_frame); btn_row.pack(fill="x")

tk.Label(json_frame, text="Metacomments (will be added to each JSON as 'metacomments')").pack(anchor="w")
comments_input = tk.Entry(json_frame, width=92)
comments_input.pack(anchor="w", pady=5)

def upload_json_file():
    global selected_json_files
    filenames = filedialog.askopenfilenames(
        title="Select JSON File(s)",
        filetypes=(("JSON Files", "*.json *.JSON *.jsonl *.JSONL *.ndjson *.NDJSON"), ("All Files", "*.*"))
    )
    if not filenames:
        return
    valid_ext = {".json", ".jsonl", ".ndjson"}
    selected_json_files = [f for f in filenames if os.path.splitext(f)[1].lower() in valid_ext]
    if not selected_json_files:
        messagebox.showwarning("No JSON files", "No .json/.jsonl/.ndjson files were selected.")
        return
    try:
        json_input.delete("1.0", tk.END)
        for file in selected_json_files:
            json_input.insert(tk.END, f"--- {os.path.basename(file)} ---\n")
            with open(file, "r", encoding="utf-8") as f:
                json_input.insert(tk.END, f.read() + "\n\n")
        messagebox.showinfo("Loaded", f"Loaded {len(selected_json_files)} JSON file(s).")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to read file(s):\n{e}")

def process_selected_json_now():
    if not selected_json_files:
        messagebox.showwarning("Nothing selected", "Use 'Upload .json File(s)' first or paste JSON.")
        return
    try:
        meta_txt = comments_input.get().strip()
        meta = {"metacomments": meta_txt} if meta_txt else None
        process_json(selected_json_files, metadata=meta)
        messagebox.showinfo("Done", f"Processed {len(selected_json_files)} JSON file(s).")
    except Exception as e:
        messagebox.showerror("Error", f"JSON processing failed:\n{e}")

def process_pasted_json_text():
    text = json_input.get("1.0", tk.END).strip()
    meta_txt = comments_input.get().strip()
    meta = {"metacomments": meta_txt} if meta_txt else None
    if not text:
        if selected_json_files:
            try:
                process_json(selected_json_files, metadata=meta)
                messagebox.showinfo("Done", f"Processed {len(selected_json_files)} JSON file(s).")
            except Exception as e:
                messagebox.showerror("Error", f"JSON processing failed:\n{e}")
        else:
            messagebox.showwarning("Empty", "Paste JSON text or use 'Upload .json File(s)' first.")
        return
    try:
        with tempfile.NamedTemporaryFile(prefix="ui_json_", suffix=".json", delete=False, mode="w", encoding="utf-8") as tf:
            tf.write(text)
            tmp_path = tf.name
        process_json([tmp_path], metadata=meta)
        messagebox.showinfo("Done", "Pasted JSON processed successfully.")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to process pasted JSON:\n{e}")

tk.Button(btn_row, text="Upload .json File(s)", command=upload_json_file).pack(side=tk.LEFT, padx=4)
tk.Button(btn_row, text="Process Selected JSON Now", command=process_selected_json_now).pack(side=tk.LEFT, padx=4)
tk.Button(btn_row, text="Process Pasted JSON", command=process_pasted_json_text).pack(side=tk.LEFT, padx=4)

status = tk.StringVar(value="Ready."); tk.Label(log_frame, textvariable=status, anchor="w").pack(fill="x")

root.mainloop()
