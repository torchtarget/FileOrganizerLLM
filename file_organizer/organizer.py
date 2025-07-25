import os
import json
import subprocess
from collections import defaultdict
import argparse
from concurrent.futures import ThreadPoolExecutor
from docx import Document
import openpyxl
from pptx import Presentation
import PyPDF2
import logging

# ------ CONFIG ------
# Configuration can be supplied via command line or environment variables.
# Environment variable fallbacks: FO_ROOT_DIR, FO_N_SAMPLE_FILES, FO_OLLAMA_MODEL

def parse_args():
    parser = argparse.ArgumentParser(description="Summarize folders using a local LLM")
    parser.add_argument("--root", required=False,
                        default=os.environ.get("FO_ROOT_DIR"),
                        help="Root directory to analyze")
    parser.add_argument("--samples", type=int,
                        default=int(os.environ.get("FO_N_SAMPLE_FILES", 10)),
                        help="Number of sample files per folder")
    parser.add_argument("--model",
                        default=os.environ.get("FO_OLLAMA_MODEL", "llama3"),
                        help="Ollama model name")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing folder_contexts.json")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose output")
    return parser.parse_args()

# --- File Extractors ---

def extract_text_txt(path, n_chars=2000):
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read(n_chars)
    except Exception:
        return ""

def extract_text_docx(path, n_chars=2000):
    try:
        doc = Document(path)
        text = "\n".join([p.text for p in doc.paragraphs])
        return text[:n_chars]
    except Exception:
        return ""

def extract_text_pdf(path, n_chars=2000):
    try:
        reader = PyPDF2.PdfReader(path)
        text = ""
        for page in reader.pages[:5]:
            text += page.extract_text() or ""
        return text[:n_chars]
    except Exception:
        return ""

def extract_text_xlsx(path, n_chars=2000):
    try:
        wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
        text = ""
        for ws in wb.worksheets[:2]:
            for row in ws.iter_rows(min_row=1, max_row=15, values_only=True):
                rowtext = "\t".join([str(cell) if cell is not None else "" for cell in row])
                text += rowtext + "\n"
        return text[:n_chars]
    except Exception:
        return ""

def extract_text_pptx(path, n_chars=2000):
    try:
        prs = Presentation(path)
        text = ""
        for i, slide in enumerate(prs.slides):
            if i > 5: break
            for shape in slide.shapes:
                if hasattr(shape, "text"):
                    text += shape.text + "\n"
        return text[:n_chars]
    except Exception:
        return ""

def extract_text_file(path, n_chars=2000):
    ext = os.path.splitext(path)[1].lower()
    if ext in [".txt", ".md", ".csv"]:
        return extract_text_txt(path, n_chars)
    elif ext == ".docx":
        return extract_text_docx(path, n_chars)
    elif ext == ".pdf":
        return extract_text_pdf(path, n_chars)
    elif ext in [".xlsx", ".xls"]:
        return extract_text_xlsx(path, n_chars)
    elif ext in [".pptx", ".ppt"]:
        return extract_text_pptx(path, n_chars)
    else:
        return ""

def get_sample_files(folder, n):
    files = [f for f in os.listdir(folder)
             if os.path.isfile(os.path.join(folder, f))]
    if not files:
        return []

    # Prefer recently modified files and distribute across extensions
    files.sort(key=lambda f: os.path.getmtime(os.path.join(folder, f)),
               reverse=True)
    by_ext = defaultdict(list)
    for f in files:
        by_ext[os.path.splitext(f)[1]].append(f)

    selected = []
    while len(selected) < n and by_ext:
        for ext in list(by_ext.keys()):
            if by_ext[ext]:
                selected.append(by_ext[ext].pop(0))
                if len(selected) >= n:
                    break
            if not by_ext[ext]:
                del by_ext[ext]
    return selected[:n]

def call_ollama(prompt, model, retries=1):
    for attempt in range(retries + 1):
        try:
            proc = subprocess.run(
                ["ollama", "run", model],
                input=prompt.encode("utf-8"),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
            )
            return proc.stdout.decode("utf-8").strip()
        except subprocess.CalledProcessError as e:
            if attempt >= retries:
                err = e.stderr.decode("utf-8", errors="replace")
                logging.error("[ollama error] %s", err)
                return ""
            logging.warning("Ollama failed, retrying...")

def get_file_summary(filepath, ollama_model):
    content = extract_text_file(filepath)
    if not content:
        return "No extractable text."
    prompt = f"Summarize this file in one sentence for a folder classification system:\n{content}"
    summary = call_ollama(prompt, model=ollama_model)
    return summary.strip()

# -- Main hierarchical logic --

def build_folder_tree(root_dir):
    """Map each folder to its children."""
    tree = defaultdict(list)
    all_folders = set()
    for folder, dirs, files in os.walk(root_dir):
        all_folders.add(folder)
        for d in dirs:
            child = os.path.join(folder, d)
            tree[folder].append(child)
    return tree, all_folders

def get_folders_in_bottom_up_order(tree, root_dir):
    """Return a list of folders, leaves first, root last."""
    visited = set()
    order = []

    def visit(folder):
        if folder in visited:
            return
        for child in tree[folder]:
            visit(child)
        visited.add(folder)
        order.append(folder)

    visit(root_dir)
    return order


def get_display_path(folder: str, root_dir: str) -> str:
    """Return folder path including root folder name for context."""
    folder = os.path.abspath(folder)
    root_dir = os.path.abspath(root_dir)
    root_name = os.path.basename(root_dir)
    if folder == root_dir:
        return root_name
    relative = os.path.relpath(folder, root_dir)
    return os.path.join(root_name, relative)

def main():
    args = parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(message)s")
    if not args.root:
        raise SystemExit("--root must be specified (or FO_ROOT_DIR env variable)")

    # Build folder hierarchy
    tree, _ = build_folder_tree(args.root)
    process_order = get_folders_in_bottom_up_order(tree, args.root)

    out_json = os.path.join(args.root, "folder_contexts.json")
    if args.resume and os.path.exists(out_json):
        with open(out_json, "r", encoding="utf-8") as f:
            folder_contexts = json.load(f)
    else:
        folder_contexts = {}

    for folder in process_order:
        if folder in folder_contexts:
            logging.info("Skipping: %s", folder)
            continue

        logging.info("\nProcessing: %s", folder)
        folder_display = get_display_path(folder, args.root)
        # --- File-based summaries
        sample_files = get_sample_files(folder, args.samples)
        sample_summaries = []

        files_to_process = sample_files[:3]
        with ThreadPoolExecutor() as ex:
            summaries = list(ex.map(lambda f: get_file_summary(os.path.join(folder, f), args.model), files_to_process))
        for fname, summary in zip(files_to_process, summaries):
            sample_summaries.append(f"File: {fname}\nSummary: {summary}")

        # --- Child context summaries
        child_contexts = []
        for child in tree[folder]:
            if child in folder_contexts:
                child_display = get_display_path(child, args.root)
                child_contexts.append(f"Subfolder '{child_display}': {folder_contexts[child]}")

        # --- Build context prompt
        prompt = f"""
You are an expert at understanding folder content and organization. The full
folder path provides important context. Subfolders inherit the meaning of the
entire path. For example, if the path contains  'Education' , treat every
subfolder as being about education . Your task is to
summarize the *main topic* of the folder below, ignoring any files that don't fit
the main theme.

Full folder path from the root: '{folder_display}'
Example file summaries (auto-generated):
{chr(10).join(sample_summaries)}
Subfolder context summaries:
{chr(10).join(child_contexts)}

Please summarize the main purpose or topic of this folder in 2-3 sentences,
taking into account its path-derived context, its own files, and the main themes
of its immediate subfolders (if any). If you see outliers, ignore them. Only
output the summary text.
"""
        logging.info("  Calling Ollama (%s)...", args.model)
        folder_summary = call_ollama(prompt, model=args.model)
        logging.info("  => %s", folder_summary)

        folder_contexts[folder] = folder_summary

        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(folder_contexts, f, ensure_ascii=False, indent=2)
        logging.info("Saved summary for %s", folder)

    logging.info("\nSaved folder contexts to %s", out_json)

if __name__ == "__main__":
    main()
