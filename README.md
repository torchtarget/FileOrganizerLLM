# FileOrganizerLLM

FileOrganizerLLM analyzes a folder tree and summarizes each directory using a local large language model. The tool scans files, extracts a text snippet, and calls `ollama` to produce concise descriptions. The resulting summaries are stored in `folder_contexts.json`.

## Requirements

- Python 3.8 or later
- [Ollama](https://github.com/jmorganca/ollama) installed locally and available on the command line
- Python packages listed in `requirements.txt`

Install the Python dependencies with:

```bash
pip install -r requirements.txt
```

## Usage

1. Run the tool and specify the root directory and other options via command line or the `FO_ROOT_DIR`, `FO_N_SAMPLE_FILES`, and `FO_OLLAMA_MODEL` environment variables. For example:

```bash
python -m file_organizer --root /path/to/root --samples 10 --model llama3 --verbose
```

A `folder_contexts.json` file will be created in the root directory containing the generated summaries.

## How it works

The script selects a few representative files from each folder, extracts up to 2000 characters of text from each, and asks the local LLM for a one-sentence summary. It processes folders from the bottom up so that subfolder summaries contribute to the description of their parent directories. The final output provides a short overview of each folder's main topic or purpose.

## License

This project is licensed under the Apache 2.0 license. See the LICENSE file for details.
