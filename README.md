# FileOrganizerLLM

FileOrganizerLLM analyzes a folder tree and summarizes each directory using a large language model. The tool scans files, extracts a text snippet, and uses a selectable provider (Ollama or OpenAI) to produce concise descriptions. The resulting summaries are stored in `folder_contexts.json`.
Each folder also receives an embedding vector derived from its path and sample file contents. These vectors are saved in `folder_vectors.json` for later use with vector search tools.

## Requirements

- Python 3.8 or later
- Either [Ollama](https://github.com/jmorganca/ollama) installed locally or an OpenAI API key

- Python packages specified in `pyproject.toml`

Install the Python dependencies with:

```bash
pip install .
```

## Usage

1. Run the tool and specify the root directory and other options via command line or the `FO_ROOT_DIR`, `FO_N_SAMPLE_FILES`, and `FO_OLLAMA_MODEL` environment variables. The provider can also be set with `FO_PROVIDER` and OpenAI credentials using `OPENAI_API_KEY`. Choose a provider with `--provider` (`ollama` or `openai`). For example:

```bash
python -m file_organizer --root /path/to/root --samples 10 --model llama3 --provider ollama --verbose
```

To use OpenAI:

```bash
python -m file_organizer --root /path/to/root --provider openai --model gpt-3.5-turbo --openai-api-key YOUR_KEY
```

A `folder_contexts.json` file will be created in the root directory containing the generated summaries.

## How it works

The script selects a few representative files from each folder, extracts up to 2000 characters of text from each, and asks the local LLM for a one-sentence summary. It processes folders from the bottom up so that subfolder summaries contribute to the description of their parent directories. The final output provides a short overview of each folder's main topic or purpose.

In addition, the text snippets and folder path are fed into a local embedding model to create a numeric vector for each directory. These vectors capture semantic information about the folder and are written to `folder_vectors.json`.

Folder path names also guide the summaries. The model interprets the entire path
from the root as context that applies to all nested folders. For example, a path
like `0. Education/Insead` means every subfolder relates to education that took
place at Insead.

## License

This project is licensed under the Apache 2.0 license. See the LICENSE file for details.
