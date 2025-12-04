# FileOrganizerLLM
Technical Specification: Semantic Folder Persona Builder
Codename: “Map Maker”
Version: 1.1
________________________________________
1. Executive Summary
Map Maker generates a semantic index for an entire NAS by creating a folder_persona.json file inside every directory.
It uses post-order traversal to derive meaning:
•	Context flows down: A folder’s position in the hierarchy constrains expected meaning.
•	Meaning flows up: The real definition of a folder emerges from its files (LEAF) or its children (BRANCH).
The core mechanism is a post-order recursive algorithm that processes child folders first, then synthesizes the persona of the current folder from either:
•	its file content (LEAF mode), or
•	the summaries of its children (BRANCH mode).
________________________________________
2. Technical Stack
•	Language: Python 3.10+
•	LLM Providers:
o	Fireworks.ai (llama-v3-70b-instruct, mixtral-8x22b)
o	Ollama (local models)
•	File Readers: pypdf, python-docx, pathlib, standard text extraction
•	Schema Enforcement: Pydantic (v2.x)
•	Optional Future Components: Vector embeddings (OpenAI, Fireworks, SentenceTransformers)
________________________________________
3. Traversal Algorithm: Post-Order
Core function:
def build_persona(current_path) -> PersonaJSON:
Traversal Logic
1.	List immediate subdirectories.
2.	Recurse into each child using post-order.
3.	Once recursion returns, classify the current folder:
o	If dominated by text files → LEAF NODE
o	Else → BRANCH NODE
To avoid infinite loops:
•	Track visited realpaths (for symlink cycles).
•	Enforce a safety depth limit if needed.
LEAF / BRANCH Decision v1.1
Compute:
text_file_count = number of identifiable textual docs
subfolder_count = number of child folders
Rule:
IF text_file_count >= MIN_TEXT_FILES (default 5)
AND text_file_count >= 2 * subfolder_count
    → LEAF
ELSE
    → BRANCH
This prevents misclassification of mixed folders.
Edge-case override:
•	If folder is empty → force BRANCH (an empty conceptual category)
•	If folder has both subfolders and many files → BRANCH, but treat loose files as a synthetic child.
________________________________________
4. LEAF NODE LOGIC (File-Driven Definition)
Use Case: Folder primarily contains documents meaningful on their own (e.g. /Business/Finance/Invoices_2023).
Workflow
1.	Path Context Extraction
o	Split path /Business/Finance/Invoices_2023 → hierarchy ["Business", "Finance", "Invoices_2023"].
o	Prepend global constraint (Business vs Private).
2.	Sample Files
o	Up to 10 text-like files.
o	Selection strategy (deterministic, no true randomness needed):
	include 3 oldest
	include 3 newest
	fill the remaining slots with evenly spaced picks from the middle set
o	Read first 2KB of each.
3.	Clean Parsing
o	Skip unreadable PDFs (OCR garbage).
o	Strip DOCX boilerplate.
4.	LLM Prompt
SYSTEM:
You are Map Maker, a semantic file system classifier.
Obey path constraints strictly. 
Reject outliers. Never hallucinate details.

CONTEXT:
Absolute path: {path}
Hierarchy: {["Business", "Finance", ...]}
GLOBAL CONSTRAINT: {root_constraint}

FILES:
{file_snippets}

TASK:
1. Identify the dominant semantic category.
2. Ignore or flag outliers.
3. Produce a concise folder persona.
4. Follow the target JSON schema exactly.
5.	Validation
o	The LLM is expected to emit a JSON object with `persona` and optional `vector_data`.
o	Pydantic validates the JSON; invalid or non-object responses fall back to a minimal persona that preserves the LLM description and derived-from files while recording validation errors in the audit block.
________________________________________
5. BRANCH NODE LOGIC (Child Aggregation)
Use Case: Folder contains conceptual subfolders, not files (e.g. /Business/Finance/).
Workflow
1.	Load all child personas.
2.	Extract:
o	short_label
o	description
o	negative_constraints
3.	If loose files exist, treat as pseudo-child "LooseFiles" with its own micro-summary.
4.	LLM Prompt:
SYSTEM:
You are Map Maker, a semantic aggregator.
You synthesize meaning from child folders only.
Never invent data not present in children or path constraints.

CONTEXT:
Absolute path: {path}
Hierarchy: {["Business", "Finance"]}
GLOBAL CONSTRAINT: {root_constraint}

CHILDREN:
{child_label_1}: {child_description_1}
{child_label_2}: {child_description_2}
...

TASK:
Write a parent-level definition that unifies these children into a precise category header.
Include negative constraints only if children imply them.
Return strict JSON only.
________________________________________
6. JSON Output Schema (folder_persona.json v1.1)
{
  "schema_version": "1.1",

  "meta": {
    "path": "/NAS/Business/Finance",
    "node_type": "BRANCH",
    "depth": 2,
    "language": "en",
    "confidence": 0.82
  },

  "constraints": {
    "path_context": "Business > Finance",
    "root_rule": "Strictly commercial, revenue-related, legal, and operational documents."
  },

  "persona": {
    "short_label": "Finance & Compliance",
    "description": "Central category for tax filings, invoicing workflows, accounting records, and financial-year documentation.",
    "derived_from": ["Tax Returns", "Invoices", "Ledger Folders"],
    "negative_constraints": ["Exclude personal finance", "Exclude HR materials"]
  },

  "vector_data": {
    "hypothetical_user_queries": [
      "Business finance records",
      "Where are tax filings?",
      "Invoices and accounting files"
    ],
    "embedding_model": null,
    "embedding": null
  },

  "audit": {
    "sample_count": 3,
    "outliers_found": 0,
    "errors": []
  }
}
________________________________________
7. Root Constraints (Semantic Seeds)
Injected into every LLM prompt within a subtree:
ROOT_CONSTRAINTS = {
    "Business": (
        "Strictly commercial, financial, legal, strategic, and operational content. "
        "EXCLUDES: personal, family, domestic, medical, hobby, intimate, or unrelated materials."
    ),
    "Private": (
        "Strictly personal, family, health, education, hobbies, and private financial documents. "
        "EXCLUDES: corporate, client, revenue-generating, or organizational materials."
    )
}
Detection rule:
•	First segment of absolute path determines constraint.
•	If unknown segment → fallback to “General: derive meaning only from content.”
________________________________________
8. Operational Enhancements (v1.1)
8.1 Incremental Rebuilds
Each folder gets a structural hash:
hash = hash(filenames + modification_times + child_hashes)
If hash unchanged → reuse existing persona.
If changed → recompute and propagate upward.
8.2 Error Handling
audit.errors collects:
•	unreadable files
•	corrupted PDFs
•	permission-denied folders
•	symlink loops (skipped)
8.3 Symlink Safety
Track:
visited_real_paths = set()
Skip re-processing if seen before.
8.4 Optional Parallelization
•	Child folders within the same parent can be processed in parallel.
•	Must enforce ordering: parent executes only after all children finish.
________________________________________
9. Future Extensions (Optional)
Not required for v1.1, but architecturally anticipated:
1.	Vector embeddings for semantic search
2.	Federation across multiple NAS locations
3.	Full audit dashboards
4.	Interactive query agent (“Semantic Navigator”)
5.	Retention policy inference
________________________________________
v1.1 Summary
This refinement introduces:
•	A robust LEAF/BRANCH decision rule
•	Mixed-folder handling
•	Safer file sampling and parsing
•	Stronger system prompts
•	Enforced root constraints at every level
•	Confidence metrics
•	Hash-based incremental recomputation
•	Cleaner JSON schema with versioning
•	Error tracking
•	Symlink-safe traversal
It’s now a production-friendly and stable semantic-indexing architecture that remains elegant and extensible.
________________________________________

## Running the prototype

### Quick Start

1. Install dependencies and the editable package (Python 3.10+):

   ```bash
   pip install -e .
   ```

2. **Option A: Using a config file (Recommended)**

   Generate a default config file:

   ```bash
   map-maker --generate-config
   ```

   This creates `config.yaml` in your current directory. Edit it with your settings:

   ```yaml
   # Specify the root path to index
   root_path: /path/to/your/NAS

   # LLM Provider Configuration
   llm:
     provider: stub  # Options: stub, fireworks, ollama

     # For Fireworks.ai:
     # api_key_env: FIREWORKS_API_KEY
     # model: accounts/fireworks/models/llama-v3-70b-instruct

     # For Ollama:
     # host: http://localhost:11434
     # model: llama3

   # Processing Options
   processing:
     follow_symlinks: false
     allow_parallel: false
     min_text_files: 5
     sample_limit: 10
     sample_bytes: 2048
   ```

   Then run:

   ```bash
   map-maker --config config.yaml
   # or simply (if config.yaml is in current directory):
   map-maker
   ```

3. **Option B: Using command-line arguments (Legacy)**

   ```bash
   map-maker /path/to/NAS --provider stub
   ```

   Configure a live LLM backend:

   * **Fireworks.ai**: set `FIREWORKS_API_KEY` (and optionally `FIREWORKS_MODEL`), then run with `--provider fireworks`.
   * **Ollama**: set `OLLAMA_HOST` if different from `http://localhost:11434` and optionally `OLLAMA_MODEL`, then run with `--provider ollama`.

### Advanced Options

The tool walks the tree in post-order, classifies each folder, and writes a `folder_persona.json` beside every directory. Re-run the command to reuse personas when the structural hash has not changed.

**Command-line flags** (override config file settings):
* `--provider {stub,fireworks,ollama}` - Choose LLM provider
* `--follow-symlinks` - Allow following symlinks (disabled by default for safety)
* `--parallel` - Enable parallel child processing within a directory
* `--config <path>` - Specify a custom config file location
* `--generate-config` - Generate a default config.yaml template

**Using environment variables for API keys** (recommended for security):
In your config file, use `api_key_env` instead of `api_key`:
```yaml
llm:
  provider: fireworks
  api_key_env: FIREWORKS_API_KEY
```

Then set the environment variable:
```bash
export FIREWORKS_API_KEY=your-key-here  # Linux/Mac
set FIREWORKS_API_KEY=your-key-here     # Windows
```

