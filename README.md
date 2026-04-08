# TiniRAG
Privacy-first RAG CLI — local LLM + SearXNG, zero cloud, zero Docker.

## Install
```bash
pip install tinirag        # or: pipx install tinirag
```

## Usage
```bash
tinirag "what is Python asyncio?"
tinirag --no-search "explain transformers"   # skip web search
tinirag -m llama3.2:3b "hello"              # override model
tinirag chat                                 # interactive REPL
tinirag setup                                # guided first-run wizard
tinirag status                               # show Ollama + SearXNG health
```

## Features
- **Auto-detects your installed Ollama models** — no need to configure a model name; TiniRAG picks the best available one automatically (prefers instruct/chat-tuned, smallest ≥ 3b parameters)
- Privacy-first: no cloud APIs, everything runs locally
- SearXNG web search (optional) for grounded, real-time answers
- Streaming output, session memory, guardrails

## Enable web search (optional)
SearXNG is not required but improves answer quality with live web results.

```bash
pipx inject tinirag searxng    # if installed via pipx
# or:
pip install searxng            # if installed via pip
```

Then run any query — TiniRAG auto-starts SearXNG on port 18888.

## Requirements
- Python 3.11+
- [Ollama](https://ollama.com) running locally (`ollama serve`)
- At least one model pulled: `ollama pull llama3.2:3b`
