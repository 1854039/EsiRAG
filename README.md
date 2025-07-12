# EsiRAG

A lightweight experimental implementation of GraphRAG with entity and relation extraction. The project provides tools to index documents and query them using a knowledge graph backed approach.

## Installation

```bash
pip install -r requirements.txt
```

API keys for OpenAI compatible models should be configured via environment variables when running scripts.

## Basic Usage

Use `run.py` to insert a corpus and query it:

```bash
python run.py --base_url <API_URL> --api_key <KEY> --model <MODEL_NAME> \
  --corpus <path/to/text> --insert_mode origin --query_mode llm
```

## Evaluation

An example evaluation script is provided under `experiments/evaluate_entity_extraction.py`. It can run entity and relation extraction on an open dataset such as TACRED or DocRED and report precision, recall and F1 scores.

```bash
python experiments/evaluate_entity_extraction.py --dataset tacred --split validation --model <MODEL_NAME>
```

For document level entity annotation with the DocRED dataset using an OpenAI compatible API you can run:

```bash
python experiments/docred_ner.py \
=======
For token level entity annotation with the WikiANN dataset using an OpenAI compatible API you can run:

```bash
python experiments/wikiann_ner.py \
  --models gpt-3.5-turbo,gpt-4o \
  --split validation --limit 100 \
  --base_url <API_URL> --api_key <KEY>
```
