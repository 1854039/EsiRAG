import argparse
import json
from itertools import accumulate

from datasets import load_dataset
from openai import OpenAI
from sklearn.metrics import precision_recall_fscore_support

from nanographrag_tmp._utils import check_and_fix_json


LABELS = ["O"]


def prepare_docred(split: str, limit: int = 0):
    ds = load_dataset("thunlp/docred", split=split)

    def convert(item):
        tokens = []
        offsets = []
        acc = 0
        for sent in item["sents"]:
            tokens.extend(sent)
            offsets.append(acc)
            acc += len(sent)
        labels = ["O"] * len(tokens)
        for entity in item["vertexSet"]:
            for mention in entity:
                start, end = mention["pos"]
                start += offsets[mention["sent_id"]]
                end += offsets[mention["sent_id"]]
                ent_type = mention.get("type", "MISC")
                if f"B-{ent_type}" not in LABELS:
                    LABELS.extend([f"B-{ent_type}", f"I-{ent_type}"])
                labels[start] = f"B-{ent_type}"
                for i in range(start + 1, end):
                    labels[i] = f"I-{ent_type}"
        return {"tokens": tokens, "labels": labels}

    ds = ds.map(convert)
    if limit:
        ds = ds.select(range(limit))
    return ds


def ner_call(client: OpenAI, model: str, tokens: list[str]) -> list[str]:
    label_space = ", ".join(LABELS)
    messages = [
        {"role": "system", "content": "You are a named entity recognition model."},
        {
            "role": "user",
            "content": (
                f"Tokens: {tokens}\n"
                f"Provide a label from [{label_space}] for each token in order.\n"
                f"Return JSON as {{\"labels\": [<labels>]}}"
            ),
        },
    ]
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=200,
    )
    response = completion.choices[0].message.content
    fixed = check_and_fix_json(response)
    return json.loads(fixed)["labels"]


def evaluate(ds, client: OpenAI, model: str, context_size: int):
    y_true = []
    y_pred = []
    for item in ds:
        tokens = item["tokens"][:context_size] if context_size > 0 else item["tokens"]
        gold = item["labels"][:context_size] if context_size > 0 else item["labels"]
        try:
            pred = ner_call(client, model, tokens)
        except Exception as e:
            print(f"Error calling model: {e}")
            continue
        if len(pred) != len(gold):
            continue
        y_true.extend(gold)
        y_pred.extend(pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=LABELS[1:], average="micro"
    )
    return precision, recall, f1


def main():
    parser = argparse.ArgumentParser(description="Evaluate NER on DocRED")
    parser.add_argument("--models", required=True, help="Comma separated model names")
    parser.add_argument("--base_url", default=None)
    parser.add_argument("--api_key", default=None)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--context_sizes", default="0", help="Comma separated token counts")
    args = parser.parse_args()

    ds = prepare_docred(args.split, args.limit)
    client = OpenAI(api_key=args.api_key, base_url=args.base_url)
    context_sizes = [int(x) for x in args.context_sizes.split(",")]
    for model in args.models.split(","):
        for c in context_sizes:
            p, r, f = evaluate(ds, client, model, c)
            print(f"Model={model} context={c} precision={p:.3f} recall={r:.3f} f1={f:.3f}")


if __name__ == "__main__":
    main()
