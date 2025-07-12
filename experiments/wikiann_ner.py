import argparse
import json

from datasets import load_dataset
from openai import OpenAI
from sklearn.metrics import precision_recall_fscore_support

from nanographrag_tmp._utils import check_and_fix_json

LABELS = [
    "O",
    "B-PER",
    "I-PER",
    "B-ORG",
    "I-ORG",
    "B-LOC",
    "I-LOC",
]


def ner_call(client: OpenAI, model: str, tokens: list[str]) -> list[str]:
    """Call the model to label tokens.

    The model should return JSON: {"labels": ["..."]} with a label for each token.
    """
    label_space = ", ".join(LABELS)
    messages = [
        {
            "role": "system",
            "content": "You are a named entity recognition model."
        },
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


def evaluate(ds, label_list, client: OpenAI, model: str, context_size: int):
    y_true = []
    y_pred = []
    for item in ds:
        tokens = item["tokens"][:context_size] if context_size > 0 else item["tokens"]
        labels = item["ner_tags"][:context_size] if context_size > 0 else item["ner_tags"]
        gold = [label_list[t] for t in labels]
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
        y_true, y_pred, labels=LABELS, average="micro"
    )
    return precision, recall, f1


def main():
    parser = argparse.ArgumentParser(description="Evaluate NER on WikiANN")
    parser.add_argument("--models", required=True, help="Comma separated model names")
    parser.add_argument("--base_url", default=None)
    parser.add_argument("--api_key", default=None)
    parser.add_argument("--lang", default="en")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--context_sizes", default="0", help="Comma separated token counts")
    args = parser.parse_args()

    ds = load_dataset("wikiann", args.lang, split=args.split)
    if args.limit:
        ds = ds.select(range(args.limit))
    label_list = ds.features["ner_tags"].feature.names

    client = OpenAI(api_key=args.api_key, base_url=args.base_url)

    context_sizes = [int(x) for x in args.context_sizes.split(",")]
    for model in args.models.split(","):
        for c in context_sizes:
            p, r, f = evaluate(ds, label_list, client, model, c)
            print(f"Model={model} context={c} precision={p:.3f} recall={r:.3f} f1={f:.3f}")


if __name__ == "__main__":
    main()
