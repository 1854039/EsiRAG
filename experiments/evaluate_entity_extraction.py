import argparse
from collections import Counter
from datasets import load_dataset
from nanographrag_tmp import GraphRAG


def load_examples(dataset_name: str, split: str):
    """Load an open dataset and yield text with entity/relation labels.

    Currently supports the TACRED and DocRED datasets via the HuggingFace
    `datasets` library. The returned samples follow a unified format:
    `{text: str, entities: List[str], relations: List[Tuple[str,str,str]]}`.
    """
    if dataset_name.lower() == "tacred":
        ds = load_dataset("tacred", split=split)
        for example in ds:
            tokens = example["tokens"]
            text = " ".join(tokens)
            subj = " ".join(tokens[example["subj_start"] : example["subj_end"] + 1])
            obj = " ".join(tokens[example["obj_start"] : example["obj_end"] + 1])
            entities = [subj, obj]
            relations = [(subj, obj, example["relation"])]
            yield {"text": text, "entities": entities, "relations": relations}
    elif dataset_name.lower() == "docred":
        ds = load_dataset("docred", split=split)
        for example in ds:
            text = " ".join(example["sents"].pop()) if isinstance(example["sents"], list) else " ".join(example["sents"])
            ents = [m[0] for m in example["vertexSet"]]
            relations = []
            for r in example["labels"]:
                head = ents[r["h"]]
                tail = ents[r["t"]]
                relations.append((head, tail, r["r"]))
            yield {"text": text, "entities": ents, "relations": relations}
    else:
        raise ValueError(f"Unsupported dataset {dataset_name}")


def evaluate(rag: GraphRAG, samples, mode: str, n: int):
    true_entities = Counter()
    pred_entities = Counter()
    true_relations = Counter()
    pred_relations = Counter()

    for sample in samples:
        rag.insert(sample["text"], mode, n)
        graph = rag.chunk_entity_relation_graph._graph
        pred_ents = list(graph.nodes())
        pred_rels = [
            (u, v, graph.edges[u, v].get("description", "")) for u, v in graph.edges()
        ]
        true_entities.update(sample["entities"])
        pred_entities.update(pred_ents)
        true_relations.update(sample["relations"])
        pred_relations.update(pred_rels)

    def _score(true_c, pred_c):
        tp = sum((true_c & pred_c).values())
        fp = sum((pred_c - true_c).values())
        fn = sum((true_c - pred_c).values())
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        return precision, recall, f1

    entity_scores = _score(true_entities, pred_entities)
    relation_scores = _score(true_relations, pred_relations)
    return entity_scores, relation_scores


def main():
    parser = argparse.ArgumentParser(description="Evaluate entity extraction")
    parser.add_argument("--dataset", choices=["tacred", "docred"], required=True)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--mode", default="origin")
    parser.add_argument("--n", type=int, default=1, help="gleaning iterations")
    parser.add_argument("--model", required=False, help="LLM model name")
    args = parser.parse_args()

    rag = GraphRAG()
    samples = list(load_examples(args.dataset, args.split))
    ent_scores, rel_scores = evaluate(rag, samples, args.mode, args.n)
    print("Entity Precision: {:.3f}, Recall: {:.3f}, F1: {:.3f}".format(*ent_scores))
    print(
        "Relation Precision: {:.3f}, Recall: {:.3f}, F1: {:.3f}".format(*rel_scores)
    )


if __name__ == "__main__":
    main()
