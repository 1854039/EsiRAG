import dspy
import numpy as np
from collections import Counter

import xml.etree.ElementTree as ET
import dspy
from difflib import SequenceMatcher
import re
from tqdm import tqdm
import torch
from typing import List, Dict

def parse_graphml(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()

    # Define namespaces
    ns = {'graphml': 'http://graphml.graphdrawing.org/xmlns'}

    # Extract nodes
    entities = []
    for node in root.findall('.//graphml:node', ns):
        entity_id = node.get('id')
        entity_type = node.find('.//graphml:data[@key="d0"]', ns).text.strip('"')
        description = node.find('.//graphml:data[@key="d1"]', ns).text.strip('"')
        entities.append({'entity_name': entity_id, 'entity_type': entity_type, 'description': description})

    # Extract edges (relationships)
    relationships = []
    for edge in root.findall('.//graphml:edge', ns):
        source = edge.get('source')
        target = edge.get('target')
        description = edge.find('.//graphml:data[@key="d5"]', ns).text.strip('"')
        relationships.append({'source': source, 'target': target, 'description': description})

    return entities, relationships

def create_prediction(entities, relationships):
    # Create dspy.Prediction object
    prediction = dspy.Prediction(
        entities=entities,  # 直接传递实体列表
        relationships=relationships  # 直接传递关系列表
    )
    return prediction


class AssessRelationship(dspy.Signature):
    """Assess the similarity of two relationships."""
    gold_relationship = dspy.InputField()
    predicted_relationship = dspy.InputField()
    similarity_score = dspy.OutputField(desc="Similarity score between 0 and 1")


def relationship_similarity_metric(gold: dspy.Example, pred: dspy.Prediction, trace=None) -> float:
    similarity_scores = []
    
    for gold_rel, pred_rel in zip(gold.relationships.context, pred.relationships.context):
        assessment = dspy.Predict(AssessRelationship)(
            gold_relationship=gold_rel,
            predicted_relationship=pred_rel
        )
        
        try:
            score = float(assessment.similarity_score)
            similarity_scores.append(score)
        except ValueError:
            similarity_scores.append(0.0)
    
    return np.mean(similarity_scores) if similarity_scores else 0.0


def entity_recall_metric(gold: dspy.Example, pred: dspy.Prediction, trace=None) -> float:
    true_set = set(item.entity_name for item in gold.entities.context)
    pred_set = set(item.entity_name for item in pred.entities.context)
    true_positives = len(pred_set.intersection(true_set))
    false_negatives = len(true_set - pred_set)
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    return recall

def calculate_entropy(distribution):
    """Calculate the entropy of a distribution."""
    total = sum(distribution.values())
    probabilities = [count / total for count in distribution.values()]
    entropy = -sum(p * np.log2(p) for p in probabilities if p > 0)
    return entropy

def calculate_normalized_entropy(distribution):
    """Calculate the normalized entropy of a distribution."""
    if not distribution:
        return 0.0
        
    total = sum(distribution.values())
    if total == 0:
        return 0.0
        
    # 计算概率
    probabilities = [count / total for count in distribution.values()]
    
    # 确保概率和为1
    if not np.isclose(sum(probabilities), 1.0):
        print(f"Warning: Probabilities sum to {sum(probabilities)}")
        return 0.0
    
    # 计算熵
    entropy = -sum(p * np.log2(p) for p in probabilities if p > 0)
    
    # 计算最大熵
    n = len(distribution)
    max_entropy = np.log2(n) if n > 0 else 0
    
    # 归一化
    if max_entropy > 0:
        normalized_entropy = entropy / max_entropy
    else:
        normalized_entropy = 0.0
        
    return normalized_entropy

def calculate_semantic_similarity(str1, str2):
    """Calculate semantic similarity between two strings using GPU if available."""
    # 将字符串转换为小写并移除特殊字符
    str1 = re.sub(r'[^a-zA-Z0-9\s]', '', str1.lower())
    str2 = re.sub(r'[^a-zA-Z0-9\s]', '', str2.lower())
    
    # 使用SequenceMatcher计算相似度
    return SequenceMatcher(None, str1, str2).ratio()

def group_similar_relations(relationships, similarity_threshold=0.8):
    """Group relationships with similar semantic meanings with progress bar."""
    relation_groups = {}
    
    # 获取所有独特的关系描述
    descriptions = [rel['description'] for rel in relationships]
    processed = set()
    
    # 添加进度条
    pbar = tqdm(enumerate(descriptions), total=len(descriptions), desc="Grouping similar relations")
    for i, desc1 in pbar:
        if desc1 in processed:
            continue
            
        current_group = [desc1]
        processed.add(desc1)
        
        # 使用GPU加速相似度计算（如果可用）
        for j, desc2 in enumerate(descriptions):
            if desc2 not in processed and i != j:
                if calculate_semantic_similarity(desc1, desc2) >= similarity_threshold:
                    current_group.append(desc2)
                    processed.add(desc2)
        
        if len(current_group) > 0:
            relation_groups[desc1] = current_group
            
        pbar.set_postfix({'groups': len(relation_groups)})
    
    return relation_groups

def calculate_granularity_score(relationships):
    """Calculate the ontology granularity score."""
    # 对关系进行分组
    relation_groups = group_similar_relations(relationships)
    
    # 计算平均语义变体数量
    total_variants = sum(len(group) for group in relation_groups.values())
    num_groups = len(relation_groups)
    
    if num_groups == 0:
        return 0.0
    
    # 计算基础粒度分数
    base_score = total_variants / num_groups
    
    # 归一化分数（使用sigmoid函数将分数映射到0-1范围）
    normalized_score = 2 / (1 + np.exp(-base_score)) - 1
    
    return normalized_score

def semantic_granularity_metric(pred: dspy.Prediction) -> float:
    """Calculate the semantic granularity metric for the knowledge graph."""
    return calculate_granularity_score(pred.relationships)

def graph_entropy_metric(pred: dspy.Prediction) -> float:
    """Calculate the optimized entropy metric for the knowledge graph."""
    # 计算实体类型的归一化熵
    entity_types = [entity['entity_type'] for entity in pred.entities]
    entity_distribution = Counter(entity_types)
    entity_entropy = calculate_normalized_entropy(entity_distribution)

    # 计算关系的归一化熵
    relationship_descriptions = [rel['description'] for rel in pred.relationships]
    relationship_distribution = Counter(relationship_descriptions)
    relationship_entropy = calculate_normalized_entropy(relationship_distribution)

    # 计算语义粒度
    semantic_granularity = semantic_granularity_metric(pred)

    # 综合评分 (调整后的权重)
    weights = {
        'entity': 0.33,
        'relationship': 0.33,
        'granularity': 0.34
    }
    
    total_score = (
        weights['entity'] * entity_entropy +
        weights['relationship'] * relationship_entropy +
        weights['granularity'] * semantic_granularity
    )
    
    return total_score

def report_metrics(pred: dspy.Prediction) -> dict:
    """Report all metrics separately for the knowledge graph with progress tracking."""
    print("\nCalculating Knowledge Graph Metrics...")
    
    # 计算实体类型的归一化熵
    print("Computing entity entropy...")
    entity_types = [entity['entity_type'] for entity in pred.entities]
    entity_distribution = Counter(entity_types)
    entity_entropy = calculate_normalized_entropy(entity_distribution)

    # 计算关系的归一化熵
    print("Computing relationship entropy...")
    relationship_descriptions = [rel['description'] for rel in pred.relationships]
    relationship_distribution = Counter(relationship_descriptions)
    relationship_entropy = calculate_normalized_entropy(relationship_distribution)

    # 计算语义粒度
    print("Computing semantic granularity...")
    semantic_granularity = semantic_granularity_metric(pred)

    # 计算总体熵值
    total_entropy = np.mean([entity_entropy, relationship_entropy])

    print("All metrics computed successfully!")

    return {
        "entropy": {
            "total": total_entropy,
            "entity": entity_entropy,
            "relationship": relationship_entropy
        },
        "granularity": semantic_granularity
    }

# Example usage
if __name__ == "__main__":
    print("Loading graph from file...")
    entities, relationships = parse_graphml('/root/autodl-tmp/nano-graphrag/exp3_copy/working_dir_mistralai_mistral-7b-instruct-v0.3_sci_entropy_local/graph_chunk_entity_relation.graphml')

    #entities, relationships = parse_graphml('/root/autodl-tmp/nano-graphrag/exp3_copy/working_dir_gpt-4o-mini_mmc_entropy_local/graph_chunk_entity_relation.graphml')
    prediction = create_prediction(entities, relationships)
    #entities, relationships = parse_graphml('/root/autodl-tmp/nano-graphrag/exp3_copy/working_dir_mistralai_mistral-7b-instruct-v0.3_mh_entropy_local/graph_chunk_entity_relation.graphml')
    
    print("\nStarting metrics calculation...")
    metrics = report_metrics(prediction)
    
    print("\nKnowledge Graph Metrics Report:")
    print("=" * 40)
    print("\nEntropy Metrics:")
    print(f"  Total Entropy:          {metrics['entropy']['total']:.4f}")
    print(f"  Entity Entropy:         {metrics['entropy']['entity']:.4f}")
    print(f"  Relationship Entropy:   {metrics['entropy']['relationship']:.4f}")
    print("\nGranularity Metrics:")
    print(f"  Semantic Granularity:   {metrics['granularity']:.4f}")
    print("=" * 40)
