import asyncio
import json
import re
from typing import Union
from collections import Counter, defaultdict
import random
from ._utils import (
    logger,
    clean_str,
    compute_mdhash_id,
    decode_tokens_by_tiktoken,
    encode_string_by_tiktoken,
    is_float_regex,
    list_of_list_to_csv,
    pack_user_ass_to_openai_messages,
    split_string_by_multi_markers,
    truncate_list_by_token_size
)
from .base import (
    BaseGraphStorage,
    BaseKVStorage,
    BaseVectorStorage,
    SingleCommunitySchema,
    CommunitySchema,
    TextChunkSchema,
    QueryParam,
)
from .prompt import GRAPH_FIELD_SEP, PROMPTS
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import math


def chunking_by_token_size(
    content: str, overlap_token_size=128, max_token_size=1024, tiktoken_model="gpt-4o"
):
    tokens = encode_string_by_tiktoken(content, model_name=tiktoken_model)
    results = []
    for index, start in enumerate(
        range(0, len(tokens), max_token_size - overlap_token_size)
    ):
        chunk_content = decode_tokens_by_tiktoken(
            tokens[start : start + max_token_size], model_name=tiktoken_model
        )
        results.append(
            {
                "tokens": min(max_token_size, len(tokens) - start),
                "content": chunk_content.strip(),
                "chunk_order_index": index,
            }
        )
    return results


async def _handle_entity_relation_summary(
    entity_or_relation_name: str,
    description: str,
    global_config: dict,
) -> str:
    use_llm_func: callable = global_config["cheap_model_func"]
    llm_max_tokens = global_config["cheap_model_max_token_size"]
    tiktoken_model_name = global_config["tiktoken_model_name"]
    summary_max_tokens = global_config["entity_summary_to_max_tokens"]

    tokens = encode_string_by_tiktoken(description, model_name=tiktoken_model_name)
    if len(tokens) < summary_max_tokens:  # No need for summary
        return description
    prompt_template = PROMPTS["summarize_entity_descriptions"]
    use_description = decode_tokens_by_tiktoken(
        tokens[:llm_max_tokens], model_name=tiktoken_model_name
    )
    context_base = dict(
        entity_name=entity_or_relation_name,
        description_list=use_description.split(GRAPH_FIELD_SEP),
    )
    use_prompt = prompt_template.format(**context_base)
    logger.debug(f"Trigger summary: {entity_or_relation_name}")
    summary = await use_llm_func(use_prompt, max_tokens=summary_max_tokens)
    return summary


async def _handle_single_entity_extraction(
    record_attributes: list[str],
    chunk_key: str,
):
    # 首先检查列表是否为空
    if not record_attributes or len(record_attributes) < 4:
        return None
        
    try:
        if record_attributes[0] != '"entity"':
            return None
        # add this record as a node in the G
        entity_name = clean_str(record_attributes[1].upper())
        if not entity_name.strip():
            return None
        entity_type = clean_str(record_attributes[2].upper())
        entity_description = clean_str(record_attributes[3])
        entity_source_id = chunk_key
        return dict(
            entity_name=entity_name,
            entity_type=entity_type,
            description=entity_description,
            source_id=entity_source_id,
        )
    except Exception as e:
        logger.error(f"Error in entity extraction: {str(e)}")
        return None

async def _handle_single_relationship_extraction(
    record_attributes: list[str],
    chunk_key: str,
):
    # 首先检查列表是否为空或长度不足
    if not record_attributes or len(record_attributes) < 5:
        return None
        
    try:
        if record_attributes[0] != '"relationship"':
            return None
    # add this record as edge
        source = clean_str(record_attributes[1].upper())
        target = clean_str(record_attributes[2].upper())
        edge_description = clean_str(record_attributes[3])
        edge_source_id = chunk_key
        weight = (
            float(record_attributes[-1]) if is_float_regex(record_attributes[-1]) else 1.0
        )
        return dict(
            src_id=source,
            tgt_id=target,
            weight=weight,
            description=edge_description,
            source_id=edge_source_id,
        )
    except Exception as e:
        logger.error(f"Error in relationship extraction: {str(e)}")
        return None

async def _merge_nodes_then_upsert(
    entity_name: str,
    nodes_data: list[dict],
    knwoledge_graph_inst: BaseGraphStorage,
    global_config: dict,
):
    already_entitiy_types = []
    already_source_ids = []
    already_description = []

    already_node = await knwoledge_graph_inst.get_node(entity_name)
    if already_node is not None:
        already_entitiy_types.append(already_node["entity_type"])
        already_source_ids.extend(
            split_string_by_multi_markers(already_node["source_id"], [GRAPH_FIELD_SEP])
        )
        already_description.append(already_node["description"])

    entity_type = sorted(
        Counter(
            [dp["entity_type"] for dp in nodes_data] + already_entitiy_types
        ).items(),
        key=lambda x: x[1],
        reverse=True,
    )[0][0]
    description = GRAPH_FIELD_SEP.join(
        sorted(set([dp["description"] for dp in nodes_data] + already_description))
    )
    source_id = GRAPH_FIELD_SEP.join(
        set([dp["source_id"] for dp in nodes_data] + already_source_ids)
    )
    description = await _handle_entity_relation_summary(
        entity_name, description, global_config
    )
    node_data = dict(
        entity_type=entity_type,
        description=description,
        source_id=source_id,
    )
    await knwoledge_graph_inst.upsert_node(
        entity_name,
        node_data=node_data,
    )
    node_data["entity_name"] = entity_name
    return node_data


async def _merge_edges_then_upsert(
    src_id: str,
    tgt_id: str,
    edges_data: list[dict],
    knwoledge_graph_inst: BaseGraphStorage,
    global_config: dict,
):
    already_weights = []
    already_source_ids = []
    already_description = []
    already_order = []
    if await knwoledge_graph_inst.has_edge(src_id, tgt_id):
        already_edge = await knwoledge_graph_inst.get_edge(src_id, tgt_id)
        already_weights.append(already_edge["weight"])
        already_source_ids.extend(
            split_string_by_multi_markers(already_edge["source_id"], [GRAPH_FIELD_SEP])
        )
        already_description.append(already_edge["description"])
        already_order.append(already_edge.get("order", 1))

    # [numberchiffre]: `Relationship.order` is only returned from DSPy's predictions
    order = min([dp.get("order", 1) for dp in edges_data] + already_order)
    weight = sum([dp["weight"] for dp in edges_data] + already_weights)
    description = GRAPH_FIELD_SEP.join(
        sorted(set([dp["description"] for dp in edges_data] + already_description))
    )
    source_id = GRAPH_FIELD_SEP.join(
        set([dp["source_id"] for dp in edges_data] + already_source_ids)
    )
    for need_insert_id in [src_id, tgt_id]:
        if not (await knwoledge_graph_inst.has_node(need_insert_id)):
            await knwoledge_graph_inst.upsert_node(
                need_insert_id,
                node_data={
                    "source_id": source_id,
                    "description": description,
                    "entity_type": '"UNKNOWN"',
                },
            )
    description = await _handle_entity_relation_summary(
        (src_id, tgt_id), description, global_config
    )
    await knwoledge_graph_inst.upsert_edge(
        src_id,
        tgt_id,
        edge_data=dict(
            weight=weight,
            description=description,
            source_id=source_id,
            order=order
        ),
    )

async def compute_score(glean_result,content,use_llm_func):
    prompt = """Please evaluate whether the description and type in {{glean_result}} accurately match the information in {{content}}. The instructions are as follows:
        1.	Examine the description and type in {{glean_result}} and determine if they align with the information in {{content}}.
        2.	Provide a score between 0 and 100 based on how well they match:
        •	0 indicates no match at all.
        •	100 indicates a perfect match.
        3.	Scoring guidelines:
        •	Give a high score close to 100 if the description and type are fully consistent and accurate.
        •	Assign a moderate score if there is partial alignment or minor discrepancies.
        •	Give a low score if the description and type are entirely mismatched or irrelevant.
        4. Exactly follow the output example format

    Please Note: Focus on specific differences and ensure the score reflects the accuracy of the match between the two.

    Output Example:

    {{score: 80}}
    """
    LLMcontent = f""" Here's the content:

    {content}

    Here's the glean_result:

    {glean_result}"""

    try:
        # 构造 LLM 输入
        LLMinput = prompt + LLMcontent.format(content=content, glean_result=glean_result)
        response = await use_llm_func(LLMinput)
        
        # 确保从 response 中获取文本内容
        if hasattr(response, 'choices') and len(response.choices) > 0:
            # 处理 ChatCompletion 对象
            response_text = response.choices[0].message.content
        else:
            # 如果是字符串，直接使用
            response_text = str(response)
        
        # 使用正则表达式匹配分数
        match = re.match(r"\{score:\s*(\d+)\s*\}", response_text)
        if match:
            score = int(match.group(1))
        else:
            score = 0
    except Exception as e:
        print(f"Error during processing: {e}")
        score = 0

    return score



def entropy(data):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(data)
    dense_matrix = tfidf_matrix.toarray()
    # 对整个矩阵按列求和，计算每个词的总TF-IDF值
    column_sums = dense_matrix.sum(axis=0)
    total_sum = column_sums.sum()
    if total_sum == 0:
        return 0
    # 归一化每个词的概率
    probabilities = column_sums / total_sum
    # 计算总信息熵
    total_entropy = -sum(p * math.log2(p) for p in probabilities if p > 0)

    return total_entropy


def _preprocess_text(text, entity_type):
    """
    根据不同类型数据进行预处理
    :param text: 输入文本
    :param entity_type: 'entity' 或 'relation' 或 None
    :return: 预处理后的文本
    """
    if not isinstance(text, str):
        return ""
        
    # 基础清理
    text = text.lower().strip()
    
    if entity_type == 'entity':
        # 实体预处理
        # 1. 移除特殊字符，但保留实体名称中可能出现的连字符和下划线
        text = re.sub(r'[^\w\s-]', ' ', text)
        # 2. 标准化空白字符
        text = re.sub(r'\s+', ' ', text)
        # 3. 移除常见的无意义词缀（如 "the", "a", "an" 等）
        stop_words = {'the', 'a', 'an', 'in', 'on', 'at', 'of'}
        text = ' '.join(word for word in text.split() if word not in stop_words)
        
    elif entity_type == 'relation':
        # 关系预处理
        # 1. 保留关系描述中的关键动词和介词
        text = re.sub(r'[^\w\s]', ' ', text)
        # 2. 标准化空白字符
        text = re.sub(r'\s+', ' ', text)
        # 3. 移除数字和特殊标记
        text = re.sub(r'\d+', '', text)
        # 4. 保留动词和介词等关系词
        # 这里可以使用 NLTK 或 spaCy 进行词性标注，保留关系相关的词
        
    else:
        # 通用预处理
        # 1. 基本的文本清理
        text = re.sub(r'[^\w\s]', ' ', text)
        # 2. 标准化空白字符
        text = re.sub(r'\s+', ' ', text)
        # 3. 移除数字
        text = re.sub(r'\d+', '', text)
    
    return text.strip()

def compute_entropy_v1(data, entity_type=None):
    """
    计算不同类型数据的熵值
    :param data: 输入数据列表
    :param entity_type: 'entity' 或 'relation' 或 None
    """
    # 预处理数据
    processed_data = [_preprocess_text(text, entity_type) for text in data]
    
    # 根据类型选择不同的特征提取方法
    if entity_type == 'entity':
        vectorizer = TfidfVectorizer(
            analyzer='word',
            ngram_range=(1, 2),  # 考虑单词和词组
            stop_words='english',
            min_df=2,  # 至少出现2次的词才考虑
            max_df=0.95  # 出现在95%以上文档中的词被视为停用词
        )
    elif entity_type == 'relation':
        vectorizer = TfidfVectorizer(
            analyzer='char',
            ngram_range=(2, 3),  # 考虑字符级别的n-gram
            min_df=2,
            max_df=0.95
        )
    else:
        vectorizer = TfidfVectorizer(
            analyzer='word',
            ngram_range=(1, 3),
            min_df=2,
            max_df=0.95
        )
    
    # 计算TF-IDF
    tfidf_matrix = vectorizer.fit_transform(processed_data)
    dense_matrix = tfidf_matrix.toarray()
    
    # 计算熵值
    column_sums = dense_matrix.sum(axis=0)
    total_sum = column_sums.sum()
    
    if total_sum == 0:
        return 0
        
    probabilities = column_sums / total_sum
    probabilities = np.clip(probabilities, 1e-10, 1)
    
    # 计算归一化熵值
    total_entropy = -np.sum(probabilities * np.log2(probabilities))
    normalized_entropy = total_entropy / np.log2(len(probabilities))
    
    return normalized_entropy

# def combined_entropy_score(data, entity_type=None):
#     """
#     综合多个指标的熵值计算
#     """
#     # 基础熵值
#     base_entropy = compute_entropy_v1(data, entity_type)
    
#     # 语义熵值（如果可用预训练模型）
#     semantic_score = semantic_entropy(data, model)
    
#     # 结构熵值（针对图结构）
#     structure_score = compute_structure_entropy(data)
    
#     # 加权组合
#     weights = [0.4, 0.3, 0.3]  # 可调整权重
#     combined_score = (
#         weights[0] * base_entropy +
#         weights[1] * semantic_score +
#         weights[2] * structure_score
#     )



async def extract_entities(
    chunks: dict[str, TextChunkSchema],
    knwoledge_graph_inst: BaseGraphStorage,
    entity_vdb: BaseVectorStorage,
    mode:str,
    global_config: dict,
    n:int
) -> Union[BaseGraphStorage, None]:
    use_llm_func: callable = global_config["best_model_func"]
    entity_extract_max_gleaning = global_config["entity_extract_max_gleaning"]

    ordered_chunks = list(chunks.items())

    entity_extract_prompt = PROMPTS["entity_extraction"]
    entity_extract_prompt_small = PROMPTS["entity_extraction_small"]
    context_base = dict(
        tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
        record_delimiter=PROMPTS["DEFAULT_RECORD_DELIMITER"],
        completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
        entity_types=",".join(PROMPTS["DEFAULT_ENTITY_TYPES"]),
    )
    continue_prompt = PROMPTS["entiti_continue_extraction"]
    if_loop_prompt = PROMPTS["entiti_if_loop_extraction"]

    already_processed = 0
    already_entities = 0
    already_relations = 0
  
    async def _process_single_content(chunk_key_dp: tuple[str, TextChunkSchema]):
        nonlocal already_processed, already_entities, already_relations
        chunk_key = chunk_key_dp[0]
        chunk_dp = chunk_key_dp[1]
        content = chunk_dp["content"]
        final_result=""

        if mode == "origin":      
            hint_prompt = entity_extract_prompt_small.format(**context_base, input_text=content)
            final_result = await use_llm_func(hint_prompt)
            if isinstance(final_result, list):
                final_result = final_result[0]["text"]
            history = pack_user_ass_to_openai_messages(hint_prompt, final_result)
                # 新增：初始化当前迭代的计数器
            current_entities = 0
            current_relations = 0
            entity_extract_max_gleaning = n
            for now_glean_index in range(entity_extract_max_gleaning):
        
                glean_result = await use_llm_func(continue_prompt, history_messages=history)
    
                history += pack_user_ass_to_openai_messages(continue_prompt, glean_result)
            
                final_result += glean_result
            
                if now_glean_index == entity_extract_max_gleaning - 1:
                    break

                if_loop_result: str = await use_llm_func(
                    if_loop_prompt, history_messages=history
                )
        
                if_loop_result = if_loop_result.strip().strip('"').strip("'").lower()
                if if_loop_result != "yes":
                    break
            records = split_string_by_multi_markers(
                final_result,
                [context_base["record_delimiter"], context_base["completion_delimiter"]],
            )
            maybe_nodes = defaultdict(list)
            maybe_edges = defaultdict(list)
            for record in records:
                record = re.search(r"\((.*)\)", record)
                if record is None:
                    continue
                record = record.group(1)
                record_attributes = split_string_by_multi_markers(
                    record, [context_base["tuple_delimiter"]]
                )
                if_entities = await _handle_single_entity_extraction(
                    record_attributes, chunk_key
                )
                logger.info(f"if_entity{if_entities}")
                if if_entities is not None:
                    maybe_nodes[if_entities["entity_name"]].append(if_entities)
                    continue

                if_relation = await _handle_single_relationship_extraction(
                    record_attributes, chunk_key
                )
            
                if if_relation is not None:
                    maybe_edges[(if_relation["src_id"], if_relation["tgt_id"])].append(
                        if_relation
                    )
            already_processed += 1
            already_entities += len(maybe_nodes)
            already_relations += len(maybe_edges)
            now_ticks = PROMPTS["process_tickers"][
                already_processed % len(PROMPTS["process_tickers"])
            ]
            print(
                f"{now_ticks} Processed {already_processed} chunks, {already_entities} entities(duplicated), {already_relations} relations(duplicated)\r",
                end="",
                flush=True,
            )
            return dict(maybe_nodes), dict(maybe_edges)
        else:
            if mode == "beam":
                #My modifed part
                beam_size = 5
                beam_candidates = []
                for beam in range(beam_size):  
                    hint_prompt = entity_extract_prompt.format(**context_base, input_text=content)
                    
                    # 生成当前候选的 final_result
                    final_result = await use_llm_func(hint_prompt)     #熵 判断初始化，mixed - system 方差大继续生成beam，
                    #小模型效果差的原因：初始化beam能力差。初始化的beam结果不好，后续的beam搜索能力差，导致最终结果不好。
                    
                    # 为每个候选创建一个历史记录（如果有需要的话，可以根据您的实际需求修改）
                    history = pack_user_ass_to_openai_messages(hint_prompt, final_result)
                    
                    # 随机生成一个 0-100 之间的初始得分
                    beam_candidates.append({
                        "history": history,
                        "final_result": final_result,  # 每个候选的初始结果
                        "score": random.randint(0, 100)  # 随机分数
                    })
            
                for now_glean_index in range(entity_extract_max_gleaning):
                    new_candidates = []

            # 遍历当前的 beam 候选
                    for candidate in beam_candidates:
                        history = candidate["history"]
                        final_result = candidate["final_result"]
                        score = candidate["score"]
            
                        glean_result = await use_llm_func(continue_prompt, history_messages=history)

                        updated_history = history + pack_user_ass_to_openai_messages(continue_prompt, glean_result)
                        updated_final_result = final_result + glean_result

                        glean_score = await compute_score(glean_result,content,use_llm_func)
                        #glean_score = random.randint(0, 100)

                        new_candidates.append({
                        "history": updated_history,
                        "final_result": updated_final_result,
                        "score": score + glean_score})
                    
                    new_candidates.sort(key=lambda x: x["score"], reverse=True)
                    beam_candidates = new_candidates[:beam_size]

                # 判断是否继续生成
                    continue_generation = False
                    for candidate in beam_candidates:
                        history = candidate["history"]

                        if_loop_result = await use_llm_func(if_loop_prompt, history_messages=history)
                        if_loop_result = if_loop_result.strip().strip('"').strip("'").lower()

                        # 如果任意候选的 if_loop_result 为 "yes"，继续生成
                        if if_loop_result == "yes":
                            continue_generation = True
                            break

                    if not continue_generation:
                            break
                
                best_candidate = max(beam_candidates, key=lambda x: x["score"])
                final_result = best_candidate["final_result"]
                records = split_string_by_multi_markers(
                    final_result,
                    [context_base["record_delimiter"], context_base["completion_delimiter"]],
                )

                maybe_nodes = defaultdict(list)
                maybe_edges = defaultdict(list)
                for record in records:
                    record = re.search(r"\((.*)\)", record)
                    if record is None:
                        continue
                    record = record.group(1)
                    record_attributes = split_string_by_multi_markers(
                        record, [context_base["tuple_delimiter"]]
                    )
                    if_entities = await _handle_single_entity_extraction(
                        record_attributes, chunk_key
                    )

                    if if_entities is not None:
                        maybe_nodes[if_entities["entity_name"]].append(if_entities)
                        continue

                    if_relation = await _handle_single_relationship_extraction(
                        record_attributes, chunk_key
                    )
                
                    if if_relation is not None:
                        maybe_edges[(if_relation["src_id"], if_relation["tgt_id"])].append(
                            if_relation
                        )
                already_processed += 1
                already_entities += len(maybe_nodes)
                already_relations += len(maybe_edges)
                now_ticks = PROMPTS["process_tickers"][
                    already_processed % len(PROMPTS["process_tickers"])
                ]
                print(
                    f"{now_ticks} Processed {already_processed} chunks, {already_entities} entities(duplicated), {already_relations} relations(duplicated)\r",
                    end="",
                    flush=True,
                )
                return dict(maybe_nodes), dict(maybe_edges)
                #My modified part           
            if mode == "entropy":
                #My modifed part
                beam_size = 5
                beam_candidates = []
                entropy_threshold = 0.2
    
                for beam in range(beam_size):  
                    hint_prompt = entity_extract_prompt_small.format(**context_base, input_text=content)
                    # 生成当前候选的 final_result
                    hint_prompt = "This is beam {beam_size}(ignore this sentence)".format(beam_size=beam) + hint_prompt
                    final_result = await use_llm_func(hint_prompt)     #熵 判断初始化，mixed - system 方差大继续生成beam，
                    #小模型效果差的原因：初始化beam能力差。初始化的beam结果不好，后续的beam搜索能力差，导致最终结果不好。
                    # 为每个候选创建一个历史记录（如果有需要的话，可以根据您的实际需求修改）
                    LLM_score = await compute_score(final_result,content,use_llm_func)
                    history = pack_user_ass_to_openai_messages(hint_prompt, final_result)
                    # 随机生成一个 0-100 之间的初始得分
                    beam_candidates.append({
                        "history": history,
                        "final_result": final_result,  # 每个候选的初始结果
                        "score": LLM_score  # 随机分数
                        })
                candidate = max(beam_candidates, key=lambda x: x["score"])
                
                continue_generation = True
                current_candidate = candidate
                now_glean_index = 0
                while continue_generation and now_glean_index < n: #entity_extract_max_gleaning:
                    new_candidates = []
                    now_glean_index += 1
                    # 沿着当前候选项生成 5 个新的 beam
                    for i in range(beam_size):
                        history = current_candidate["history"]
                        final_result = current_candidate["final_result"]
                        score = current_candidate["score"]
                        # 调用 LLM 继续生成
                        # 提取第一个 user 的 content
                        first_user_content = None
                        for item in history:
                            if item['role'] == 'user':
                                first_user_content = item['content']
                                break  # 找到第一个 user 后立即停止
                        
                        continue_prompt_beam = first_user_content +"Here's the first time result for beam {beamsize}:".format(beamsize = i) + final_result +  continue_prompt
                        
                        glean_result = await use_llm_func(
                            continue_prompt_beam,
                            history_messages=history
                        )
                        updated_history = history + pack_user_ass_to_openai_messages(continue_prompt_beam, glean_result)
                        updated_final_result = final_result + glean_result
                        # 计算信息增益和得分
                        final_records = split_string_by_multi_markers(
                            final_result,
                            [context_base["record_delimiter"], context_base["completion_delimiter"]]
                        )
                        progress_records = split_string_by_multi_markers(
                            updated_final_result,
                            [context_base["record_delimiter"], context_base["completion_delimiter"]]
                        )
                        logger.info(f"progress_records{progress_records}")
                        information_gain = entropy(progress_records) - entropy(final_records)
                        # 添加新候选项
                        new_candidates.append({
                            "history": updated_history,
                            "final_result": updated_final_result,
                            "score": information_gain,
                            "entropy": entropy(progress_records)  # 保存信息增益以便检查
                        })

                    
                    # 从 5 个 beam 中选择信息增益最大的一个作为新的候选项
                    new_candidates.sort(key=lambda x: x["score"], reverse=True)
                    current_candidate = new_candidates[0]
 
                    #检查信息增益是否低于阈值
                    if current_candidate["score"] < entropy_threshold:
                        continue_generation = False
                        break

                final_result = current_candidate["final_result"]

                records = split_string_by_multi_markers(
                    final_result,
                    [context_base["record_delimiter"], context_base["completion_delimiter"]],
                )
                maybe_nodes = defaultdict(list)
                maybe_edges = defaultdict(list)
                for record in records:
                    record = re.search(r"\((.*)\)", record)
                    if record is None:
                        continue
                    record = record.group(1)
                    record_attributes = split_string_by_multi_markers(
                        record, [context_base["tuple_delimiter"]]
                    )
                    if_entities = await _handle_single_entity_extraction(
                        record_attributes, chunk_key
                    )

                    if if_entities is not None:
                        maybe_nodes[if_entities["entity_name"]].append(if_entities)
                        continue

                    if_relation = await _handle_single_relationship_extraction(
                        record_attributes, chunk_key
                    )
                
                    if if_relation is not None:
                        maybe_edges[(if_relation["src_id"], if_relation["tgt_id"])].append(
                            if_relation
                        )
                already_processed += 1
                already_entities += len(maybe_nodes)
                already_relations += len(maybe_edges)
                now_ticks = PROMPTS["process_tickers"][
                    already_processed % len(PROMPTS["process_tickers"])
                ]
                print(
                    f"{now_ticks} Processed {already_processed} chunks, {already_entities} entities(duplicated), {already_relations} relations(duplicated)\r",
                    end="",
                    flush=True,
                )
                return dict(maybe_nodes), dict(maybe_edges)


    results = await asyncio.gather(
        *[_process_single_content(c) for c in ordered_chunks]
    )

    maybe_nodes = defaultdict(list)
    maybe_edges = defaultdict(list)
    for m_nodes, m_edges in results:
        for k, v in m_nodes.items():
            maybe_nodes[k].extend(v)
        for k, v in m_edges.items():
            # it's undirected graph
            maybe_edges[tuple(sorted(k))].extend(v)
    all_entities_data = await asyncio.gather(
        *[
            _merge_nodes_then_upsert(k, v, knwoledge_graph_inst, global_config)
            for k, v in maybe_nodes.items()
        ]
    )
    await asyncio.gather(
        *[
            _merge_edges_then_upsert(k[0], k[1], v, knwoledge_graph_inst, global_config)
            for k, v in maybe_edges.items()
        ]
    )

    if not len(all_entities_data):
        logger.warning("Didn't extract any entities, maybe your LLM is not working")
        return None
    if entity_vdb is not None:
        data_for_vdb = {
            compute_mdhash_id(dp["entity_name"], prefix="ent-"): {
                "content": dp["entity_name"] + dp["description"],
                "entity_name": dp["entity_name"],
            }
            for dp in all_entities_data
        }
        await entity_vdb.upsert(data_for_vdb)
    return knwoledge_graph_inst


def _pack_single_community_by_sub_communities(
    community: SingleCommunitySchema,
    max_token_size: int,
    already_reports: dict[str, CommunitySchema],
) -> tuple[str, int]:
    # TODO
    all_sub_communities = [
        already_reports[k] for k in community["sub_communities"] if k in already_reports
    ]
    all_sub_communities = sorted(
        all_sub_communities, key=lambda x: x["occurrence"], reverse=True
    )
    may_trun_all_sub_communities = truncate_list_by_token_size(
        all_sub_communities,
        key=lambda x: x["report_string"],
        max_token_size=max_token_size,
    )
    sub_fields = ["id", "report", "rating", "importance"]
    sub_communities_describe = list_of_list_to_csv(
        [sub_fields]
        + [
            [
                i,
                c["report_string"],
                c["report_json"].get("rating", -1),
                c["occurrence"],
            ]
            for i, c in enumerate(may_trun_all_sub_communities)
        ]
    )
    already_nodes = []
    already_edges = []
    for c in may_trun_all_sub_communities:
        already_nodes.extend(c["nodes"])
        already_edges.extend([tuple(e) for e in c["edges"]])
    return (
        sub_communities_describe,
        len(encode_string_by_tiktoken(sub_communities_describe)),
        set(already_nodes),
        set(already_edges),
    )


async def _pack_single_community_describe(
    knwoledge_graph_inst: BaseGraphStorage,
    community: SingleCommunitySchema,
    max_token_size: int = 12000,
    already_reports: dict[str, CommunitySchema] = {},
    global_config: dict = {},
) -> str:
    nodes_in_order = sorted(community["nodes"])
    edges_in_order = sorted(community["edges"], key=lambda x: x[0] + x[1])

    nodes_data = await asyncio.gather(
        *[knwoledge_graph_inst.get_node(n) for n in nodes_in_order]
    )
    edges_data = await asyncio.gather(
        *[knwoledge_graph_inst.get_edge(src, tgt) for src, tgt in edges_in_order]
    )
    node_fields = ["id", "entity", "type", "description", "degree"]
    edge_fields = ["id", "source", "target", "description", "rank"]
    nodes_list_data = [
        [
            i,
            node_name,
            node_data.get("entity_type", "UNKNOWN"),
            node_data.get("description", "UNKNOWN"),
            await knwoledge_graph_inst.node_degree(node_name),
        ]
        for i, (node_name, node_data) in enumerate(zip(nodes_in_order, nodes_data))
    ]
    nodes_list_data = sorted(nodes_list_data, key=lambda x: x[-1], reverse=True)
    nodes_may_truncate_list_data = truncate_list_by_token_size(
        nodes_list_data, key=lambda x: x[3], max_token_size=max_token_size // 2
    )
    edges_list_data = [
        [
            i,
            edge_name[0],
            edge_name[1],
            edge_data.get("description", "UNKNOWN"),
            await knwoledge_graph_inst.edge_degree(*edge_name),
        ]
        for i, (edge_name, edge_data) in enumerate(zip(edges_in_order, edges_data))
    ]
    edges_list_data = sorted(edges_list_data, key=lambda x: x[-1], reverse=True)
    edges_may_truncate_list_data = truncate_list_by_token_size(
        edges_list_data, key=lambda x: x[3], max_token_size=max_token_size // 2
    )

    truncated = len(nodes_list_data) > len(nodes_may_truncate_list_data) or len(
        edges_list_data
    ) > len(edges_may_truncate_list_data)

    # If context is exceed the limit and have sub-communities:
    report_describe = ""
    need_to_use_sub_communities = (
        truncated and len(community["sub_communities"]) and len(already_reports)
    )
    force_to_use_sub_communities = global_config["addon_params"].get(
        "force_to_use_sub_communities", False
    )
    if need_to_use_sub_communities or force_to_use_sub_communities:
        logger.debug(
            f"Community {community['title']} exceeds the limit or you set force_to_use_sub_communities to True, using its sub-communities"
        )
        report_describe, report_size, contain_nodes, contain_edges = (
            _pack_single_community_by_sub_communities(
                community, max_token_size, already_reports
            )
        )
        report_exclude_nodes_list_data = [
            n for n in nodes_list_data if n[1] not in contain_nodes
        ]
        report_include_nodes_list_data = [
            n for n in nodes_list_data if n[1] in contain_nodes
        ]
        report_exclude_edges_list_data = [
            e for e in edges_list_data if (e[1], e[2]) not in contain_edges
        ]
        report_include_edges_list_data = [
            e for e in edges_list_data if (e[1], e[2]) in contain_edges
        ]
        # if report size is bigger than max_token_size, nodes and edges are []
        nodes_may_truncate_list_data = truncate_list_by_token_size(
            report_exclude_nodes_list_data + report_include_nodes_list_data,
            key=lambda x: x[3],
            max_token_size=(max_token_size - report_size) // 2,
        )
        edges_may_truncate_list_data = truncate_list_by_token_size(
            report_exclude_edges_list_data + report_include_edges_list_data,
            key=lambda x: x[3],
            max_token_size=(max_token_size - report_size) // 2,
        )
    nodes_describe = list_of_list_to_csv([node_fields] + nodes_may_truncate_list_data)
    edges_describe = list_of_list_to_csv([edge_fields] + edges_may_truncate_list_data)
    return f"""-----Reports-----
```csv
{report_describe}
```
-----Entities-----
```csv
{nodes_describe}
```
-----Relationships-----
```csv
{edges_describe}
```"""


def _community_report_json_to_str(parsed_output: dict) -> str:
    """refer official graphrag: index/graph/extractors/community_reports"""
    title = parsed_output.get("title", "Report")
    summary = parsed_output.get("summary", "")
    findings = parsed_output.get("findings", [])

    def finding_summary(finding: dict):
        if isinstance(finding, str):
            return finding
        return finding.get("summary")

    def finding_explanation(finding: dict):
        if isinstance(finding, str):
            return ""
        return finding.get("explanation")

    report_sections = "\n\n".join(
        f"## {finding_summary(f)}\n\n{finding_explanation(f)}" for f in findings
    )
    return f"# {title}\n\n{summary}\n\n{report_sections}"


async def generate_community_report(
    community_report_kv: BaseKVStorage[CommunitySchema],
    knwoledge_graph_inst: BaseGraphStorage,
    global_config: dict,
):
    llm_extra_kwargs = global_config["special_community_report_llm_kwargs"]
    use_llm_func: callable = global_config["best_model_func"]
    use_string_json_convert_func: callable = global_config[
        "convert_response_to_json_func"
    ]

    community_report_prompt = PROMPTS["community_report"]
    communities_schema = await knwoledge_graph_inst.community_schema()
  
    community_keys, community_values = list(communities_schema.keys()), list(
        communities_schema.values()
    )

    already_processed = 0
    
    async def _form_single_community_report(
        community: SingleCommunitySchema, already_reports: dict[str, CommunitySchema]
    ):
        nonlocal already_processed
        describe = await _pack_single_community_describe(
            knwoledge_graph_inst,
            community,
            max_token_size=global_config["best_model_max_token_size"],
            already_reports=already_reports,
            global_config=global_config,
        )
        # prompt = community_report_prompt.format(input_text=describe)

        # token_count = len(encode_string_by_tiktoken(prompt))
        # threshold = global_config["best_model_max_token_size"] // 2
        # if token_count > threshold:    
        #         # Split sections
        #     sections = describe.split("-----")
            
        #     # Check if middle sections need compression
        #     if len(sections) >= 3:
        #         # Compress REPORT section
        #         sections[2] = await _compress_section_with_summarization(sections[1])

        #     # Rejoin sections
        #     compressed_describe = "-----".join(sections)
        #     prompt = community_report_prompt.format(input_text=compressed_describe)
        prompt = community_report_prompt.format(input_text=describe)
        response = await use_llm_func(prompt, **llm_extra_kwargs)
        
        data = use_string_json_convert_func(response)
        logger.info(f"###########res{response}")
        logger.info(f"###########{data}")
        i=0
        while data == None and i<5 :
            response = await use_llm_func(prompt, **llm_extra_kwargs)
            i+=1
            data = use_string_json_convert_func(response)
            logger.info(f"###########res{i}{response}")

            logger.info(f"###########{i}{data}")


        already_processed += 1
        now_ticks = PROMPTS["process_tickers"][
            already_processed % len(PROMPTS["process_tickers"])
        ]
        print(
            f"{now_ticks} Processed {already_processed} communities\r",
            end="",
            flush=True,
        )
        return data

    levels = sorted(set([c["level"] for c in community_values]), reverse=True)
    logger.info(f"Generating by levels: {levels}")
    community_datas = {}
    for level in levels:
        this_level_community_keys, this_level_community_values = zip(
            *[
                (k, v)
                for k, v in zip(community_keys, community_values)
                if v["level"] == level
            ]
        )
        this_level_communities_reports = await asyncio.gather(
            *[
                _form_single_community_report(c, community_datas)
                for c in this_level_community_values
            ]
        )
        community_datas.update(
            {
                k: {
                    "report_string": _community_report_json_to_str(r),
                    "report_json": r,
                    **v,
                }
                for k, r, v in zip(
                    this_level_community_keys,
                    this_level_communities_reports,
                    this_level_community_values,
                )
            }
        )
    print()  # clear the progress bar
    await community_report_kv.upsert(community_datas)


async def _find_most_related_community_from_entities(
    node_datas: list[dict],
    query_param: QueryParam,
    community_reports: BaseKVStorage[CommunitySchema],
):
    related_communities = []
    for node_d in node_datas:
        if "clusters" not in node_d:
            continue
        related_communities.extend(json.loads(node_d["clusters"]))
    related_community_dup_keys = [
        str(dp["cluster"])
        for dp in related_communities
        if dp["level"] <= query_param.level
    ]
    related_community_keys_counts = dict(Counter(related_community_dup_keys))
    _related_community_datas = await asyncio.gather(
        *[community_reports.get_by_id(k) for k in related_community_keys_counts.keys()]
    )
    related_community_datas = {
        k: v
        for k, v in zip(related_community_keys_counts.keys(), _related_community_datas)
        if v is not None
    }
    related_community_keys = sorted(
        related_community_keys_counts.keys(),
        key=lambda k: (
            related_community_keys_counts[k],
            related_community_datas[k]["report_json"].get("rating", -1),
        ),
        reverse=True,
    )
    sorted_community_datas = [
        related_community_datas[k] for k in related_community_keys
    ]

    use_community_reports = truncate_list_by_token_size(
        sorted_community_datas,
        key=lambda x: x["report_string"],
        max_token_size=query_param.local_max_token_for_community_report,
    )
    if query_param.local_community_single_one:
        use_community_reports = use_community_reports[:1]
    return use_community_reports


async def _find_most_related_text_unit_from_entities(
    node_datas: list[dict],
    query_param: QueryParam,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    knowledge_graph_inst: BaseGraphStorage,
):
    text_units = [
        split_string_by_multi_markers(dp["source_id"], [GRAPH_FIELD_SEP])
        for dp in node_datas
    ]
    edges = await asyncio.gather(
        *[knowledge_graph_inst.get_node_edges(dp["entity_name"]) for dp in node_datas]
    )
    all_one_hop_nodes = set()
    for this_edges in edges:
        if not this_edges:
            continue
        all_one_hop_nodes.update([e[1] for e in this_edges])
    all_one_hop_nodes = list(all_one_hop_nodes)
    all_one_hop_nodes_data = await asyncio.gather(
        *[knowledge_graph_inst.get_node(e) for e in all_one_hop_nodes]
    )
    all_one_hop_text_units_lookup = {
        k: set(split_string_by_multi_markers(v["source_id"], [GRAPH_FIELD_SEP]))
        for k, v in zip(all_one_hop_nodes, all_one_hop_nodes_data)
        if v is not None
    }
    all_text_units_lookup = {}
    for index, (this_text_units, this_edges) in enumerate(zip(text_units, edges)):
        for c_id in this_text_units:
            if c_id in all_text_units_lookup:
                continue
            relation_counts = 0
            for e in this_edges:
                if (
                    e[1] in all_one_hop_text_units_lookup
                    and c_id in all_one_hop_text_units_lookup[e[1]]
                ):
                    relation_counts += 1
            all_text_units_lookup[c_id] = {
                "data": await text_chunks_db.get_by_id(c_id),
                "order": index,
                "relation_counts": relation_counts,
            }
    if any([v is None for v in all_text_units_lookup.values()]):
        logger.warning("Text chunks are missing, maybe the storage is damaged")
    all_text_units = [
        {"id": k, **v} for k, v in all_text_units_lookup.items() if v is not None
    ]
    all_text_units = sorted(
        all_text_units, key=lambda x: (x["order"], -x["relation_counts"])
    )
    all_text_units = truncate_list_by_token_size(
        all_text_units,
        key=lambda x: x["data"]["content"],
        max_token_size=query_param.local_max_token_for_text_unit,
    )
    all_text_units: list[TextChunkSchema] = [t["data"] for t in all_text_units]
    return all_text_units


async def _find_most_related_edges_from_entities(
    node_datas: list[dict],
    query_param: QueryParam,
    knowledge_graph_inst: BaseGraphStorage,
):
    all_related_edges = await asyncio.gather(
        *[knowledge_graph_inst.get_node_edges(dp["entity_name"]) for dp in node_datas]
    )
    all_edges = set()
    for this_edges in all_related_edges:
        all_edges.update([tuple(sorted(e)) for e in this_edges])
    all_edges = list(all_edges)
    all_edges_pack = await asyncio.gather(
        *[knowledge_graph_inst.get_edge(e[0], e[1]) for e in all_edges]
    )
    all_edges_degree = await asyncio.gather(
        *[knowledge_graph_inst.edge_degree(e[0], e[1]) for e in all_edges]
    )
    all_edges_data = [
        {"src_tgt": k, "rank": d, **v}
        for k, v, d in zip(all_edges, all_edges_pack, all_edges_degree)
        if v is not None
    ]
    all_edges_data = sorted(
        all_edges_data, key=lambda x: (x["rank"], x["weight"]), reverse=True
    )
    all_edges_data = truncate_list_by_token_size(
        all_edges_data,
        key=lambda x: x["description"],
        max_token_size=query_param.local_max_token_for_local_context,
    )
    return all_edges_data


async def _build_local_query_context(
    query,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    community_reports: BaseKVStorage[CommunitySchema],
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
):
    results = await entities_vdb.query(query, top_k=query_param.top_k)
    if not len(results):
        return None
    node_datas = await asyncio.gather(
        *[knowledge_graph_inst.get_node(r["entity_name"]) for r in results]
    )
    if not all([n is not None for n in node_datas]):
        logger.warning("Some nodes are missing, maybe the storage is damaged")
    node_degrees = await asyncio.gather(
        *[knowledge_graph_inst.node_degree(r["entity_name"]) for r in results]
    )
    node_datas = [
        {**n, "entity_name": k["entity_name"], "rank": d}
        for k, n, d in zip(results, node_datas, node_degrees)
        if n is not None
    ]
    use_communities = await _find_most_related_community_from_entities(
        node_datas, query_param, community_reports
    )
    use_text_units = await _find_most_related_text_unit_from_entities(
        node_datas, query_param, text_chunks_db, knowledge_graph_inst
    )
    use_relations = await _find_most_related_edges_from_entities(
        node_datas, query_param, knowledge_graph_inst
    )
    logger.info(
        f"Using {len(node_datas)} entites, {len(use_communities)} communities, {len(use_relations)} relations, {len(use_text_units)} text units"
    )
    entites_section_list = [["id", "entity", "type", "description", "rank"]]
    for i, n in enumerate(node_datas):
        entites_section_list.append(
            [
                i,
                n["entity_name"],
                n.get("entity_type", "UNKNOWN"),
                n.get("description", "UNKNOWN"),
                n["rank"],
            ]
        )
    entities_context = list_of_list_to_csv(entites_section_list)

    relations_section_list = [
        ["id", "source", "target", "description", "weight", "rank"]
    ]
    for i, e in enumerate(use_relations):
        relations_section_list.append(
            [
                i,
                e["src_tgt"][0],
                e["src_tgt"][1],
                e["description"],
                e["weight"],
                e["rank"],
            ]
        )
    relations_context = list_of_list_to_csv(relations_section_list)

    communities_section_list = [["id", "content"]]
    for i, c in enumerate(use_communities):
        communities_section_list.append([i, c["report_string"]])
    communities_context = list_of_list_to_csv(communities_section_list)

    text_units_section_list = [["id", "content"]]
    for i, t in enumerate(use_text_units):
        text_units_section_list.append([i, t["content"]])
    text_units_context = list_of_list_to_csv(text_units_section_list)
    return f"""
-----Reports-----
```csv
{communities_context}
```
-----Entities-----
```csv
{entities_context}
```
-----Relationships-----
```csv
{relations_context}
```
-----Sources-----
```csv
{text_units_context}
```
"""


async def local_query(
    query,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    community_reports: BaseKVStorage[CommunitySchema],
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
    global_config: dict,
) -> str:
    use_model_func = global_config["best_model_func"]
    context = await _build_local_query_context(
        query,
        knowledge_graph_inst,
        entities_vdb,
        community_reports,
        text_chunks_db,
        query_param,
    )
    if query_param.only_need_context:
        return context
    if context is None:
        return PROMPTS["fail_response"]
    sys_prompt_temp = PROMPTS["local_rag_response"]
    sys_prompt = sys_prompt_temp.format(
        context_data=context, response_type=query_param.response_type
    )
    response = await use_model_func(
        query,
        system_prompt=sys_prompt,
    )
    return response


async def _map_global_communities(
    query: str,
    communities_data: list[CommunitySchema],
    query_param: QueryParam,
    global_config: dict,
):
    use_string_json_convert_func = global_config["convert_response_to_json_func"]
    use_model_func = global_config["best_model_func"]
    community_groups = []
    while len(communities_data):
        this_group = truncate_list_by_token_size(
            communities_data,
            key=lambda x: x["report_string"],
            max_token_size=query_param.global_max_token_for_community_report,
        )
        community_groups.append(this_group)
        communities_data = communities_data[len(this_group) :]
   
    async def _process(community_truncated_datas: list[CommunitySchema]) -> dict:
        communities_section_list = [["id", "content", "rating", "importance"]]
        for i, c in enumerate(community_truncated_datas):
            communities_section_list.append(
                [
                    i,
                    c["report_string"],
                    c["report_json"].get("rating", 0),
                    c["occurrence"],
                ]
            )
        community_context = list_of_list_to_csv(communities_section_list)
        sys_prompt_temp = PROMPTS["global_map_rag_points"]
        sys_prompt = sys_prompt_temp.format(context_data=community_context)
  
        response = await use_model_func(
            query,
            system_prompt=sys_prompt,
            **query_param.global_special_community_map_llm_kwargs,
        )
        data = use_string_json_convert_func(response)
      
        return data.get("points", [])

    logger.info(f"Grouping to {len(community_groups)} groups for global search")
    responses = await asyncio.gather(*[_process(c) for c in community_groups])
    return responses


async def global_query(
    query,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    community_reports: BaseKVStorage[CommunitySchema],
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
    global_config: dict,
) -> str:
    community_schema = await knowledge_graph_inst.community_schema()

    community_schema = {
        k: v for k, v in community_schema.items() if v["level"] <= query_param.level
    }

    if not len(community_schema):
        return PROMPTS["fail_response"]
    use_model_func = global_config["best_model_func"]
    sorted_community_schemas = sorted(
        community_schema.items(),
        key=lambda x: x[1]["occurrence"],
        reverse=True,
    )
    sorted_community_schemas = sorted_community_schemas[
        : query_param.global_max_consider_community
    ]
    community_datas = await community_reports.get_by_ids(
        [k[0] for k in sorted_community_schemas]
    )
    community_datas = [c for c in community_datas if c is not None]
    community_datas = [
        c
        for c in community_datas
        if c["report_json"].get("rating", 0) >= query_param.global_min_community_rating
    ]
    community_datas = sorted(
        community_datas,
        key=lambda x: (x["occurrence"], x["report_json"].get("rating", 0)),
        reverse=True,
    )
    logger.info(f"Revtrieved {len(community_datas)} communities")

    map_communities_points = await _map_global_communities(
        query, community_datas, query_param, global_config
    )
    
    final_support_points = []
    for i, mc in enumerate(map_communities_points):
        for point in mc:
            if "description" not in point:
                continue
            final_support_points.append(
                {
                    "analyst": i,
                    "answer": point["description"],
                    "score": point.get("score", 1),
                }
            )
    final_support_points = [p for p in final_support_points if p["score"] > 0]
    if not len(final_support_points):
        return PROMPTS["fail_response"]
    final_support_points = sorted(
        final_support_points, key=lambda x: x["score"], reverse=True
    )
    final_support_points = truncate_list_by_token_size(
        final_support_points,
        key=lambda x: x["answer"],
        max_token_size=query_param.global_max_token_for_community_report,
    )
    points_context = []
    for dp in final_support_points:
        points_context.append(
            f"""----Analyst {dp['analyst']}----
Importance Score: {dp['score']}
{dp['answer']}
"""
        )
    points_context = "\n".join(points_context)
  
    if query_param.only_need_context:
        return points_context
    sys_prompt_temp = PROMPTS["global_reduce_rag_response"]
    
    response = await use_model_func(
        query,
        sys_prompt_temp.format(
            report_data=points_context, response_type=query_param.response_type
        ),
    )
    return response


async def naive_query(
    query,
    chunks_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
    global_config: dict,
):
    use_model_func = global_config["best_model_func"]
    results = await chunks_vdb.query(query, top_k=query_param.top_k)
    if not len(results):
        return PROMPTS["fail_response"]
    chunks_ids = [r["id"] for r in results]
    chunks = await text_chunks_db.get_by_ids(chunks_ids)

    maybe_trun_chunks = truncate_list_by_token_size(
        chunks,
        key=lambda x: x["content"],
        max_token_size=query_param.naive_max_token_for_text_unit,
    )
    logger.info(f"Truncate {len(chunks)} to {len(maybe_trun_chunks)} chunks")
    section = "--New Chunk--\n".join([c["content"] for c in maybe_trun_chunks])
    if query_param.only_need_context:
        return section
    sys_prompt_temp = PROMPTS["naive_rag_response"]
    sys_prompt = sys_prompt_temp.format(
        content_data=section, response_type=query_param.response_type
    )
    response = await use_model_func(
        query,
        system_prompt=sys_prompt,
    )
    return response

async def llm_query(
    query, 
    query_param: QueryParam,
    global_config: dict, ):
    use_model_func = global_config["best_model_func"]
    response = await use_model_func(
        query,
        system_prompt=None,
)

    return response
    
