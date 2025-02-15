import os
import logging
import numpy as np
import requests
import json
from nano_graphrag_test import GraphRAG, QueryParam
from nano_graphrag_test.base import BaseKVStorage
from nano_graphrag_test._utils import compute_args_hash, wrap_embedding_func_with_attrs
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
import argparse
import torch
from time import time
from openai import OpenAI, APIConnectionError, RateLimitError

import re
import yaml
import math
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple
from dataclasses import dataclass
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
logging.basicConfig(level=logging.WARNING)
logging.getLogger("nano-graphrag").setLevel(logging.INFO)
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    retry_if_result,
)

device = "cpu"  # the device to load the model onto

load_model_time = 0
load_embedding_model_time = 0
text_read_time = 0
indexing_time = 0
#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

# Load embedding model
start = time()
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
EMBED_MODEL = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
load_embedding_model_time = time() - start
print(f"Embedding model loading time: {load_embedding_model_time:.2f} seconds")

@wrap_embedding_func_with_attrs(
    embedding_dim=EMBED_MODEL.config.hidden_size,
    max_token_size=EMBED_MODEL.config.max_position_embeddings
)
async def local_embedding(texts: list[str]) -> np.ndarray:
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to('cpu')
    with torch.no_grad():
        model_output = EMBED_MODEL(**inputs)
    embeddings = mean_pooling(model_output, inputs['attention_mask'])
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    return embeddings

# Custom retry condition to check if the result is empty
def is_empty_result(result):
    return result is None or result == ""

@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=(retry_if_exception_type((RateLimitError, APIConnectionError)))
)
async def ollama_model_if_cache(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    #ollama_client = ollama.AsyncClient()
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    # Get the cached response if having-------------------
    hashing_kv: BaseKVStorage = kwargs.pop("hashing_kv", None)
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})
    if hashing_kv is not None:
        args_hash = compute_args_hash(args.model, messages)
        if_cache_return = await hashing_kv.get_by_id(args_hash)
        if if_cache_return is not None:
            return if_cache_return["return"]
    # -----------------------------------------------------

    client = OpenAI(
        api_key=args.api_key,
        base_url=args.base_url,
    )
    completion = client.chat.completions.create(
        model=args.model,
        messages=[
            {
            "role": "user",
            "content": prompt
            }
        ],
       
        stream=False,

        response_format={"type": "json_object"}

    )

    # # print the time delay and text received
    # result = ''.join([m.content for m in collected_messages])
    # Cache the response if having-------------------
    if hashing_kv is not None:
        await hashing_kv.upsert({args_hash: {"return": result, "model": args.model}})
    # -----------------------------------------------------
    return result


def remove_if_exist(file):
    if os.path.exists(file):
        os.remove(file)

def load_scibench_texts(directory):
    texts = []
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            with open(os.path.join(directory, filename), 'r') as file:
                texts.append(file.read())
    return "\n".join(texts)


def insert():
    global text_read_time, indexing_time
    # # 设置包含所有 txt 文件的目录路径
    FAKE_TEXT = ""
    if args.corpus == "law":
        # 设置包含所有 txt 文件的目录路径
        txt_directory = "./LegalBench-RAG/corpus/privacy_qa"  # 将此路径替换为你的 txt 文件夹路径
        # 存储拼接的文本内容
        FAKE_TEXT = ""

        # 获取所有 txt 文件名，并排序（如果需要）y
        all_files = sorted([f for f in os.listdir(txt_directory) if f.endswith(".txt")])

        start = time()
        for filename in all_files:  
            # 遍历 txt_directory 中的所有文件
            if filename.endswith(".txt"):  # 检查是否是 txt 文件
                file_path = os.path.join(txt_directory, filename)
                with open(file_path, encoding="utf-8-sig") as f:
                    # 将每个文件的内容追加到 FAKE_TEXT 中
                    FAKE_TEXT += f.read() + "\n"  # 在每个文件后面添加换行符
        text_read_time = time() - start
        print(f"Text reading and concatenation time: {text_read_time:.2f} seconds")

    if args.corpus == "mh":
        with open("./MultiHop/corpus.json", encoding="utf-8-sig") as f:
            full_data = json.load(f)  # 加载 JSON 数据
            # 取前一半的记录
            half_length = len(full_data) // 40
            half_data = full_data[:half_length]
            FAKE_TEXT = json.dumps(half_data)  # 转回字符串
    if args.corpus == "mmc":
        print(f"Reading MIMIC-CXR corpus")
        # 设置包含所有 txt 文件的目录路径
        txt_directory = "../data/mimic_ex/dataset"  # 将此路径替换为你的 txt 文件夹路径
        # 存储拼接的文本内容
        FAKE_TEXT = ""

        # 获取所有 txt 文件名，并排序（如果需要）y
        all_files = sorted([f for f in os.listdir(txt_directory) if f.endswith(".txt")])

        start = time()
        # 遍历前 50 个 txt 文件
        for filename in all_files[:20]:  # For testing, using just the first file
            # 遍历 txt_directory 中的所有文件
            if filename.endswith(".txt"):  # 检查是否是 txt 文件
                file_path = os.path.join(txt_directory, filename)
                with open(file_path, encoding="utf-8-sig") as f:
                    # 将每个文件的内容追加到 FAKE_TEXT 中
                    FAKE_TEXT += f.read() + "\n"  # 在每个文件后面添加换行符
        text_read_time = time() - start
        print(f"Text reading and concatenation time: {text_read_time:.2f} seconds")

    if args.corpus == "sci":
        # Load SciBench data
        FAKE_TEXT = load_scibench_texts('./scibench-main/dataset/original/wolfram')

    # Remove existing files if they exist (to reset the data)
    remove_if_exist(f"{args.working_dir}/vdb_entities.json")
    remove_if_exist(f"{args.working_dir}/kv_store_full_docs.json")
    remove_if_exist(f"{args.working_dir}/kv_store_text_chunks.json")
    remove_if_exist(f"{args.working_dir}/kv_store_community_reports.json")
    remove_if_exist(f"{args.working_dir}/graph_chunk_entity_relation.graphml")

    # Initialize GraphRAG
    rag = GraphRAG(
        working_dir=args.working_dir,
        enable_llm_cache=True,
        best_model_func=lambda *args, **kwargs: ollama_model_if_cache(*args, **kwargs, args=args),
        cheap_model_func=lambda *args, **kwargs: ollama_model_if_cache(*args, **kwargs, args=args),
        embedding_func=local_embedding,
    )

    # Insert the concatenated text and index it
    start = time()
    rag.insert(FAKE_TEXT,args.insert_mode)
    indexing_time = time() - start
    print(f"Indexing time: {indexing_time:.2f} seconds")

def extract_choices(answer):
    """
    解析大模型的回答，提取出单选或多选的 A, B, C, or D。
    
    参数:
        answer (str): 大模型的回答文本
    
    返回:
        List[str]: 包含 A, B, C, or D 的列表，按顺序返回。
    """
    # 使用正则表达式查找 A, B, C, D，忽略大小写
    choices = re.findall(r'\b[A-D]\b', answer.upper())
    
    # 返回提取到的选项列表
    return choices

def extract_answer(text):
    # 定义正则表达式来匹配 "response": {"answer": "value"}
    pattern = r'"response":\s*{\s*"answer":\s*"([^"]+)"\s*}'
    
    # 使用 re.search 查找匹配项
    match = re.search(pattern, text)
    
    if match:
        # 提取并返回答案
        answer = match.group(1)
        return answer
    else:
        return None


def query(prompt,mode):
    try:
        rag = GraphRAG(
            working_dir=args.working_dir,
            best_model_func=lambda *args, **kwargs: ollama_model_if_cache(*args, **kwargs, args=args),
            cheap_model_func=lambda *args, **kwargs: ollama_model_if_cache(*args, **kwargs, args=args),
            embedding_func=local_embedding,
        )
        

        response = rag.query(
            prompt, param=QueryParam(mode=mode)
        )
        return response

    except Exception as e:
        print(f"An error occurred while querying: {e}")
        return "Error occurred during query"



        # 构建更精确的 prompt
prompt_law = '''Based on the provided privacy policy context, please answer the following question accurately and concisely. Focus only on information explicitly stated in the policy.

Question: {question}

Please provide a direct answer based solely on the privacy policy information. If the information is not explicitly mentioned in the policy, state that it's not specified.'''

prompt_mmc ='''Answer the question by selecting the following options. This is a {choice_type} choice question.

Question: {question}

Choices:

	•	A: {opa}
	•	B: {opb}
	•	C: {opc}
	•	D: {opd}

Please answer with only A, B, C, or D.
'''
prompt_mh = '''
You are an advanced question-answering model focused on the Multihop dataset.

### Task Description
Below is a question followed by some context from different sources. Please answer the question based on the evidence. The answer to the question is a word or entity. If the provided information is insufficient to answer the question, respond 'Insufficient Information'. Answer directly without explanation in strictly JSON format.

### Evidence
{evidence}

### Question
{question}

### Response Format
 "response": {{
        "answer": "<ANSWER HERE OR 'I don't know'>"
    }}

'''
def parse_not(inputs):
    try:
        if not inputs:
            return '',''
        if '\times' in inputs:
            x,ab=inputs.split('\times')
        elif '\\times' in inputs:
            x,ab=inputs.split('\\times')
        elif '*' in inputs:
            x,ab=inputs.split('*')
        else:
            return inputs
        return x,ab
    except:
        return '',''

def cal_not(inputs):
    
    try:
        x,ab=list(inputs)
        match_number = re.compile('10\^[{]?\ *-?[0-9]+\ *[}]?')
        ab=re.findall(match_number, ab)[0]
        ab=ab[ab.find('^')+1:]
        if '{' in ab:
            ab=ab[ab.find('{')+1:]
        if '}' in ab:
            ab=ab[:ab.find('}')]
        x=x.strip()
        out=float(x)*10**float(ab)
        # print(float(x)*10**float(ab))
        return str(out)
    except:
        print('error')
    return inputs

def remove_boxed(s):
        left = "oxed{" #change
        try:
            assert s[:len(left)] == left
            assert s[-1] == "}"
            answer = s[len(left):-1]
            if "=" in answer:
                answer = answer.split("=")[-1].lstrip(" ")
            return answer
        except:
            return None
def last_boxed_only_string(string):
        idx = string.rfind("oxed") #change
        if idx < 0:
            idx = string.rfind("\\fbox")
            if idx < 0:
                return None
        i = idx
        right_brace_idx = None
        num_left_braces_open = 0
        while i < len(string):
            if string[i] == "{":
                num_left_braces_open += 1
            if string[i] == "}":
                num_left_braces_open -= 1
                if num_left_braces_open == 0:
                    right_brace_idx = i
                    break
            i += 1

        if right_brace_idx == None:
            retval = None
        else:
            retval = string[idx:right_brace_idx + 1]

        return retval
def parse_math_answer(raw_string):
    return remove_boxed(last_boxed_only_string(raw_string))
def zero_prompt(system_prompt, problem_text):
    prompt = ""
    if system_prompt:
        prompt += f"System: {system_prompt}\n"
    prompt += f"User: Q: {problem_text}\nA: The answer is"
    return prompt
def equiv(model_output, answer, unit):
    if model_output is None:
        return False
    print(f"model_output: {model_output}")
    model_output=model_output.replace(',', '')
    try:
        ans=float(answer.strip())
        first=math.isclose(float(model_output.strip()), ans, rel_tol=0.05)
    except:
        first=False
    try: 
        model=model_output.strip().split()[0]
        second=math.isclose(float(model.strip()), ans, rel_tol=0.05)
    except:
        second=False
    if first or second:
        return True
    return False
def remove_not(x):
    match_number = re.compile('[\$]?\ *10\^[{]?\ *-?[0-9]+\ *[}]?\ *[\$]?')
    result=re.findall(match_number, x)
    if len(result) !=0:
        return re.split(match_number, x)[-1]
    return None



@dataclass
class QAEvalResult:
    query: str
    model_answer: str
    ground_truth_answers: List[str]
    precision: float
    recall: float
    accuracy: float
    
def compute_text_overlap(text1: str, text2: str, model) -> float:
    # 使用sentence-transformers计算文本相似度
    embeddings = model.encode([text1, text2])
    similarity = np.dot(embeddings[0], embeddings[1])
    return similarity

def evaluate_answer(model_answer: str, ground_truth_answers: List[str], model, threshold: float = 0.8) -> Tuple[float, float, float]:
    # 计算与每个标准答案的相似度
    similarities = [compute_text_overlap(model_answer, gt, model) for gt in ground_truth_answers]
    
    # 精确率：模型答案与最相似的标准答案的相似度
    precision = max(similarities) if similarities else 0.0
    
    # 召回率：模型答案与所有标准答案的平均相似度
    recall = sum(similarities) / len(similarities) if similarities else 0.0
    
    # 准确率：如果模型答案与任何一个标准答案的相似度超过阈值，则认为是正确的
    accuracy = 1.0 if any(sim >= threshold for sim in similarities) else 0.0
    
    return precision, recall, accuracy

def custom_precision_recall(truth, preds) -> Tuple[float, float]:
    # 1. 找出所有独特类别
    labels = set(truth) | set(preds)
    
    # 2. 定义字典来存储每个类别的TP、FP、FN
    tp = {label: 0 for label in labels}  # True Positives
    fp = {label: 0 for label in labels}  # False Positives
    fn = {label: 0 for label in labels}  # False Negatives

    # 3. 计算TP、FP、FN
    for t, p in zip(truth, preds):
        if t == p:
            tp[t] += 1
        else:
            fp[p] += 1
            fn[t] += 1

    # 4. 计算精确率和召回率
    precision = sum(tp[label] for label in labels) / (sum(tp[label] + fp[label] for label in labels) or 1)
    recall = sum(tp[label] for label in labels) / (sum(tp[label] + fn[label] for label in labels) or 1)
    
    return precision, recall

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run GraphRAG with specified API configuration.")
    parser.add_argument("--base_url", type=str, required=True, help="Base URL for OpenAI API")
    parser.add_argument("--api_key", type=str, required=True, help="Base URL for OpenAI API")
    parser.add_argument("--model", type=str, required=True, help="Base URL for OpenAI API")

    parser.add_argument("--corpus", type=str, help="Path to the corpus file for insertion")
    parser.add_argument("--insert_mode", type=str, required=True, help="origin or beam")
    parser.add_argument("--query_mode", type=str, required=True, help="local or llm")


    args = parser.parse_args()
        # 自动生成工作目录名称
    args.working_dir = f"working_dir_{args.model.replace('/', '_')}_{args.corpus}_{args.insert_mode}_{args.query_mode}_index"
    #args.working_dir = f"working_dir_{args.model.replace('/', '_')}_{args.corpus}_{args.insert_mode}_local"
   
    # 确保工作目录存在
    os.makedirs(args.working_dir, exist_ok=True)
    if args.query_mode !="llm":

        insert()

        # Summary of all performance times
        print("\nPerformance Summary:")
        print(f"Model loading time: {load_model_time:.2f} seconds")
        print(f"Embedding model loading time: {load_embedding_model_time:.2f} seconds")
        print(f"Text reading time: {text_read_time:.2f} seconds")
        print(f"Indexing time: {indexing_time:.2f} seconds")
#----------------query----------------------
#     responses = {}
#     predictions = {}
#     evaluation_results = {}
#     if args.corpus == "law":
#     # 加载 privacy_qa 数据集
#         with open('./LegalBench-RAG/benchmarks/privacy_qa.json', 'r') as f:
#             privacy_qa = json.load(f)
#         start = time()
        
#         for item in privacy_qa['tests']:
#             try:
#                 question = item['query']
#                 prompt = prompt_law.format(question=question)
#                 response = query(prompt, mode=args.query_mode)
#                 responses[question] = response
#                 print(f"Processed question: {question[:50]}...")
#             except Exception as e:
#                 print(f"Error processing question: {question}")
#                 print(f"Error: {e}")
#                 responses[question] = "Error occurred"
        
#         end = time()
#         print(f"Time cost: {end - start:.2f} seconds")
#         model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        
#         predictions = responses
#         output_path = f"responses_{args.model.replace('/', '_')}_{args.corpus}_{args.insert_mode}_{args.query_mode}.json"
#         with open(output_path, "w", encoding="utf-8") as f:
#             json.dump(responses, f, indent=4, ensure_ascii=False)
#         with open('./LegalBench-RAG/benchmarks/privacy_qa.json', 'r') as f:
#             ground_truth = json.load(f)
#         results = []
#         total_precision = 0.0
#         total_recall = 0.0
#         total_accuracy = 0.0  # 新增变量
#         count = 0
#         for item in ground_truth['tests']:
#             query = item['query']
#             if query not in predictions:
#                 continue
                
#             model_answer = predictions[query]
#             ground_truth_answers = [snippet['answer'] for snippet in item['snippets']]
            
#             precision, recall, accuracy = evaluate_answer(model_answer, ground_truth_answers, model)
#             results.append(QAEvalResult(
#                 query=query,
#                 model_answer=model_answer,
#                 ground_truth_answers=ground_truth_answers,
#                 precision=precision,
#                 recall=recall,
#                 accuracy=accuracy  # 新增字段
#             ))
            
#             total_precision += precision
#             total_recall += recall
#             total_accuracy += accuracy
#             count += 1

#         avg_precision = total_precision / count if count > 0 else 0
#         avg_recall = total_recall / count if count > 0 else 0
#         avg_accuracy = total_accuracy / count if count > 0 else 0  # 计算平均准确率
#         f1_score = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall) if (avg_precision + avg_recall) > 0 else 0
    
#         # 保存详细评估结果
#         evaluation_results = {
#             'metrics': {
#                 'average_precision': avg_precision,
#                 'average_recall': avg_recall,
#                 'average_accuracy': avg_accuracy,
#                 'f1_score': f1_score
#             },
#             'detailed_results': [vars(result) for result in results]
#         }

#         print(f"Average Precision: {avg_precision:.4f}")
#         print(f"Average Recall: {avg_recall:.4f}")
#         print(f"Average Accuracy: {avg_accuracy:.4f}")  # 新增输出
#         print(f"F1 Score: {f1_score:.4f}")

#     if args.corpus == "mh":
#         with open('./MultiHop/MultiHopRAG.jsonl', 'r') as f:
#             test_set=[json.loads(line) for line in f]
#         start= time()
#         temp_ground_truth = {}

#         responses = {}
#         for id,item in enumerate(test_set):
#             question = item['query']
#             evidence = item['evidence_list']
#             temp_ground_truth[id] =  item['answer']

#             prompt = prompt_mh.format(question=question, evidence=evidence)
#             # 调用大模型返回 response
#             response = query(prompt, mode=args.query_mode)
#             # 解析 response 并存储结果
#             responses[id] = extract_answer(response)

#         def custom_accuracy_score(truth, preds):
#             # 检查输入长度是否一致
#             if len(truth) != len(preds):
#                 raise ValueError("Length of truth and preds must be the same.")
            
#             # 计算匹配的数量
#             correct_count = sum(t == p for t, p in zip(truth, preds))
            
#             # 计算准确率
#             accuracy = correct_count / len(truth)
#             return accuracy

#         def custom_f1_score(truth, preds, average='macro'):
#             # 1. 找出所有独特类别
#             labels = set(truth) | set(preds)
            
#             # 2. 定义字典来存储每个类别的TP、FP、FN
#             tp = {label: 0 for label in labels}  # True Positives
#             fp = {label: 0 for label in labels}  # False Positives
#             fn = {label: 0 for label in labels}  # False Negatives

#             # 3. 计算TP、FP、FN
#             for t, p in zip(truth, preds):
#                 if t == p:
#                     tp[t] += 1
#                 else:
#                     fp[p] += 1
#                     fn[t] += 1

#             # 4. 计算每个类别的F1分数
#             f1_scores = []
#             for label in labels:
#                 precision = tp[label] / (tp[label] + fp[label]) if tp[label] + fp[label] > 0 else 0
#                 recall = tp[label] / (tp[label] + fn[label]) if tp[label] + fn[label] > 0 else 0
#                 f1 = (2 * precision * recall) / (precision + recall) if precision + recall > 0 else 0
#                 f1_scores.append(f1)

#             # 5. 计算宏平均F1分数
#             if average == 'macro':
#                 return sum(f1_scores) / len(labels) if len(labels) > 0 else 0
#             else:
#                 raise ValueError("Currently, only 'macro' average is supported.")
#         # 创建一个新的字典，用于保存 predictions 中 item 对应的 final_decision
#         ground_truth = {}
#         output_path = f"responses_{args.model.replace('/', '_')}_{args.corpus}_{args.insert_mode}_{args.query_mode}.json"
#         with open(output_path, "w", encoding="utf-8") as f:
#             json.dump(responses, f, indent=4, ensure_ascii=False)
#         predictions = responses
#         for id,item in predictions.items():
        
#             ground_truth[id] =  temp_ground_truth[int(id)]
        
#         assert set(list(ground_truth)) == set(list(predictions)), 'Please predict all and only the instances in the test set.'

#         pmids = list(ground_truth)

#         truth = [ground_truth[pmid] for pmid in pmids]
#         preds = [predictions[pmid] for pmid in pmids]

#         acc = custom_accuracy_score(truth, preds)
#         maf = custom_f1_score(truth, preds, average='macro')

#         print('Accuracy %f' % acc)
#         print('Macro-F1 %f' % maf)
        
        
#         # 计算精确率和召回率
#         precision, recall = custom_precision_recall(truth, preds)
#         print('Precision %f' % precision)
#         print('Recall %f' % recall)

#         # 保存评估结果
#         evaluation_results = {
#             'metrics': {
#                 'accuracy': acc,
#                 'macro_f1': maf,
#                 'precision': precision,
#                 'recall': recall
#             }
#         }


#     if args.corpus == "mmc":
#         with open('./MedMCQA/dev.json', 'r') as f:
#             MedMCQA = [json.loads(line) for line in f]
#         print(len(MedMCQA))
#         responses = {}
#         start= time()

#         for item in MedMCQA:
#             try:
#                 question = item['question']
#                 prompt = prompt_mmc.format(question=question,choice_type = item['choice_type'],opa  = item['opa'],
#                                             opb  = item['opb'], opc = item['opc'], opd  = item['opc'])
#                 # 调用大模型返回 response
#                 response = query(prompt, mode=args.query_mode)
#                 # 解析 response 并存储结果
#                 responses[item['id']] = extract_choices(response)
#             except Exception as e:
#                 # 发生异常时，记录为 'unknown'
#                 print(f"Error processing item {item['id']}: {e}")
#                 responses[item['id']] = "unknown"
#         end = time()
#         print(f"Time cost: {end - start:.2f} seconds")   

#         output_path = f"responses_{args.model.replace('/', '_')}_{args.corpus}_{args.insert_mode}_{args.query_mode}.json"
#         with open(output_path, "w", encoding="utf-8") as f:
#             json.dump(responses, f, indent=4, ensure_ascii=False)
#         ground_truth = {}

#         # 遍历 predictions，查找对应的 ground_truth 中的 "final_decision"
#         for item in MedMCQA:
#             cop_options = item["cop"]

#             # 将 cop 转化为 A、B、C、D 格式
#         # 创建一个字典以映射数字到对应的选项
#             option_mapping = {
#                 1: "A",
#                 2: "B",
#                 3: "C",
#                 4: "D"
#             }
#             option = option_mapping.get(cop_options, "N/A")  # 使用 get 方法防止 KeyError
#             # 连接选项为一个字符串

#             # 输出选项字符串
        
#             ground_truth[item['id']] =  option

#         first_letters = {}

#         keys = []
#         # 遍历字典并提取每个列表的第一个字母
#         for key, value in responses.items():
#             if value:  # 确保列表不为空
#                 keys.append(key)
#                 first_letters[key]= value[0] 
        
#         truth = {}
#         for key in keys:
#             truth[key] = ground_truth[key]

#         print(len(set(list(truth))))
#         print(len(set(list(first_letters))))
#         assert set(list(truth)) == set(list(first_letters)), 'Please predict all and only the instances in the test set.'

#         pmids = list(first_letters)
#         truth = [ground_truth[pmid] for pmid in pmids]
#         preds = [first_letters[pmid] for pmid in pmids]

#         acc = accuracy_score(truth, preds)
#         maf = f1_score(truth, preds, average='macro')
#         precision = precision_score(truth, preds, average='macro', zero_division=0)
#         recall = recall_score(truth, preds, average='macro', zero_division=0)
#         # 保存详细评估结果
#         evaluation_results = {
#             'metrics': {
#                 'average_precision': precision,
#                 'average_recall': recall,
#                 'average_accuracy': acc,
#                 'f1_score': maf
#             },
#         }
#         print('Accuracy %f' % acc)
#         print('Macro-F1 %f' % maf)
#         print('Precision %f' % precision)
#         print('Recall %f' % recall)



#     if args.corpus == "sci":
#         # Load SciBench data
#         with open('./scibench-main/dataset/original/thermo.json', 'r') as f:
#             scibench_data = json.load(f)
#         outputs = []
#         answers = []
#         types = []
#         list_equiv = []
#         model_outputs = []
#         ls_dict = []

#         correct = 0
#         total = 0
#         count = 0
#         responses = {}
#         start = time()

#         for problem_data in scibench_data:
#             prob_book = problem_data["source"]
#             unit_prob = problem_data["unit"]
#             if remove_not(problem_data["unit"]):
#                 unit_prob = remove_not(problem_data["unit"])
#             problem_text = problem_data["problem_text"] + " The unit of the answer is " + problem_data["unit"] + "."
#             sys_prompt = """
# Please provide a clear and step-by-step solution for a scientific problem in the categories of Chemistry, Physics, or Mathematics. The problem will specify the unit of measurement, which should not be included in the answer. Express the final answer as a decimal number with three digits after the decimal point. Conclude the answer by stating "The answer is therefore \\boxed{[ANSWER]}."
# """  # or set a specific system prompt if needed
#             messages = zero_prompt(sys_prompt, problem_text)
#             # Use ollama_model_if_cache instead of call_engine
#             response = query(messages, mode=args.query_mode)

#             model_output = parse_math_answer(response)
#             answer = problem_data["answer_number"]
#             if unit_prob != problem_data["unit"]:
#                 model_output = cal_not(parse_not(model_output))
#                 answer = cal_not((answer, problem_data["unit"]))

#             types.append(prob_book)
#             outputs.append(model_output)
#             answers.append(answer + "@@" + problem_data["unit"])
#             model_outputs.append(response)

#             print("Model output:")
#             print(model_output)
#             print("Correct answer:")
#             print(answer)
#             print(problem_data["unit"])
#             print("--------------------------------------------")
#             try:
#                 res_equiv = equiv(model_output, answer, problem_data["unit"])
#             except:
#                 res_equiv = False
#             if res_equiv:
#                 correct += 1
#             total += 1

#             print(str(correct) + "/" + str(total))
#             list_equiv.append(res_equiv)
#             ls_dict.append({'correct': res_equiv, 'gpt solution': response, "correct answer": answer + "@@" + problem_data["unit"],
#                             "gpt answer": model_output, "source book": prob_book})

#         output_path = f"responses_{args.model.replace('/', '_')}_{args.corpus}_{args.insert_mode}_{args.query_mode}.json"
#         with open(output_path, "w", encoding="utf-8") as f:
#             json.dump(model_outputs, f, indent=4, ensure_ascii=False)
#             # 计算评估指标
#         truth = [1] * len(outputs)  # 所有ground truth都是1
#         preds = []

#         for output, answer in zip(outputs, answers):
#             try:
#                 # 处理 None 值
#                 if output is None:
#                     preds.append(0)
#                     continue
                    
#                 answer_num, answer_unit = answer.split('@@')
#                 res = equiv(output, answer_num, answer_unit)
#                 preds.append(1 if res else 0)
#             except:
#                 preds.append(0)

#         # 计算四个指标
#         acc = accuracy_score(truth, preds)
#         maf = f1_score(truth, preds, average='macro')
#         print(f"truth: {truth}")
#         print(f"preds: {preds}")
#         precision = precision_score(truth, preds, average='macro', zero_division=0)
#         recall = recall_score(truth, preds, average='macro', zero_division=0)

#         print('Accuracy: {:.4f}'.format(acc))
#         print('Macro-F1: {:.4f}'.format(maf)) 
#         print('Precision: {:.4f}'.format(precision))
#         print('Recall: {:.4f}'.format(recall))

#         # Save evaluation results
#         evaluation_results = {
#             'metrics': {
#                 'average_precision': precision,
#                 'average_recall': recall,
#                 'average_accuracy': acc,
#                 'f1_score': maf
#             },
#             'detailed_results': [vars(result) for result in ls_dict]
#         }
# # Define the filename based on model, corpus, and insert/query mode
#     filename = f"{args.model.replace('/', '_')}_{args.corpus}_{args.insert_mode}_{args.query_mode}_evaluation.json"

#     # Write the evaluation results to a JSON file
#     with open(filename, 'w') as outfile:
#         json.dump(evaluation_results, outfile, indent=4)
