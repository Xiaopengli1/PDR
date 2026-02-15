import os
import json  # 添加json模块
import logging
import datetime
import math
import re
from collections import Counter
from itertools import permutations
from functools import lru_cache

# try:
#     # Optional -- only needed if you enable stemming/synonyms
import nltk
from nltk.corpus import wordnet as wn
from nltk.stem import PorterStemmer
nltk.download('wordnet')
# except ImportError:
#     nltk = None   # Fallback to exact matching only

_word_re = re.compile(r"\w+", re.UNICODE)
_stemmer = PorterStemmer() if nltk else None

from deepsearcher.online_query import query
from deepsearcher.configuration import Configuration, init_config, ModuleFactory
from deepsearcher.personalized_understanding import personalized_understanding
from deepsearcher.agent import NaiveRAG

from rouge_score import rouge_scorer
import traceback

def calc_single_score(result, ground_truth):
    """计算单组文本的相似度指标"""
    # 初始化 ROUGE 计算器
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    # 计算 ROUGE 指标
    rouge_scores = scorer.score(ground_truth, result)

    meteor_grade = meteor_score(result, ground_truth)
    
    # 计算 Unigram F1
    result_tokens = result.split()
    gt_tokens = ground_truth.split()
    
    # 创建词集合
    result_set = set(result_tokens)
    gt_set = set(gt_tokens)
    
    # 计算交集
    overlap = len(result_set & gt_set)
    
    # 计算精确率、召回率和 F1
    precision = overlap / len(result_set) if len(result_set) > 0 else 0
    recall = overlap / len(gt_set) if len(gt_set) > 0 else 0
    
    if precision + recall > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = 0
    
    return {
        "rouge1": rouge_scores["rouge1"].fmeasure,
        "rouge2": rouge_scores["rouge2"].fmeasure,
        "rougeL": rouge_scores["rougeL"].fmeasure,
        "f1": f1,
        "meteor": meteor_grade
    }

def calc_final_scores(results, ground_truths):
    """计算多组文本的最终平均分数"""
    # 验证输入
    assert len(results) == len(ground_truths), "结果和参考文本数量不一致"
    
    # 初始化累积变量
    total_rouge1 = 0.0
    total_rouge2 = 0.0
    total_rougeL = 0.0
    total_f1 = 0.0
    total_meteor = 0.0
    
    num_pairs = len(results)
    
    # 遍历所有文本对
    for i in range(num_pairs):
        scores = calc_single_score(results[i], ground_truths[i])        
        # 累加分数
        total_rouge1 += scores["rouge1"]
        total_rouge2 += scores["rouge2"]
        total_rougeL += scores["rougeL"]
        total_f1 += scores["f1"]
        total_meteor += scores["meteor"]
    
    # 计算宏平均
    return {
        "rouge1": total_rouge1 / num_pairs,
        "rouge2": total_rouge2 / num_pairs,
        "rougeL": total_rougeL / num_pairs,
        "f1": total_f1 / num_pairs,
        "meteor": total_meteor/num_pairs,
    }


def _tokens(text):
    """Simple whitespace/word tokenizer → list[str]."""
    return _word_re.findall(text.lower())


@lru_cache(maxsize=10_000)
def _stems(word):
    return {_stemmer.stem(word)} if _stemmer else {word}


@lru_cache(maxsize=10_000)
def _synonyms(word):
    if not wn:          # NLTK/WordNet not installed
        return {word}
    syns = set()
    for syn in wn.synsets(word):
        for lemma in syn.lemmas():
            syns.add(lemma.name().lower().replace("_", " "))
    return syns | {word}


def _match_score(cand_tok, ref_tok, use_stem, use_syn):
    """Return 1 if tokens match under current matcher, else 0."""
    if cand_tok == ref_tok:
        return 1
    if use_stem and (_stems(cand_tok) & _stems(ref_tok)):
        return 1
    if use_syn and (_synonyms(cand_tok) & _synonyms(ref_tok)):
        return 1
    return 0


def _best_alignment(cand, ref, use_stem, use_syn):
    """
    Find alignment that maximizes matches.
    Returns (matches, chunks).
    """
    best_m = best_chunks = 0

    # Try all permutations of reference indices up to length 10;
    # beyond that use greedy (tractable for MT eval).
    n = len(cand)
    r = len(ref)
    cand_idx = list(range(n))
    if r <= 10:
        perms = permutations(range(r), min(n, r))
    else:               # fallback greedy for long refs
        perms = [range(min(n, r))]

    for ref_order in perms:
        matched = []
        used_ref = set()
        for i, c_idx in enumerate(cand_idx):
            for j in ref_order:
                if j in used_ref:
                    continue
                if _match_score(cand[c_idx], ref[j], use_stem, use_syn):
                    matched.append((c_idx, j))
                    used_ref.add(j)
                    break
        m = len(matched)
        if m == 0:
            continue

        # chunks = contiguous matches in ref order
        matched.sort(key=lambda x: x[0])
        chunks = 1
        for k in range(1, m):
            if matched[k][1] != matched[k - 1][1] + 1:
                chunks += 1

        if m > best_m or (m == best_m and chunks < best_chunks):
            best_m, best_chunks = m, chunks

    return best_m, best_chunks


def meteor_score(candidate,
                 reference,
                 alpha=0.9, beta=3.0, gamma=0.5,
                 use_stem=True, use_syn=True):
    """
    Compute METEOR (default params from Banerjee & Lavie, 2005).

    Args
    ----
    candidate : str
    reference : str
    alpha     : recall weight (default 0.9 ⇒ recall 9× precision)
    beta, gamma: chunk penalty params
    use_stem  : enable Porter stemming matches (needs NLTK)
    use_syn   : enable WordNet synonym matches (needs NLTK)

    Returns
    -------
    float in [0, 1]
    """
    cand_tok = _tokens(candidate)
    ref_tok = _tokens(reference)

    m, chunks = _best_alignment(cand_tok, ref_tok, use_stem, use_syn)
    if m == 0:
        return 0.0

    precision = m / len(cand_tok)
    recall = m / len(ref_tok)
    f_mean = (precision * recall) / (alpha * precision + (1 - alpha) * recall)

    penalty = gamma * (chunks / m) ** beta
    return f_mean * (1 - penalty)


# Suppress unnecessary logging from third-party libraries
logging.getLogger("httpx").setLevel(logging.WARNING)

config = Configuration()

init_config(config = config)

module_factory = ModuleFactory(config)

llm = module_factory.create_llm()

 
embedding_model = module_factory.create_embedding()
vector_db = module_factory.create_vector_db()

# Load your local data
from deepsearcher.offline_loading import load_from_local_files

task = "report"


_script_dir = os.path.dirname(os.path.abspath(__file__))
_data_root = os.path.join(_script_dir, "..", "data")
if task == "topic":
    test_dir = os.path.join(_data_root, "topic", "test")
elif task == "abstract":
    test_dir = os.path.join(_data_root, "abstract", "test")
elif task == "report":
    test_dir = os.path.join(_data_root, "report")
elif task == "speech":
    test_dir = os.path.join(_data_root, "speech")


results = []
ground_truths = []
all_scores = []
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# 创建JSON Lines文件
jsonl_path = f"evaluation_Ours_R1_{task}_{timestamp}.jsonl"

# 打开文件用于写入JSON Lines
with open(jsonl_path, 'w', encoding='utf-8') as jsonl_file:
    # 遍历 test 下的所有子文件夹（作者目录）
    for author_dir in os.listdir(test_dir):
        author_path = os.path.join(test_dir, author_dir)
        error_info = ""  # 记录错误信息
        
        # 初始化样本结果字典
        sample_result = {
            "author": author_dir,
            "task": task,
            "timestamp": timestamp,
            "generation": None,
            "ground_truth": None,
            "scores": {
                "rouge1": None,
                "rouge2": None,
                "rougeL": None,
                "f1": None,
                "meteor": None
            },
            "error": None,
            "success": False
        }
        
        try:
            if not os.path.isdir(author_path):
                continue
                
            print(f"Processing: {author_path}")
            
            ## Personalized-info-load
            try:
                load_from_local_files(
                    paths_or_directory=os.path.join(author_path, "knowledge_base"),
                    collection_name=f"personalized_knowledge",
                    collection_description="A collection of user expert knowledge..."
                )
            except Exception as e:
                error_info = f"知识库加载失败: {str(e)}"
                raise
                
            try:
                personalized_understanding(
                    paths_or_directory=os.path.join(author_path, "knowledge_base"),
                    llm=llm
                )
            except Exception as e:
                error_info = f"个性化理解失败: {str(e)}"
                raise

            # 读取输入文件
            try:
                input_path = os.path.join(author_path, 'input.txt')
                with open(input_path, 'r', encoding='utf-8') as f:
                    input_query = f.read()
            except Exception as e:
                error_info = f"输入文件读取失败: {str(e)}"
                raise

            # 模型查询
            try:
                result = query(
                    original_query=input_query,
                    max_iter=2,
                    personalized_info_address=author_path + "/knowledge_base/personalized_summary.json"
                )
                sample_result["generation"] = result[0]  # 保存生成结果
            except Exception as e:
                error_info = f"模型查询失败: {str(e)}"
                raise

            # 读取真实结果
            try:
                output_path = os.path.join(author_path, 'output.txt')
                with open(output_path, 'r', encoding='utf-8') as f:
                    ground_truth = f.read()
                sample_result["ground_truth"] = ground_truth  # 保存真实结果
            except Exception as e:
                error_info = f"真实结果读取失败: {str(e)}"
                raise

            # 计算分数
            try:
                scores = calc_single_score(result[0], ground_truth)
                # 更新样本分数
                sample_result["scores"] = {
                    "rouge1": scores["rouge1"],
                    "rouge2": scores["rouge2"],
                    "rougeL": scores["rougeL"],
                    "f1": scores["f1"],
                    "meteor": scores["meteor"]
                }
                sample_result["success"] = True
            except Exception as e:
                error_info = f"分数计算失败: {str(e)}"
                raise

            # 打印结果
            print("-----------Generation---------------")
            print(result[0])
            print("-----------Ground Truth-------------")
            print(ground_truth)
            print("------------------------------------")
            
            print(f"ROUGE-1 F1: {scores['rouge1']}")
            print(f"ROUGE-2 F1: {scores['rouge2']}")
            print(f"ROUGE-L F1: {scores['rougeL']}")
            print(f"F1: {scores['f1']}")
            print(f"METEOR: {scores['meteor']}")

            results.append(result[0])
            ground_truths.append(ground_truth)

            # 记录当前样本的分数
            sample_scores = {
                "rouge1": scores["rouge1"],
                "rouge2": scores["rouge2"],
                "rougeL": scores["rougeL"],
                "f1": scores["f1"],
                "meteor": scores['meteor']
            }
            all_scores.append(sample_scores)
            
        except Exception as e:
            # 打印详细错误日志
            traceback.print_exc()
            print(f"处理 {author_dir} 时出错: {error_info}")
            sample_result["error"] = error_info
            
        finally:
            print("Start printing!!!!")
            # 将样本结果以JSON行格式写入文件
            jsonl_file.write(json.dumps(sample_result, ensure_ascii=False) + '\n')
            jsonl_file.flush()  # 确保立即写入磁盘
            print("------------------------------------")

# 最终评分（仅对成功处理的样本）
if results and ground_truths:
    print("--------------Final Score----------------")
    try:
        final_scores = calc_final_scores(results, ground_truths)
        print(final_scores)
        
        # 创建最终评分记录
        final_record = {
            "type": "final_scores",
            "task": task,
            "timestamp": timestamp,
            "scores": final_scores,
            "sample_count": len(results)
        }
        
        # 将最终评分追加到JSON Lines文件
        with open(jsonl_path, 'a', encoding='utf-8') as jsonl_file:
            jsonl_file.write(json.dumps(final_record, ensure_ascii=False) + '\n')
            
    except Exception as e:
        print(f"最终评分失败: {str(e)}")
        # 创建错误记录
        error_record = {
            "type": "final_scores_error",
            "task": task,
            "timestamp": timestamp,
            "error": str(e)
        }
        with open(jsonl_path, 'a', encoding='utf-8') as jsonl_file:
            jsonl_file.write(json.dumps(error_record, ensure_ascii=False) + '\n')
    print("--------------Final Score----------------")
else:
    print("警告：没有成功处理的样本，跳过最终评分")
    # 创建空结果记录
    empty_record = {
        "type": "final_scores_empty",
        "task": task,
        "timestamp": timestamp,
        "message": "No successful samples for final scoring"
    }
    with open(jsonl_path, 'a', encoding='utf-8') as jsonl_file:
        jsonl_file.write(json.dumps(empty_record, ensure_ascii=False) + '\n')

print(f"结果已保存至: {jsonl_path}")