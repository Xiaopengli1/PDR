import os
import json
import logging
import datetime
import traceback
import re
from collections import Counter
from itertools import permutations
from functools import lru_cache

# NLP dependencies
import nltk
from nltk.corpus import wordnet as wn
from nltk.stem import PorterStemmer
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

_word_re = re.compile(r"\w+", re.UNICODE)
_stemmer = PorterStemmer()

from deepsearcher.online_query import query
from deepsearcher.configuration import Configuration, init_config, ModuleFactory
from deepsearcher.personalized_understanding import personalized_understanding
from deepsearcher.llm.qwen3 import QwenLLM  # 明确使用Qwen3
from deepsearcher.offline_loading import load_from_local_files
from rouge_score import rouge_scorer

def calc_single_score(result, ground_truth):
    """计算单组文本的相似度指标"""
    # 初始化ROUGE计算器
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    # 计算ROUGE指标
    rouge_scores = scorer.score(ground_truth, result)
    
    # 计算METEOR指标
    meteor_grade = meteor_score(result, ground_truth)
    
    # 计算Unigram F1
    result_tokens = result.split()
    gt_tokens = ground_truth.split()
    
    # 创建词集合
    result_set = set(result_tokens)
    gt_set = set(gt_tokens)
    
    # 计算交集
    overlap = len(result_set & gt_set)
    
    # 计算精确率、召回率和F1
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
        "meteor": total_meteor / num_pairs,
    }

def _tokens(text):
    """简单空白/词标记器 → list[str]"""
    return _word_re.findall(text.lower())

@lru_cache(maxsize=10_000)
def _stems(word):
    return {_stemmer.stem(word)}

@lru_cache(maxsize=10_000)
def _synonyms(word):
    syns = set()
    for syn in wn.synsets(word):
        for lemma in syn.lemmas():
            syns.add(lemma.name().lower().replace("_", " "))
    return syns | {word}

def _match_score(cand_tok, ref_tok):
    """如果令牌匹配返回1，否则返回0"""
    if cand_tok == ref_tok:
        return 1
    if _stems(cand_tok) & _stems(ref_tok):
        return 1
    if _synonyms(cand_tok) & _synonyms(ref_tok):
        return 1
    return 0

def _best_alignment(cand, ref):
    """
    寻找最大化匹配的对齐方式
    返回 (匹配数, 块数)
    """
    best_m = best_chunks = 0
    n = len(cand)
    r = len(ref)
    cand_idx = list(range(n))
    
    # 对长参考文本使用贪心算法
    perms = [range(min(n, r))] if r > 10 else permutations(range(r), min(n, r))

    for ref_order in perms:
        matched = []
        used_ref = set()
        for i, c_idx in enumerate(cand_idx):
            for j in ref_order:
                if j in used_ref:
                    continue
                if _match_score(cand[c_idx], ref[j]):
                    matched.append((c_idx, j))
                    used_ref.add(j)
                    break
        m = len(matched)
        if m == 0:
            continue

        # 块 = 参考顺序中的连续匹配
        matched.sort(key=lambda x: x[0])
        chunks = 1
        for k in range(1, m):
            if matched[k][1] != matched[k-1][1] + 1:
                chunks += 1

        if m > best_m or (m == best_m and chunks < best_chunks):
            best_m, best_chunks = m, chunks

    return best_m, best_chunks

def meteor_score(candidate, reference, alpha=0.9, beta=3.0, gamma=0.5):
    """
    计算METEOR分数 (Banerjee & Lavie, 2005的默认参数)
    
    参数:
    candidate : str
    reference : str
    
    返回:
    float in [0, 1]
    """
    cand_tok = _tokens(candidate)
    ref_tok = _tokens(reference)

    m, chunks = _best_alignment(cand_tok, ref_tok)
    if m == 0:
        return 0.0

    precision = m / len(cand_tok)
    recall = m / len(ref_tok)
    f_mean = (precision * recall) / (alpha * precision + (1 - alpha) * recall)

    penalty = gamma * (chunks / m) ** beta
    return f_mean * (1 - penalty)

# 配置日志
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

# 初始化配置
config = Configuration()
init_config(config=config)
module_factory = ModuleFactory(config)

# 明确使用Qwen3模型
llm = QwenLLM(model_name="Qwen/Qwen3-14B")

logging.info("Qwen3-14B模型初始化成功")

# 任务配置

# task = "abstract"
# test_dir = "./data/Longlamp/abstract/test"

task = "topic"


if task=="topic":
    test_dir = "../data/topic/test"
elif task=="abstract":
    test_dir = "../data/abstract/test"
elif task=="report":
    test_dir = "../data/report"
elif task=="speech":
    test_dir = "../data/speech"

results = []
ground_truths = []
all_scores = []
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
jsonl_path = f"evaluation_Qwen3_14B_{task}_{timestamp}.jsonl"

# 处理每个作者目录
with open(jsonl_path, 'w', encoding='utf-8') as jsonl_file:
    for author_dir in os.listdir(test_dir):
        # print("-------------")
        # print(author_dir)
        # print("-------------")
        # if author_dir!="Tingjun Hou":
        #     continue
        author_path = os.path.join(test_dir, author_dir)
        sample_result = {
            "author": author_dir,
            "task": task,
            "generation": None,
            "ground_truth": None,
            "scores": None,
            "error": None,
            "success": False,
            "timestamp": timestamp
        }
        
        if not os.path.isdir(author_path):
            continue
            
        try:
            logging.info(f"处理作者: {author_dir}")
            kb_path = os.path.join(author_path, "knowledge_base")


            from deepsearcher.llm.qwen3 import QwenLLM
            from deepsearcher.offline_loading import load_from_local_files


            # 加载个性化知识库
            load_from_local_files(paths_or_directory=kb_path, collection_name = f"personalized_knowledge",
                                collection_description = "A collection of user expert knowledge, including writing notes, and answers to some certain questions.")

            personalized_understanding(paths_or_directory=kb_path, llm=llm)
            logging.info("个性化知识库加载和理解完成")
            
            # 读取输入
            with open(author_path+'/input.txt', 'r', encoding='utf-8') as f:
                input_query = f.read()
            
            # 执行查询
            result = query(original_query=input_query, max_iter=1, personalized_info_address=kb_path+ "/personalized_summary.json")[0]
            sample_result["generation"] = result
            
            # 读取真实结果
            with open(author_path+'/output.txt', 'r', encoding='utf-8') as f:
                ground_truth = f.read()
            sample_result["ground_truth"] = ground_truth
            
            # 计算分数
            scores = calc_single_score(result, ground_truth)
            sample_result["scores"] = scores
            sample_result["success"] = True
            
            # 收集结果
            results.append(result)
            ground_truths.append(ground_truth)
            all_scores.append(scores)
            
            logging.info(f"分数计算完成: ROUGE-L={scores['rougeL']:.4f}, METEOR={scores['meteor']:.4f}")
            
        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}"
            sample_result["error"] = error_msg
            logging.error(f"处理 {author_dir} 时出错: {error_msg}")
            traceback.print_exc()
        
        finally:
            # 写入JSONL记录
            jsonl_file.write(json.dumps(sample_result, ensure_ascii=False) + '\n')
            jsonl_file.flush()

# 计算最终分数
if results and ground_truths:
    final_scores = calc_final_scores(results, ground_truths)
    logging.info("最终分数计算完成")
    
    # 添加最终分数记录
    final_record = {
        "type": "final_scores",
        "task": task,
        "timestamp": timestamp,
        "scores": final_scores,
        "sample_count": len(results)
    }
    
    with open(jsonl_path, 'a', encoding='utf-8') as jsonl_file:
        jsonl_file.write(json.dumps(final_record, ensure_ascii=False) + '\n')
    
    # 打印最终结果
    print("\n============== 最终结果 ==============")
    for metric, score in final_scores.items():
        print(f"{metric.upper()}: {score:.4f}")
    print("=" * 40)
else:
    logging.warning("没有有效样本可计算最终分数")

print(f"\n评估结果已保存至: {os.path.abspath(jsonl_path)}")





