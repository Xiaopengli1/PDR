"""
PDR-Eval Script: Run LLM-as-Judge evaluation on JSONL outputs from examples/demo.py.
Requires: openai-compatible client, evaluation_prompt from this package.
"""
import os
import re
import json
from pathlib import Path

try:
    from evaluation.evaluation_prompt import generate_static_score_prompt
except ImportError:
    from evaluation_prompt import generate_static_score_prompt

# Initialize LLM client (OpenAI-compatible API). Set DEEPSEEK_API_KEY or configure below.
try:
    from openai import OpenAI
    client = OpenAI(api_key=os.environ.get("DEEPSEEK_API_KEY", ""), base_url="https://api.deepseek.com")
except Exception:
    client = None  # Set client before running if the above fails.

jsonl_files = list(Path(".").glob("*.jsonl")) or list(Path("..").glob("*.jsonl"))

if not client:
    raise RuntimeError(
        "LLM client not initialized. Set DEEPSEEK_API_KEY or configure OpenAI-compatible client in script."
    )

for path in jsonl_files:
    in_path = str(path)
    out_path = "update_" + in_path

    with open(in_path, "r", encoding="utf-8") as fin, \
        open(out_path, "w", encoding="utf-8") as fout:
        for idx, line in enumerate(fin, 1):
            if not line.strip():
                continue                          # 跳过空行

            try:
                data = json.loads(line)           # 解析 JSON
                gen_article  = data["generation"]
                label_article = data["ground_truth"]

                if gen_article and label_article:
                    prompt = generate_static_score_prompt.format(
                        gen_article=gen_article, label_article=label_article
                    )
                    resp = client.chat.completions.create(
                        model="deepseek-reasoner",
                        messages=[{"role": "user", "content": prompt}],
                    )
                    resp = resp.choices[0].message.content

                    print(resp)


                    match = re.search(r"```[a-zA-Z]*\s*(\{.*\})\s*```", resp, re.S)
                    clean_json = match.group(1) if match else resp   # 如果没有包裹直接用原串

                    data_score = json.loads(clean_json)


                    data["scores"]["Comprehensiveness"] = data_score["Comprehensiveness"]["score"]
                    data["scores"]["Readability"] = data_score["Readability"]["score"]
                    data["scores"]["Content Personalisation"] = data_score["Content Personalisation"]["score"]
                    data["scores"]["Presentation Personalisation"] = data_score["Presentation Personalisation"]["score"]

                fout.write(json.dumps(data, ensure_ascii=False) + "\n")

            except Exception as e:
                # ★ 出错就跳过，保留行号方便排查
                print(f"[WARN] 第 {idx} 行处理失败：{e!r}")
                # 想完全静默就把 print 去掉
                continue
    print("Evaluation complete.")