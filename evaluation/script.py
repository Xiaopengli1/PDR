
import re
import json
from pathlib import Path


jsonl_files = list(Path('.').glob('*.jsonl'))


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

                if gen_article and label_article :

                    resp = client.chat.completions.create(
                        model="deepseek-reasoner",
                        messages=[{"role": "user", "content":generate_static_score_prompt}]
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
        break

    print("处理完毕。")