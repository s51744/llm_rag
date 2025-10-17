# llm_exam_tester_rag.py

import os
import time
import re
import json
import argparse
from datetime import datetime
from pathlib import Path
import difflib

import pandas as pd
import requests

# 匯入你現有的 LLMExamTester 定義（假設與此檔同目錄）
# 如果 LLMExamTester 在另一檔中，請調整 import 路徑。
from llm_exam_tester_full import LLMExamTester

# 匯入向量庫
from rag_system.vector_store import TCMVectorStore

class LLMExamTesterRAG(LLMExamTester):
    """
    在原本 LLMExamTester 的基礎上加入 RAG：
    - 使用 TCMVectorStore 檢索相關片段
    - 把檢索到的片段注入 prompt，再呼叫原本的 call_llm()
    """
    def __init__(self, port="8090", limit=None, delay=0.5, vectordb_dir="rag_system/vectordb", top_k=5, score_threshold=0.5, max_context_tokens=2048):
        # Pass max_context_tokens down to base class so it's available project-wide
        super().__init__(port=port, limit=limit, delay=delay, max_context_tokens=max_context_tokens)
        self.vectordb_dir = vectordb_dir
        self.top_k = top_k
        self.score_threshold = score_threshold
        # Local copy for convenience
        self.max_context_tokens = max_context_tokens

        # 初始化向量庫
        self.vector_store = TCMVectorStore(store_dir=self.vectordb_dir)
        loaded = self.vector_store.load_index()
        if not loaded:
            print("⚠️ 無法載入向量索引（rag_system/vectordb）。請先執行 setup/build index。")
        else:
            print("✓ 向量索引載入成功")

    def build_rag_prompt(self, question_text: str, contexts: list, top_n: int = 3, snippet_chars: int = 500, buffer_chars: int = 50) -> str:
        """
        將檢索到的 contexts 與題目組合成 prompt（中文）。
        contexts: list of dict, each dict 包含 'content' 與 metadata（book_title, chapter_title, similarity_score, related_option）
        參數:
        - top_n: 取最相關前幾段
        - snippet_chars: 每段取出的最大字元數（先行切片）
        - buffer_chars: 在計算可用 chars 時保留的緩衝空間（避免超出模型 window）
        回傳: 完整 prompt 字串
        """
        # Defensive: 若 contexts 為 None 或空 list
        if not contexts:
            combined_contexts = "【參考資料】無檢索到相關資料。"
            print("[DEBUG] contexts: 無")
            contexts = []  # 確保後續處理不出錯
        else:
            # 只保留最相關 top_n 條
            contexts = contexts[:top_n]

            # 建構每段 snippet 與 metadata
            pieces = []
            for i, c in enumerate(contexts):
                content = (c.get('content') or "").strip().replace("\n", " ")
                meta = f"{c.get('book_title','unknown')} | {c.get('chapter_title','')}".strip()
                score = c.get('similarity_score', 0.0)
                related_option = c.get('related_option', None)
                related_tag = f" [最相關選項: {related_option}]" if related_option else ""
                snippet = content[:snippet_chars]
                pieces.append(f"【來源 {i+1}】{meta} (score={score:.3f}){related_tag}\n{snippet}")

            combined_contexts = "\n\n".join(pieces)
            print(f"[DEBUG] contexts: {len(contexts)} 條, combined_contexts 長度: {len(combined_contexts)}")

        # 控制輸入 LLM 的總上下文大小（使用 max_context_tokens if defined）
        if getattr(self, "max_context_tokens", None):
            try:
                # 優先使用 tokenizer 計算，若沒有 tokenizer 則採用 heuristics (chars per token)
                max_tokens = int(self.max_context_tokens)
                chars_per_token = 4  # fallback heuristic
                if hasattr(self, "tokenizer") and self.tokenizer is not None:
                    # 嘗試用 tokenizer 計算更精準的 tokens -> chars 映射
                    # 這段只在有 tokenizer 時生效
                    sample = "。".join(combined_contexts.split("\n")[:3])
                    tok_count = len(self.tokenizer.encode(sample))
                    if tok_count > 0:
                        chars_per_token = max(1.0, len(sample) / tok_count)
                max_chars = int(max_tokens * chars_per_token)

                # 為 question_text 與 instructions 保留空間
                reserve_for_question = len(question_text) + 200  # 預留 200 chars for instruction overhead
                allowed_context_chars = max(0, max_chars - reserve_for_question - buffer_chars)

                if allowed_context_chars <= 0:
                    # 無空間放 context，改為最小提示
                    print("[DEBUG] allowed_context_chars <= 0，將只保留參考資料說明段")
                    combined_contexts = combined_contexts[:0]
                elif len(combined_contexts) > allowed_context_chars:
                    print("[DEBUG] combined_contexts 超過 allowed_context_chars，進行截斷")
                    # 盡量以段落為單位截斷（保留完整行）
                    truncated = combined_contexts[:allowed_context_chars]
                    # 若最後一行被截斷，剔除不完整片段
                    if "\n" in truncated:
                        truncated = truncated.rsplit("\n", 1)[0]
                    combined_contexts = truncated
            except Exception as e:
                print(f"[DEBUG] context 截斷失敗: {e}")

        prompt = (
            f"""以下提供多段中醫教材的參考資料（來自你的教材索引），請**僅根據下列資料的內容**來回答題目，並優先參考每段資料後方標註的「最相關選項」；若資料不足請回答「資料不足」或「無法從提供資料推斷出答案」。

參考資料：
{combined_contexts}

--- 
題目如下，請只回答選項字母（A、B、C、D）：
{question_text}

請直接輸出一個大寫英文字母（A/B/C/D），不要額外說明。"""
        )
        return prompt

    def process_csv_file(self, csv_path):
        """
        覆寫原有 process_csv_file，讓每題都先檢索並把 context 注入 prompt。
        其餘結果格式與原本一致。
        """
        file_basename = os.path.basename(csv_path)
        limit_text = f" (限制 {self.limit} 題)" if self.limit else ""
        print(f"\n處理文件: {file_basename}{limit_text}")

        try:
            df = pd.read_csv(csv_path, encoding='utf-8')
            if self.limit:
                df = df.head(self.limit)

            file_results = []
            total_in_file = len(df)
            print(f"開始處理 {total_in_file} 題...")


            for index, row in df.iterrows():
                self.total_questions += 1
                question_num = row['題號']
                correct_answer = row['答案'].strip().upper()

                progress = f"[{index + 1}/{total_in_file}]"
                print(f"{progress} 題目 {question_num}...", end=' ')

                # 取得原始格式化題目
                question_text = self.format_question(row)

                # 使用向量庫搜尋（含內容） - 以題目+所有選項作為查詢
                contexts = []
                try:
                    options_text = f"A. {row['選項A']} B. {row['選項B']} C. {row['選項C']} D. {row['選項D']}"
                    search_query = f"{row['題目']} {options_text}"
                    contexts = self.vector_store.search_with_content(search_query, top_k=self.top_k, score_threshold=self.score_threshold)
                except Exception as e:
                    print(f"\n⚠️ 檢索出現錯誤: {e}")
                    contexts = []

                # --- re-ranking: 計算每個 context 與四個選項的相似度，標註最相關選項 ---
                options = {
                    'A': str(row['選項A']),
                    'B': str(row['選項B']),
                    'C': str(row['選項C']),
                    'D': str(row['選項D'])
                }
                for c in contexts:
                    content = c.get('content', '') or ''
                    # 用 difflib SequenceMatcher 計算與每個選項的相似度
                    best_score = 0
                    best_option = None
                    for opt_key, opt_text in options.items():
                        # 取最大連續子串相似度
                        score = difflib.SequenceMatcher(None, content, opt_text).ratio()
                        if score > best_score:
                            best_score = score
                            best_option = opt_key
                    c['related_option'] = best_option
                    c['related_option_score'] = best_score
                # 依照與選項的最大相似度降冪排序
                contexts = sorted(contexts, key=lambda c: c.get('related_option_score', 0), reverse=True)

                # 建構 RAG prompt（若無 contexts 會回傳原題目）
                rag_prompt_text = self.build_rag_prompt(question_text, contexts)

                # 呼叫 LLM（使用原本 call_llm）
                start_time = time.time()
                llm_response = self.call_llm(rag_prompt_text)
                response_time = time.time() - start_time

                # 解析 LLM 回答
                llm_answer = self.extract_answer_from_response(llm_response)

                is_correct = llm_answer == correct_answer
                if is_correct:
                    self.correct_answers += 1
                    print("✓")
                else:
                    status_text = f"✗ ({correct_answer}→{llm_answer})"
                    print(status_text)

                # 儲存結果（增加 contexts metadata）
                result = {
                    'file_name': file_basename,
                    'question_number': question_num,
                    'question_text': row['題目'],
                    'option_a': row['選項A'],
                    'option_b': row['選項B'],
                    'option_c': row['選項C'],
                    'option_d': row['選項D'],
                    'correct_answer': correct_answer,
                    'llm_response': llm_response,
                    'llm_answer': llm_answer,
                    'is_correct': is_correct,
                    'response_time_seconds': round(response_time, 2),
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'retrieved_contexts': [
                        {
                            'book_title': c.get('book_title'),
                            'chapter_title': c.get('chapter_title'),
                            'similarity_score': c.get('similarity_score'),
                            'snippet': (c.get('content','') or '')[:500]
                        } for c in contexts
                    ]
                }

                file_results.append(result)
                self.results.append(result)

                if self.delay > 0:
                    time.sleep(self.delay)

            print(f"完成處理文件 {file_basename}")

        except Exception as e:
            print(f"處理文件 {csv_path} 時出錯: {e}")

        return file_results


def main():
    parser = argparse.ArgumentParser(description='LLM RAG 中醫考題測驗系統')
    parser.add_argument('--limit', type=int, default=None, help='每個文件最多處理的題目數量 (預設: 全部)')
    parser.add_argument('--delay', type=float, default=0.5, help='API 調用間隔時間(秒)')
    parser.add_argument('--port', type=str, default='8090', help='LLM 服務埠號')
    parser.add_argument('--top_k', type=int, default=5, help='檢索回傳 top_k 個片段')
    parser.add_argument('--threshold', type=float, default=0.5, help='相似度閾值（score）')
    parser.add_argument('--max-context-tokens', type=int, default=500, help='(選用) 模型上下文視窗大小（tokens），用於 RAG prompt 截斷')
    args = parser.parse_args()

    base_dir = Path("C:/Users/Personal/Downloads/BianCang-main/BianCang-main")
    csv_files = [
        base_dir / "llm_exam" / "中醫基礎醫學_106年考題(一).csv",
        base_dir / "llm_exam" / "中醫基礎醫學_106年考題(二).csv",
        base_dir / "llm_exam" / "中醫基礎醫學_112年考題(一).csv",
        base_dir / "llm_exam" / "中醫基礎醫學_112年考題(二).csv"
    ]

    # results output
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = base_dir / "llm_exam_results"
    results_dir.mkdir(parents=True, exist_ok=True)
    output_csv = results_dir / f"llm_exam_results_rag_{timestamp}.csv"
    output_jsonl = results_dir / f"llm_exam_results_rag_{timestamp}.jsonl"
    output_txt = results_dir / f"llm_exam_statistics_rag_{timestamp}.txt"

    tester = LLMExamTesterRAG(port=args.port, limit=args.limit, delay=args.delay, top_k=args.top_k, score_threshold=args.threshold, max_context_tokens=args.max_context_tokens)

    print("LLM RAG 中醫基礎醫學考題測驗系統")
    print("="*50)
    print(f"LLM API URL: {tester.base_url}")
    if args.limit:
        print(f"題目限制: 每個文件最多 {args.limit} 題")
    else:
        print("題目限制: 處理全部題目")
    print(f"檢索 top_k: {args.top_k}, 相似度閾值: {args.threshold}")
    print(f"API 調用間隔: {args.delay} 秒")

    # Test API connection
    try:
        test_response = requests.get(f"{tester.base_url}", timeout=5)
        print("✓ LLM API 連接正常")
    except:
        print("✗ 警告: 無法連接到 LLM API，請確認服務正在運行")


    processed_files = 0
    for csv_file in csv_files:
        if csv_file.exists():
            tester.process_csv_file(str(csv_file))
            processed_files += 1
        else:
            print(f"警告: 找不到文件 {csv_file}")

    if processed_files == 0:
        print("錯誤: 沒有找到任何有效的 CSV 文件")
        return

    # Print & save stats
    tester.print_statistics()
    tester.save_results_to_csv(str(output_csv))
    tester.save_statistics_to_txt(str(output_txt))

    # 保存 JSONL（包含檢索片段）
    with open(output_jsonl, "w", encoding="utf-8") as fout:
        for r in tester.results:
            fout.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"詳細 JSONL 結果已保存到: {output_jsonl}")

    print("\n測驗完成！")
    print(f"處理了 {processed_files} 個文件，共 {tester.total_questions} 題")
    if tester.total_questions > 0:
        print(f"最終準確率: {(tester.correct_answers / tester.total_questions * 100):.2f}%")

if __name__ == "__main__":
    main()
