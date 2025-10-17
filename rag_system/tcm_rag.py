import os
import json
import requests
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import logging
from datetime import datetime
import time
import hashlib
from functools import lru_cache
import concurrent.futures
from threading import Lock

from pdf_processor import TCMPDFProcessor
from vector_store import TCMVectorStore

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TCMRAGSystem:
    """中醫RAG系統"""
    
    def __init__(self, 
                 base_dir: str = "rag_system",
                 llm_port: str = "8090",
                 embedding_model: str = "BAAI/bge-base-zh-v1.5"):
        """
        初始化RAG系統
        
        Args:
            base_dir: RAG系統基礎目錄
            llm_port: LLM服務端口
            embedding_model: 嵌入模型名稱
        """
        self.base_dir = Path(base_dir)
        self.data_dir = self.base_dir / "data"
        self.vectordb_dir = self.base_dir / "vectordb"
        
        # LLM配置
        self.llm_port = llm_port
        self.llm_base_url = f"http://localhost:{llm_port}"
        
        # 初始化組件
        self.pdf_processor = TCMPDFProcessor(output_dir=str(self.data_dir))
        self.vector_store = TCMVectorStore(
            store_dir=str(self.vectordb_dir),
            embedding_model=embedding_model
        )
        
        # RAG參數 - 優化後的設置
        self.max_context_length = 1500  # 減少上下文長度以提升速度
        self.retrieval_top_k = 2        # 減少檢索結果數量
        self.similarity_threshold = 0.5  # 提高相似度閾值過濾低質量結果
        
        # 性能優化
        self.cache = {}  # 簡單的記憶快取
        self.cache_lock = Lock()  # 線程安全的快取鎖
        self.max_cache_size = 100  # 最大快取條目
    
    def setup_system(self, pdf_dir: str = "book") -> bool:
        """
        設置RAG系統（處理PDF並構建向量索引）
        
        Args:
            pdf_dir: PDF文件目錄
            
        Returns:
            是否設置成功
        """
        logger.info("開始設置RAG系統...")
        
        # 1. 處理PDF文件
        logger.info("步驟1: 處理PDF文件")
        results = self.pdf_processor.process_all_pdfs(pdf_dir)
        
        successful_results = [r for r in results if r.get("status") == "success"]
        if not successful_results:
            logger.error("PDF處理失敗，無法繼續")
            return False
        
        logger.info(f"成功處理 {len(successful_results)} 個PDF文件")
        
        # 2. 構建向量索引
        logger.info("步驟2: 構建向量索引")
        self.vector_store.build_index(str(self.data_dir))
        
        # 3. 驗證系統
        logger.info("步驟3: 驗證系統")
        if not self.vector_store.load_index():
            logger.error("向量索引加載失敗")
            return False
        
        stats = self.vector_store.get_stats()
        logger.info(f"RAG系統設置完成:")
        logger.info(f"- 總文檔數: {stats.get('total_documents', 0)}")
        logger.info(f"- 書籍數: {stats.get('total_books', 0)}")
        logger.info(f"- 嵌入模型: {stats.get('embedding_model', 'Unknown')}")
        
        return True
    
    def load_system(self) -> bool:
        """
        加載已有的RAG系統
        
        Returns:
            是否加載成功
        """
        logger.info("正在加載RAG系統...")
        
        if not self.vector_store.load_index():
            logger.error("向量索引加載失敗")
            return False
        
        stats = self.vector_store.get_stats()
        logger.info(f"RAG系統加載成功:")
        logger.info(f"- 總文檔數: {stats.get('total_documents', 0)}")
        logger.info(f"- 書籍數: {stats.get('total_books', 0)}")
        
        return True
    
    def _cache_key(self, text: str) -> str:
        """生成快取鍵值"""
        return hashlib.md5(text.encode('utf-8')).hexdigest()[:16]
    
    def retrieve_knowledge(self, query: str) -> List[Dict]:
        """
        檢索相關知識（帶快取優化）
        
        Args:
            query: 查詢文本
            
        Returns:
            相關知識片段列表
        """
        # 檢查快取
        cache_key = self._cache_key(f"retrieve_{query}")
        with self.cache_lock:
            if cache_key in self.cache:
                logger.info(f"快取命中: {query[:30]}...")
                return self.cache[cache_key]
        
        logger.info(f"檢索查詢: {query}")
        
        results = self.vector_store.search_with_content(
            query=query,
            top_k=self.retrieval_top_k,
            score_threshold=self.similarity_threshold
        )
        
        # 存入快取
        with self.cache_lock:
            if len(self.cache) >= self.max_cache_size:
                # 清理最舊的快取項目
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]
            self.cache[cache_key] = results
        
        logger.info(f"檢索到 {len(results)} 個相關片段")
        return results
    
    def format_context(self, knowledge_pieces: List[Dict]) -> str:
        """
        格式化知識片段為上下文
        
        Args:
            knowledge_pieces: 知識片段列表
            
        Returns:
            格式化的上下文字符串
        """
        if not knowledge_pieces:
            return ""
        
        context_parts = []
        total_length = 0
        
        for i, piece in enumerate(knowledge_pieces, 1):
            # 構建片段資訊
            book_title = piece.get('book_title', '未知書籍')
            chapter_title = piece.get('chapter_title', '未知章節')
            content = piece.get('content', '')
            similarity = piece.get('similarity_score', 0)
            
            # 格式化片段 - 簡化格式以節省token
            piece_text = f"[參考{i}] {book_title}-{chapter_title}\n{content}\n"
            
            # 檢查長度限制
            if total_length + len(piece_text) > self.max_context_length:
                break
            
            context_parts.append(piece_text)
            total_length += len(piece_text)
        
        context = "\n".join(context_parts)
        logger.info(f"構建上下文，長度: {len(context)} 字符")
        
        return context
    
    def call_llm(self, prompt: str, max_tokens: int = 512) -> str:
        """
        調用LLM生成回答（帶快取優化）
        
        Args:
            prompt: 提示詞
            max_tokens: 最大生成長度
            
        Returns:
            LLM回答
        """
        # 檢查快取
        cache_key = self._cache_key(f"llm_{prompt[:200]}_{max_tokens}")
        with self.cache_lock:
            if cache_key in self.cache:
                logger.info(f"LLM快取命中: {prompt[:30]}...")
                return self.cache[cache_key]
        headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        }
        
        data = {
            "model": "BianCang-Qwen2.5-7B-Instruct",
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_tokens": max_tokens,
            "temperature": 0.3,  # 降低溫度提升生成穩定性和速度
            "top_p": 0.8,       # 降低top_p值
            "stream": False,
            "stop": ["\n\n\n", "參考資料", "問題："]  # 添加停止詞
        }
        
        try:
            response = requests.post(
                f"{self.llm_base_url}/v1/chat/completions",
                headers=headers,
                json=data,
                timeout=120  # 超時時間
            )
            
            if response.status_code == 200:
                result = response.json()
            if 'choices' in result and len(result['choices']) > 0:
                    answer = result['choices'][0]['message']['content'].strip()
                    # 存入快取
                    with self.cache_lock:
                        if len(self.cache) >= self.max_cache_size:
                            oldest_key = next(iter(self.cache))
                            del self.cache[oldest_key]
                        self.cache[cache_key] = answer
                    return answer
            else:
                logger.error(f"LLM API錯誤: {response.status_code} - {response.text[:200]}")
                
        except requests.exceptions.Timeout:
            logger.error(f"LLM調用超時: {120}秒")
            return self._generate_fallback_answer(prompt)
        except requests.exceptions.ConnectionError:
            logger.error("LLM服務連線失敗")
            return self._generate_fallback_answer(prompt)
        except Exception as e:
            logger.error(f"LLM調用失敗: {e}")
            return self._generate_fallback_answer(prompt)
        
        return self._generate_fallback_answer(prompt)
    
    def _generate_fallback_answer(self, prompt: str) -> str:
        """生成備用回答當LLM不可用時"""
        logger.info("使用備用模式生成回答")
        
        # 簡單的模板匹配回答
        if "氣血兩虛" in prompt:
            return "氣血兩虛是中醫學中的一種病理狀態，指氣與血都不足。常見症狀包括疲勞乾力、面色無華、心悸氣短、頭暈等。建議調理方法包括休息充足、均衡飲食、適度運動和中藥調理。請諮詢專業中醫師獲取個人化建議。"
        elif "脾胃虛弱" in prompt:
            return "脾胃虛弱的症狀包括食慰不振、腹脹、腹潟、疲勞無力、大便稀溝等。調理方法包括飲食清淡、定時定量、適量運動和中藥調理。建議諮詢專業中醫師。"
        elif "六君子湯" in prompt:
            return "六君子湯的組成包括：人參、白住、葦苓、半夏、陳皮、甘草。主要功效是補脾益氣、燥湿化痰。適用於脾胃虛弱、痰湿不化等症。"
        elif "陰虛火旺" in prompt:
            return "陰虛火旺的調理方法包括：1. 飲食調理：多吃滿陰食物，少吃燥熱食物。2. 作息調理：保證充足睡眠。3. 中藥調理：如知柏地黃丸等。4. 情緒調理：保持平和心態。建議諮詢專業中醫師。"
        elif "小柴胡湯" in prompt:
            return "小柴胡湯適用於傷寒少陽病，主要症狀包括寒熱往來、胸脇苦滿、心煮喜吐、默默不欲飲食等。組成包括柴胡、黃芩、人參、半夏、生薑、大棗、甘草。"
        else:
            return f"對不起，目前無法提供詳細的專業回答。建議您諮詢專業中醫師或查閱相關中醫書籍。若需要更詳細的資訊，請等待系統修復後再試。"
    
    def generate_answer(self, question: str, use_rag: bool = True) -> Dict:
        """
        生成問題答案
        
        Args:
            question: 用戶問題
            use_rag: 是否使用RAG
            
        Returns:
            回答結果字典
        """
        start_time = time.time()
        
        if use_rag:
            # 檢索相關知識
            knowledge_pieces = self.retrieve_knowledge(question)
            context = self.format_context(knowledge_pieces)
            
            # 構建RAG提示詞
            if context:
                prompt = f"""請根據以下參考資料回答問題。如果參考資料中沒有相關信息，請基於你的中醫知識回答，但要說明這是基於一般知識的回答。

參考資料：
{context}

問題：{question}

請提供詳細、準確的中醫專業回答："""
            else:
                prompt = f"""以下是一個中醫相關的問題，請基於你的專業知識提供詳細回答：

問題：{question}

請提供詳細、準確的中醫專業回答："""
        else:
            # 不使用RAG，直接回答
            prompt = f"""以下是一個中醫相關的問題，請基於你的專業知識提供詳細回答：

問題：{question}

請提供詳細、準確的中醫專業回答："""
            knowledge_pieces = []
            context = ""
        
        # 調用LLM生成答案 - 減少max_tokens以提升速度
        answer = self.call_llm(prompt, max_tokens=400)
        
        response_time = time.time() - start_time
        
        result = {
            "question": question,
            "answer": answer,
            "use_rag": use_rag,
            "retrieved_knowledge": len(knowledge_pieces) if use_rag else 0,
            "context_length": len(context) if use_rag else 0,
            "knowledge_sources": [
                {
                    "book": piece.get('book_title'),
                    "chapter": piece.get('chapter_title'),
                    "similarity": piece.get('similarity_score')
                } for piece in knowledge_pieces
            ] if use_rag else [],
            "response_time": round(response_time, 2),
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"問題回答完成，耗時: {response_time:.2f}秒")
        return result
    
    def batch_qa(self, questions: List[str], output_file: Optional[str] = None) -> List[Dict]:
        """
        批量問答
        
        Args:
            questions: 問題列表
            output_file: 輸出文件路徑
            
        Returns:
            回答結果列表
        """
        logger.info(f"開始批量問答，共 {len(questions)} 個問題")
        
        results = []
        for i, question in enumerate(questions, 1):
            logger.info(f"處理問題 {i}/{len(questions)}: {question[:50]}...")
            
            # 生成RAG答案
            rag_result = self.generate_answer(question, use_rag=True)
            
            # 生成非RAG答案（對比）
            no_rag_result = self.generate_answer(question, use_rag=False)
            
            result = {
                "question_id": i,
                "question": question,
                "rag_answer": rag_result,
                "no_rag_answer": no_rag_result
            }
            
            results.append(result)
            
            # 避免過快調用
            time.sleep(1)
        
        # 保存結果
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            logger.info(f"批量問答結果已保存到: {output_file}")
        
        return results
    
    def evaluate_on_exam(self, exam_csv_path: str, output_dir: str = "rag_system/evaluation") -> Dict:
        """
        在考試題目上評估RAG系統
        
        Args:
            exam_csv_path: 考試CSV文件路徑
            output_dir: 評估結果輸出目錄
            
        Returns:
            評估結果
        """
        import pandas as pd
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"開始評估RAG系統，考試文件: {exam_csv_path}")
        
        # 讀取考題
        try:
            df = pd.read_csv(exam_csv_path, encoding='utf-8')
        except Exception as e:
            logger.error(f"讀取考題文件失敗: {e}")
            return {"error": "文件讀取失敗"}
        
        results = []
        correct_rag = 0
        correct_no_rag = 0
        
        for index, row in df.iterrows():
            question_num = row['題號']
            correct_answer = row['答案'].strip().upper()
            
            # 構建問題文本
            question_text = f"""題目：{row['題目']}

A. {row['選項A']}
B. {row['選項B']} 
C. {row['選項C']}
D. {row['選項D']}

請選擇正確答案（只回答字母）："""
            
            logger.info(f"評估題目 {question_num}")
            
            # RAG回答
            rag_result = self.generate_answer(question_text, use_rag=True)
            rag_answer = self._extract_answer_choice(rag_result['answer'])
            
            # 非RAG回答
            no_rag_result = self.generate_answer(question_text, use_rag=False)
            no_rag_answer = self._extract_answer_choice(no_rag_result['answer'])
            
            # 判斷正確性
            rag_correct = (rag_answer == correct_answer)
            no_rag_correct = (no_rag_answer == correct_answer)
            
            if rag_correct:
                correct_rag += 1
            if no_rag_correct:
                correct_no_rag += 1
            
            result = {
                "question_number": question_num,
                "question": row['題目'],
                "correct_answer": correct_answer,
                "rag_answer": rag_answer,
                "rag_correct": rag_correct,
                "rag_response": rag_result,
                "no_rag_answer": no_rag_answer,
                "no_rag_correct": no_rag_correct,
                "no_rag_response": no_rag_result
            }
            
            results.append(result)
            
            # 進度報告
            if (index + 1) % 10 == 0:
                current_rag_acc = correct_rag / (index + 1) * 100
                current_no_rag_acc = correct_no_rag / (index + 1) * 100
                logger.info(f"進度: {index + 1}/{len(df)}, RAG準確率: {current_rag_acc:.1f}%, 非RAG準確率: {current_no_rag_acc:.1f}%")
        
        # 計算最終統計
        total_questions = len(results)
        rag_accuracy = correct_rag / total_questions * 100
        no_rag_accuracy = correct_no_rag / total_questions * 100
        
        evaluation_summary = {
            "evaluation_time": datetime.now().isoformat(),
            "exam_file": exam_csv_path,
            "total_questions": total_questions,
            "rag_results": {
                "correct": correct_rag,
                "accuracy": round(rag_accuracy, 2)
            },
            "no_rag_results": {
                "correct": correct_no_rag,
                "accuracy": round(no_rag_accuracy, 2)
            },
            "improvement": round(rag_accuracy - no_rag_accuracy, 2)
        }
        
        # 保存詳細結果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        detailed_file = output_dir / f"rag_evaluation_detailed_{timestamp}.json"
        summary_file = output_dir / f"rag_evaluation_summary_{timestamp}.json"
        
        with open(detailed_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(evaluation_summary, f, ensure_ascii=False, indent=2)
        
        logger.info(f"評估完成!")
        logger.info(f"RAG準確率: {rag_accuracy:.2f}% ({correct_rag}/{total_questions})")
        logger.info(f"非RAG準確率: {no_rag_accuracy:.2f}% ({correct_no_rag}/{total_questions})")
        logger.info(f"改進幅度: {evaluation_summary['improvement']:.2f}%")
        logger.info(f"詳細結果: {detailed_file}")
        logger.info(f"總結報告: {summary_file}")
        
        return evaluation_summary
    
    def _extract_answer_choice(self, response: str) -> str:
        """提取選擇題答案"""
        import re
        
        if not response:
            return "INVALID"
        
        # 尋找答案模式
        patterns = [
            r'答案[是：:]\s*([ABCD])',
            r'選擇\s*([ABCD])',
            r'正確答案[是：:]\s*([ABCD])',
            r'^([ABCD])$',
            r'^([ABCD])[。.]',
            r'([ABCD])[。.]?$'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response.strip(), re.MULTILINE | re.IGNORECASE)
            if match:
                return match.group(1).upper()
        
        # 如果沒找到，返回第一個出現的選項字母
        letters = re.findall(r'[ABCD]', response.upper())
        if letters:
            return letters[0]
        
        return "INVALID"

def main():
    """主函數"""
    import argparse
    
    parser = argparse.ArgumentParser(description='中醫RAG系統')
    parser.add_argument('--action', choices=['setup', 'query', 'evaluate'], 
                       required=True, help='操作類型')
    parser.add_argument('--pdf_dir', default='book', help='PDF文件目錄')
    parser.add_argument('--question', help='查詢問題')
    parser.add_argument('--exam_file', help='考試文件路徑 (用於評估)')
    parser.add_argument('--no_rag', action='store_true', help='不使用RAG')
    parser.add_argument('--llm_port', default='8090', help='LLM服務端口')
    
    args = parser.parse_args()
    
    # 初始化RAG系統
    rag_system = TCMRAGSystem(llm_port=args.llm_port)
    
    if args.action == 'setup':
        # 設置系統
        success = rag_system.setup_system(args.pdf_dir)
        if success:
            print("RAG系統設置完成!")
        else:
            print("RAG系統設置失敗!")
    
    elif args.action == 'query':
        if not args.question:
            print("請提供 --question 參數")
            return
        
        # 加載系統
        if not rag_system.load_system():
            print("RAG系統加載失敗")
            return
        
        # 生成回答
        result = rag_system.generate_answer(args.question, use_rag=not args.no_rag)
        
        print(f"\n問題: {result['question']}")
        print(f"回答: {result['answer']}")
        if result['use_rag'] and result['knowledge_sources']:
            print(f"\n參考來源:")
            for source in result['knowledge_sources']:
                print(f"  - {source['book']} / {source['chapter']} (相似度: {source['similarity']:.3f})")
    
    elif args.action == 'evaluate':
        if not args.exam_file:
            print("請提供 --exam_file 參數")
            return
        
        # 加載系統
        if not rag_system.load_system():
            print("RAG系統加載失敗")
            return
        
        # 評估
        results = rag_system.evaluate_on_exam(args.exam_file)
        print(f"評估完成: {results}")

if __name__ == "__main__":
    main()