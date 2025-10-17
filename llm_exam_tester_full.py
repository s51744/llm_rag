import pandas as pd
import requests
import json
import time
from datetime import datetime
import os
import re
import argparse

class LLMExamTester:
    def __init__(self, port="8090", limit=None, delay=0.5, max_context_tokens=500):
        """
        Initialize the LLM Exam Tester

        Args:
            port (str): Port of the deployed LLM service
            limit (int): Maximum number of questions to test per file (None for all)
            delay (float): Delay between API calls in seconds
            max_context_tokens (int|None): (Optional) approximate max context window in tokens.
                This is stored for downstream RAG prompt construction; the code uses an
                approximate chars = tokens * 4 heuristic when truncating injected contexts.
        """
        self.port = port
        self.base_url = f"http://localhost:{port}"
        self.limit = limit
        self.delay = delay
        # Optional limit on context window (tokens). Stored for RAG prompt truncation.
        self.max_context_tokens = max_context_tokens
        self.results = []
        self.total_questions = 0
        self.correct_answers = 0
        
    def call_llm(self, question_text, retries=3):
        """
        Call the LLM using OpenAI-compatible HTTP API
        
        Args:
            question_text (str): The formatted question text
            retries (int): Number of retry attempts
            
        Returns:
            str: LLM response or error message
        """
        # Format the prompt for multiple choice questions
        prompt = f"""請仔細閱讀以下中醫基礎醫學題目，並選出正確答案。請只回答選項字母（A、B、C或D）。

{question_text}

請選擇正確答案："""
        
        headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        }
        
        # OpenAI-compatible chat completions API format
        data = {
            "model": "BianCang-Qwen2.5-7B-Instruct",
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_tokens": 128,
            "temperature": 0.1,  # Low temperature for consistent answers
            "top_p": 0.9,
            "stream": False
        }
        
        for attempt in range(retries):
            try:
                response = requests.post(
                    f"{self.base_url}/v1/chat/completions",
                    headers=headers,
                    json=data,
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    if 'choices' in result and len(result['choices']) > 0:
                        content = result['choices'][0]['message']['content']
                        return content.strip()
                    else:
                        return "INVALID_RESPONSE"
                else:
                    if attempt == 0:  # Only print error on first attempt
                        print(f"API Error {response.status_code}")
                    
            except requests.exceptions.RequestException as e:
                if attempt == 0:  # Only print error on first attempt
                    print(f"Request failed: {str(e)[:50]}...")
                if attempt < retries - 1:
                    time.sleep(2)  # Wait before retry
                    
        return "API_ERROR"
    
    def extract_answer_from_response(self, response):
        """
        Extract the answer choice (A, B, C, or D) from LLM response
        
        Args:
            response (str): Raw LLM response
            
        Returns:
            str: Extracted answer or 'INVALID'
        """
        if not response or response in ["API_ERROR", "MODEL_ERROR", "INVALID_RESPONSE"]:
            return response
        
        # Look for standalone A, B, C, or D patterns
        patterns = [
            r'^[ABCD]$',  # Just the letter
            r'^答案[：:]?\s*([ABCD])',  # 答案：A or 答案:A
            r'^選擇[：:]?\s*([ABCD])',  # 選擇：A
            r'答案是[：:]?\s*([ABCD])',  # 答案是：A
            r'選擇([ABCD])',  # 選擇A
            r'選項([ABCD])',  # 選項A
            r'([ABCD])[。.]?$',  # A. at the end
            r'正確答案[是為][：:]?\s*([ABCD])',  # 正確答案是：A
        ]
        
        # First try to find exact matches
        for pattern in patterns:
            match = re.search(pattern, response.strip(), re.MULTILINE)
            if match:
                if match.groups():
                    return match.group(1)
                else:
                    return match.group(0)
        
        # If no pattern matches, look for any A, B, C, D in the response
        letters = re.findall(r'[ABCD]', response)
        if letters:
            return letters[0]  # Return the first found letter
            
        return "INVALID"
    
    def format_question(self, row):
        """
        Format a question row into readable text
        
        Args:
            row: DataFrame row containing question data
            
        Returns:
            str: Formatted question text
        """
        question_num = row['題號']
        question_text = row['題目']
        option_a = row['選項A']
        option_b = row['選項B']
        option_c = row['選項C']
        option_d = row['選項D']
        
        formatted = f"""題目 {question_num}：{question_text}

A. {option_a}
B. {option_b}  
C. {option_c}
D. {option_d}"""
        
        return formatted
    
    def process_csv_file(self, csv_path):
        """
        Process a single CSV file
        
        Args:
            csv_path (str): Path to CSV file
            
        Returns:
            list: Results from processing this file
        """
        file_basename = os.path.basename(csv_path)
        limit_text = f" (限制 {self.limit} 題)" if self.limit else ""
        print(f"\n處理文件: {file_basename}{limit_text}")
        
        try:
            # Read CSV file
            df = pd.read_csv(csv_path, encoding='utf-8')
            
            # Limit the number of questions if specified
            if self.limit:
                df = df.head(self.limit)
            
            file_results = []
            total_in_file = len(df)
            
            print(f"開始處理 {total_in_file} 題...")
            
            for index, row in df.iterrows():
                self.total_questions += 1
                question_num = row['題號']
                correct_answer = row['答案'].strip().upper()
                
                # Show progress
                progress = f"[{index + 1}/{total_in_file}]"
                print(f"{progress} 題目 {question_num}...", end=' ')
                
                # Format question for LLM
                question_text = self.format_question(row)
                
                # Get LLM response
                start_time = time.time()
                llm_response = self.call_llm(question_text)
                response_time = time.time() - start_time
                
                # Extract answer from response
                llm_answer = self.extract_answer_from_response(llm_response)
                
                # Check if correct
                is_correct = llm_answer == correct_answer
                if is_correct:
                    self.correct_answers += 1
                    print("✓")
                else:
                    status_text = f"✗ ({correct_answer}→{llm_answer})"
                    print(status_text)
                
                # Store result
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
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                
                file_results.append(result)
                self.results.append(result)
                
                # Delay to avoid overwhelming the API
                if self.delay > 0:
                    time.sleep(self.delay)
                
            print(f"完成處理文件 {file_basename}")
                
        except Exception as e:
            print(f"處理文件 {csv_path} 時出錯: {e}")
            
        return file_results
    
    def calculate_statistics(self):
        """
        Calculate comprehensive accuracy statistics
        
        Returns:
            dict: Statistics dictionary
        """
        if self.total_questions == 0:
            return {}
        
        accuracy = (self.correct_answers / self.total_questions) * 100
        
        # Count by LLM answer choice
        llm_answer_counts = {'A': 0, 'B': 0, 'C': 0, 'D': 0, 'API_ERROR': 0, 'INVALID_RESPONSE': 0, 'INVALID': 0}
        
        # Count by correct answer choice
        correct_answer_counts = {'A': 0, 'B': 0, 'C': 0, 'D': 0}
        correct_by_choice = {'A': 0, 'B': 0, 'C': 0, 'D': 0}
        
        for result in self.results:
            llm_ans = result['llm_answer']
            correct_ans = result['correct_answer']
            
            if llm_ans in llm_answer_counts:
                llm_answer_counts[llm_ans] += 1
            
            if correct_ans in correct_answer_counts:
                correct_answer_counts[correct_ans] += 1
                if result['is_correct']:
                    correct_by_choice[correct_ans] += 1
        
        # Calculate accuracy by correct answer choice
        accuracy_by_choice = {}
        for choice in ['A', 'B', 'C', 'D']:
            if correct_answer_counts[choice] > 0:
                accuracy_by_choice[choice] = (correct_by_choice[choice] / correct_answer_counts[choice]) * 100
            else:
                accuracy_by_choice[choice] = 0
        
        # Calculate file-wise statistics
        file_stats = {}
        for result in self.results:
            file_name = result['file_name']
            if file_name not in file_stats:
                file_stats[file_name] = {'total': 0, 'correct': 0}
            file_stats[file_name]['total'] += 1
            if result['is_correct']:
                file_stats[file_name]['correct'] += 1
        
        # Calculate average response time
        response_times = [r['response_time_seconds'] for r in self.results]
        avg_response_time = sum(response_times) / len(response_times) if response_times else 0
        
        return {
            'total_questions': self.total_questions,
            'correct_answers': self.correct_answers,
            'overall_accuracy': round(accuracy, 2),
            'llm_answer_distribution': llm_answer_counts,
            'correct_answer_distribution': correct_answer_counts,
            'accuracy_by_correct_choice': {k: round(v, 2) for k, v in accuracy_by_choice.items()},
            'file_statistics': {k: {'total': v['total'], 'correct': v['correct'], 'accuracy': round((v['correct']/v['total'])*100, 2)} for k, v in file_stats.items()},
            'api_errors': llm_answer_counts['API_ERROR'],
            'invalid_responses': llm_answer_counts['INVALID_RESPONSE'] + llm_answer_counts['INVALID'],
            'average_response_time': round(avg_response_time, 2)
        }
    
    def print_statistics(self):
        """Print formatted statistics to console"""
        stats = self.calculate_statistics()
        
        if not stats:
            print("沒有處理任何題目")
            return
        
        print("\n" + "="*80)
        print("LLM 中醫基礎醫學考題測驗結果統計")
        print("="*80)
        print(f"總題數: {stats['total_questions']}")
        print(f"答對題數: {stats['correct_answers']}")
        print(f"整體準確率: {stats['overall_accuracy']}%")
        print(f"平均回應時間: {stats['average_response_time']} 秒")
        print(f"API 錯誤數: {stats['api_errors']}")
        print(f"無效回應數: {stats['invalid_responses']}")
        
        print(f"\n文件別統計:")
        for file_name, file_stat in stats['file_statistics'].items():
            print(f"  {file_name}: {file_stat['correct']}/{file_stat['total']} ({file_stat['accuracy']}%)")
        
        print(f"\nLLM 回答分布:")
        for choice, count in stats['llm_answer_distribution'].items():
            if count > 0:
                percentage = (count / stats['total_questions']) * 100
                print(f"  {choice}: {count} 次 ({percentage:.1f}%)")
        
        print(f"\n正確答案分布:")
        for choice, count in stats['correct_answer_distribution'].items():
            if count > 0:
                percentage = (count / stats['total_questions']) * 100
                print(f"  {choice}: {count} 題 ({percentage:.1f}%)")
        
        print(f"\n按正確答案分類的LLM準確率:")
        for choice, accuracy in stats['accuracy_by_correct_choice'].items():
            if stats['correct_answer_distribution'][choice] > 0:
                print(f"  正確答案為 {choice} 的題目: {accuracy}%")
        
        print("="*80)
    
    def save_results_to_csv(self, output_path):
        """
        Save detailed results to CSV file
        
        Args:
            output_path (str): Path for output CSV file
        """
        if not self.results:
            print("沒有結果可保存")
            return
            
        df_results = pd.DataFrame(self.results)
        df_results.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"詳細結果已保存到: {output_path}")
    
    def save_statistics_to_txt(self, output_path):
        """
        Save statistics summary to text file
        
        Args:
            output_path (str): Path for output text file
        """
        stats = self.calculate_statistics()
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("LLM 中醫基礎醫學考題測驗結果統計\n")
            f.write("="*80 + "\n")
            f.write(f"測驗時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"總題數: {stats['total_questions']}\n")
            f.write(f"答對題數: {stats['correct_answers']}\n")
            f.write(f"整體準確率: {stats['overall_accuracy']}%\n")
            f.write(f"平均回應時間: {stats['average_response_time']} 秒\n")
            f.write(f"API 錯誤數: {stats['api_errors']}\n")
            f.write(f"無效回應數: {stats['invalid_responses']}\n\n")
            
            f.write("文件別統計:\n")
            for file_name, file_stat in stats['file_statistics'].items():
                f.write(f"  {file_name}: {file_stat['correct']}/{file_stat['total']} ({file_stat['accuracy']}%)\n")
            
            f.write("\nLLM 回答分布:\n")
            for choice, count in stats['llm_answer_distribution'].items():
                if count > 0:
                    percentage = (count / stats['total_questions']) * 100
                    f.write(f"  {choice}: {count} 次 ({percentage:.1f}%)\n")
            
            f.write("\n正確答案分布:\n")
            for choice, count in stats['correct_answer_distribution'].items():
                if count > 0:
                    percentage = (count / stats['total_questions']) * 100
                    f.write(f"  {choice}: {count} 題 ({percentage:.1f}%)\n")
            
            f.write("\n按正確答案分類的LLM準確率:\n")
            for choice, accuracy in stats['accuracy_by_correct_choice'].items():
                if stats['correct_answer_distribution'][choice] > 0:
                    f.write(f"  正確答案為 {choice} 的題目: {accuracy}%\n")
        
        print(f"統計摘要已保存到: {output_path}")

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='LLM 中醫基礎醫學考題測驗系統')
    parser.add_argument('--limit', type=int, default=None, help='每個文件最多處理的題目數量 (預設: 全部)')
    parser.add_argument('--delay', type=float, default=0.5, help='API 調用間隔時間(秒) (預設: 0.5)')
    parser.add_argument('--port', type=str, default='8090', help='LLM 服務埠號 (預設: 8090)')
    parser.add_argument('--max-context-tokens', type=int, default=None, help='(選用) 模型上下文視窗大小（tokens），用於 RAG prompt 截斷')
    args = parser.parse_args()
    
    # File paths
    base_dir = "C:\\Users\\Personal\\Downloads\\BianCang-main\\BianCang-main"
    csv_files = [
        f"{base_dir}\\llm_exam\\中醫基礎醫學_106年考題(一).csv",
        f"{base_dir}\\llm_exam\\中醫基礎醫學_106年考題(二).csv",
        f"{base_dir}\\llm_exam\\中醫基礎醫學_112年考題(一).csv",
        f"{base_dir}\\llm_exam\\中醫基礎醫學_112年考題(二).csv"
    ]
    
    # Output file paths
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"{base_dir}\\llm_exam_results"
    output_csv = f"{results_dir}\\llm_exam_results_full_{timestamp}.csv"
    output_txt = f"{results_dir}\\llm_exam_statistics_{timestamp}.txt"
    
    # Initialize tester
    tester = LLMExamTester(port=args.port, limit=args.limit, delay=args.delay, max_context_tokens=args.max_context_tokens)
    
    print("LLM 中醫基礎醫學考題測驗系統")
    print("="*50)
    print(f"LLM API URL: {tester.base_url}")
    if args.limit:
        print(f"題目限制: 每個文件最多 {args.limit} 題")
    else:
        print(f"題目限制: 處理全部題目")
    print(f"API 調用間隔: {args.delay} 秒")
    
    # Test API connection
    try:
        test_response = requests.get(f"{tester.base_url}", timeout=5)
        print("✓ LLM API 連接正常")
    except:
        print("✗ 警告: 無法連接到 LLM API，請確認服務正在運行")
        print("請確認您的部署命令正在運行中...")
        return
    
    # Process CSV files
    processed_files = 0
    for csv_file in csv_files:
        if os.path.exists(csv_file):
            tester.process_csv_file(csv_file)
            processed_files += 1
        else:
            print(f"警告: 找不到文件 {csv_file}")
    
    if processed_files == 0:
        print("錯誤: 沒有找到任何有效的 CSV 文件")
        return
    
    # Print statistics
    tester.print_statistics()
    
    # Save detailed results and statistics
    tester.save_results_to_csv(output_csv)
    tester.save_statistics_to_txt(output_txt)
    
    print(f"\n測驗完成！")
    print(f"處理了 {processed_files} 個文件，共 {tester.total_questions} 題")
    print(f"最終準確率: {(tester.correct_answers / tester.total_questions * 100):.2f}%")
    
if __name__ == "__main__":
    main()