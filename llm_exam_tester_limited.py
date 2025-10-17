import pandas as pd
import requests
import json
import time
from datetime import datetime
import os
import re

class LLMExamTesterLimited:
    def __init__(self, port="8090", limit=5):
        """
        Initialize the LLM Exam Tester - Limited version for testing
        
        Args:
            port (str): Port of the deployed LLM service
            limit (int): Maximum number of questions to test
        """
        self.port = port
        self.base_url = f"http://localhost:{port}"
        self.limit = limit
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
            "max_tokens": 64,
            "temperature": 0.1,
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
                    print(f"API Error {response.status_code}: {response.text}")
                    
            except requests.exceptions.RequestException as e:
                print(f"Request failed (attempt {attempt + 1}): {e}")
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
        
        # Look for standalone A, B, C, or D at the beginning or end
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
        Process a single CSV file (limited number of questions)
        
        Args:
            csv_path (str): Path to CSV file
            
        Returns:
            list: Results from processing this file
        """
        print(f"\n處理文件: {os.path.basename(csv_path)} (限制 {self.limit} 題)")
        
        try:
            # Read CSV file
            df = pd.read_csv(csv_path, encoding='utf-8')
            
            # Limit the number of questions
            df = df.head(self.limit)
            
            file_results = []
            
            for index, row in df.iterrows():
                self.total_questions += 1
                question_num = row['題號']
                correct_answer = row['答案'].strip().upper()
                
                print(f"處理題目 {question_num}...", end=' ')
                
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
                    print(f"✗ (正確答案: {correct_answer}, LLM回答: {llm_answer})")
                
                # Store result
                result = {
                    'file_name': os.path.basename(csv_path),
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
                
                # Small delay to avoid overwhelming the API
                time.sleep(0.5)
                
        except Exception as e:
            print(f"處理文件 {csv_path} 時出錯: {e}")
            
        return file_results
    
    def print_statistics(self):
        """Print formatted statistics to console"""
        if self.total_questions == 0:
            print("沒有處理任何題目")
            return
            
        accuracy = (self.correct_answers / self.total_questions) * 100
        
        print("\n" + "="*60)
        print("LLM 測驗結果統計 (限制版)")
        print("="*60)
        print(f"總題數: {self.total_questions}")
        print(f"答對題數: {self.correct_answers}")
        print(f"整體準確率: {accuracy:.2f}%")
        
        # Count answer distribution
        answer_counts = {'A': 0, 'B': 0, 'C': 0, 'D': 0, 'API_ERROR': 0, 'INVALID': 0}
        for result in self.results:
            llm_ans = result['llm_answer']
            if llm_ans in answer_counts:
                answer_counts[llm_ans] += 1
        
        print(f"\nLLM 回答分布:")
        for choice, count in answer_counts.items():
            percentage = (count / self.total_questions) * 100 if self.total_questions > 0 else 0
            print(f"  {choice}: {count} 次 ({percentage:.1f}%)")
        
        print("="*60)
    
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

def main():
    """Main execution function"""
    # File paths
    base_dir = "C:\\Users\\Personal\\Downloads\\BianCang-main\\BianCang-main"
    csv_file1 = f"{base_dir}\\llm_exam\\中醫基礎醫學_106年考題(一).csv"
    
    # Output file path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = "C:\\Users\\Personal\\Downloads\\BianCang-main\\BianCang-main\\llm_exam_results"
    output_csv = f"{results_dir}\\llm_exam_results_limited_{timestamp}.csv"
    
    # Initialize tester with limit
    tester = LLMExamTesterLimited(limit=5)
    
    print("開始測試 LLM 中醫基礎醫學考題 (限制版)...")
    print(f"LLM API URL: {tester.base_url}")
    
    # Test API connection
    try:
        test_response = requests.get(f"{tester.base_url}", timeout=5)
        print("✓ LLM API 連接正常")
    except:
        print("✗ 警告: 無法連接到 LLM API，請確認服務正在運行")
        print("請確認您的部署命令正在運行中...")
        return
    
    # Process first CSV file only
    if os.path.exists(csv_file1):
        tester.process_csv_file(csv_file1)
    else:
        print(f"警告: 找不到文件 {csv_file1}")
        return
    
    # Print statistics
    tester.print_statistics()
    
    # Save detailed results
    tester.save_results_to_csv(output_csv)
    
    print(f"\n限制版測試完成！")
    
if __name__ == "__main__":
    main()