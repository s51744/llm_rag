#!/usr/bin/env python3
"""
中醫RAG系統快速設置腳本

這個腳本會自動完成以下步驟:
1. 檢查並安裝必要的依賴包
2. 處理PDF教科書
3. 構建向量索引
4. 運行測試查詢
5. 評估RAG效果
"""

import subprocess
import sys
import os
from pathlib import Path
import time

def install_package(package):
    """安裝Python包"""
    print(f"正在安裝 {package}...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✓ {package} 安裝成功")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {package} 安裝失敗: {e}")
        return False

def check_and_install_dependencies():
    """檢查並安裝依賴包"""
    print("檢查依賴包...")
    
    required_packages = [
        "PyPDF2",
        "pdfplumber", 
        "faiss-cpu",
        "sentence-transformers",
        "torch",
        "numpy",
        "pandas",
        "requests"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
            print(f"✓ {package} 已安裝")
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"需要安裝以下包: {', '.join(missing_packages)}")
        
        for package in missing_packages:
            if not install_package(package):
                print(f"依賴包 {package} 安裝失敗，請手動安裝")
                return False
    
    print("✓ 所有依賴包已安裝")
    return True

def run_rag_setup():
    """運行RAG系統設置"""
    print("\n" + "="*60)
    print("開始設置中醫RAG系統")
    print("="*60)
    
    # 切換到rag_system目錄
    os.chdir("rag_system")
    
    try:
        # 運行RAG設置
        print("正在處理PDF文件和構建向量索引...")
        result = subprocess.run([
            sys.executable, "tcm_rag.py", 
            "--action", "setup",
            "--pdf_dir", "../book"
        ], capture_output=True, text=True, encoding='utf-8')
        
        if result.returncode == 0:
            print("✓ RAG系統設置成功!")
            print(result.stdout)
        else:
            print("✗ RAG系統設置失敗:")
            print(result.stderr)
            return False
    
    except Exception as e:
        print(f"✗ 運行設置時出錯: {e}")
        return False
    
    finally:
        # 切換回原目錄
        os.chdir("..")
    
    return True

def run_test_query():
    """運行測試查詢"""
    print("\n" + "="*60)
    print("測試RAG查詢功能")
    print("="*60)
    
    test_questions = [
        "脾胃虛弱的症狀有哪些？",
        "六君子湯的組成和功效是什麼？",
        "什麼是氣血兩虛？如何調理？",
        "傷寒論中的小柴胡湯適用於什麼證候？"
    ]
    
    os.chdir("rag_system")
    
    try:
        for i, question in enumerate(test_questions, 1):
            print(f"\n測試查詢 {i}: {question}")
            print("-" * 40)
            
            # 運行查詢
            result = subprocess.run([
                sys.executable, "tcm_rag.py",
                "--action", "query",
                "--question", question
            ], capture_output=True, text=True, encoding='utf-8')
            
            if result.returncode == 0:
                print(result.stdout)
            else:
                print(f"查詢失敗: {result.stderr}")
                
            time.sleep(2)  # 避免過快調用
    
    except Exception as e:
        print(f"測試查詢時出錯: {e}")
    
    finally:
        os.chdir("..")

def run_evaluation():
    """運行評估"""
    print("\n" + "="*60)
    print("評估RAG系統效果")
    print("="*60)
    
    exam_files = [
        "llm_exam/中醫基礎醫學_106年考題(一).csv",
        "llm_exam/中醫基礎醫學_106年考題(二).csv"
    ]
    
    os.chdir("rag_system")
    
    try:
        for exam_file in exam_files:
            if Path(f"../{exam_file}").exists():
                print(f"\n評估文件: {exam_file}")
                print("-" * 40)
                
                result = subprocess.run([
                    sys.executable, "tcm_rag.py",
                    "--action", "evaluate", 
                    "--exam_file", f"../{exam_file}"
                ], capture_output=True, text=True, encoding='utf-8')
                
                if result.returncode == 0:
                    print(result.stdout)
                else:
                    print(f"評估失敗: {result.stderr}")
                
                time.sleep(5)  # 評估之間的間隔
            else:
                print(f"考試文件不存在: {exam_file}")
    
    except Exception as e:
        print(f"評估時出錯: {e}")
    
    finally:
        os.chdir("..")

def check_llm_service():
    """檢查LLM服務是否運行"""
    print("檢查LLM服務狀態...")
    
    import requests
    try:
        response = requests.get("http://localhost:8090", timeout=5)
        print("✓ LLM服務正在運行")
        return True
    except:
        print("✗ LLM服務未運行")
        print("請先啟動LLM服務:")
        print("python swift_deploy_ultrafast.py --model QLUNLP/BianCang-Qwen2.5-7B-Instruct --ckpt_dir outputs/identity-lora/qwen2_5-7b-instruct/v1-20250917-215539/checkpoint-1 --template qwen --infer_backend pt --port 8090 --max_new_tokens 256 --device_map auto --torch_dtype float16 --gpu_memory_utilization 0.9")
        return False

def main():
    print("="*80)
    print("中醫RAG系統自動設置工具")
    print("="*80)
    
    # 檢查當前目錄
    if not Path("book").exists():
        print("✗ 未找到 'book' 目錄，請確保PDF文件放在book文件夾中")
        return
    
    if not Path("rag_system").exists():
        print("✗ 未找到 'rag_system' 目錄，請確保腳本在正確的位置運行")
        return
    
    # 步驟1: 檢查LLM服務
    if not check_llm_service():
        print("請先啟動LLM服務後再運行此腳本")
        return
    
    # 步驟2: 安裝依賴
    if not check_and_install_dependencies():
        print("依賴安裝失敗，請手動解決依賴問題")
        return
    
    # 步驟3: 設置RAG系統
    if not run_rag_setup():
        print("RAG系統設置失敗")
        return
    
    # 步驟4: 測試查詢
    run_test_query()
    
    # 步驟5: 評估系統
    print("\n是否要運行完整評估？(這將需要較長時間)")
    user_input = input("輸入 'y' 繼續評估，其他鍵跳過: ").lower().strip()
    
    if user_input == 'y':
        run_evaluation()
    
    print("\n" + "="*80)
    print("RAG系統設置完成！")
    print("="*80)
    print("你現在可以:")
    print("1. 使用 rag_system/tcm_rag.py --action query --question '你的問題' 進行查詢")
    print("2. 檢查 rag_system/vectordb/ 目錄查看向量數據庫文件")
    print("3. 檢查 rag_system/data/ 目錄查看處理後的文本數據")
    print("4. 檢查 rag_system/evaluation/ 目錄查看評估結果")
    print("="*80)

if __name__ == "__main__":
    main()