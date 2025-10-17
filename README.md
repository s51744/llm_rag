# 🏥 扁倉中醫RAG系統

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![RAG](https://img.shields.io/badge/RAG-Optimized-orange.svg)]()

一個專為中醫知識問答設計的高性能RAG（檢索增強生成）系統，結合了扁倉Qwen2.5-7B中醫大模型和中醫教科書知識庫，提供準確、有依據的中醫專業回答。

## 🌟 主要特色

- 🔍 **智能檢索**: 基於向量相似度的中醫知識檢索
- ⚡ **高性能快取**: 雙層快取機制，顯著提升查詢速度
- 🛡️ **故障回退**: 當LLM不可用時提供專業的備用回答
- 📚 **豐富知識庫**: 包含7本經典中醫教科書
- 🧵 **線程安全**: 支持併發訪問和批量處理
- 📊 **性能監控**: 內建性能測試和評估工具

## 📁 項目結構

```
BianCang-main/
├── 📚 book/                              # 中醫教科書PDF文件
│   ├── 中國醫學史.pdf
│   ├── 中醫藥物學講義(中國醫長庚合編).pdf
│   ├── 傷寒論承啟版(知音出版社).pdf
│   ├── 內經-知音出版(程士德).pdf
│   ├── 方劑學-唐娜櫻(中國醫_長庚).pdf
│   ├── 方劑學承啟版-許濟群(知音出版社).pdf
│   └── 難經新解.pdf
├── 🔧 rag_system/                        # RAG系統核心模組
│   ├── 📄 data/                          # 處理後的文本數據
│   │   ├── processing_summary.json       # 處理統計報告
│   │   └── *_processed.json             # 各書籍處理後的結構化數據
│   ├── 🗄️ vectordb/                      # 向量數據庫
│   │   ├── tcm_knowledge.index          # FAISS向量索引
│   │   ├── metadata.json                # 文檔元數據
│   │   └── embeddings.pkl               # 向量嵌入數據
│   ├── pdf_processor.py                 # PDF處理模組
│   ├── vector_store.py                  # 向量數據庫管理
│   └── tcm_rag.py                       # RAG系統核心邏輯
├── 📝 llm_exam/                          # 中醫考試題目
│   ├── 中醫基礎醫學_106年考題(一).csv
│   ├── 中醫基礎醫學_106年考題(二).csv
│   ├── 中醫基礎醫學_112年考題(一).csv
│   └── 中醫基礎醫學_112年考題(二).csv
├── 🧪 llm_exam_results/                  # 考試評估結果
├── 📋 templates/                         # 提示詞模板
│   ├── eval/prompt.md                   # 評估提示詞
│   └── train/Medical_Record_SFT_TEMPLATE.md
├── 🖼️ assets/                            # 項目資源文件
├── 🎯 ChP-TCM/                           # 中醫藥典知識問答數據
├── 🚀 setup_rag.py                       # 一鍵安裝腳本
├── 🌐 web_ui.py                          # Streamlit網頁界面
├── ⚡ swift_deploy_ultrafast.py          # Swift部署腳本
├── 🧪 test_rag_performance.py            # 性能測試工具
├── 📊 llm_exam_tester_full.py            # 完整考試測試器
├── 📋 llm_exam_tester_limited.py         # 限制考試測試器
└── 📖 README.md                          # 本文檔
```

## 🚀 快速開始

### 1. 環境準備

```bash
# 確保Python 3.8+環境
python --version

# 安裝必要依賴
pip install PyPDF2 pdfplumber faiss-cpu sentence-transformers torch numpy pandas requests streamlit
```

### 2. 啟動LLM服務

```bash
# 啟動扁倉中醫大模型服務（端口8090）
python swift_deploy_ultrafast.py \
    --model QLUNLP/BianCang-Qwen2.5-7B-Instruct \
    --ckpt_dir outputs/identity-lora/qwen2_5-7b-instruct/v1-20250917-215539/checkpoint-1 \
    --template qwen \
    --infer_backend pt \
    --port 8090 \
    --max_new_tokens 256 \
    --device_map auto \
    --torch_dtype float16 \
    --gpu_memory_utilization 0.9
```

### 3. 一鍵設置RAG系統

```bash
# 自動設置整個RAG系統
python setup_rag.py
```

這個腳本會自動完成：
- ✅ 檢查並安裝必要依賴
- ✅ 處理PDF教科書提取文本（約5-10分鐘）
- ✅ 構建向量索引（約3-5分鐘）
- ✅ 運行測試查詢驗證系統
- ✅ （可選）運行完整評估

## 📖 使用方法

### 🔍 基本查詢

```bash
# 單個問題查詢（使用RAG）
python rag_system/tcm_rag.py --action query --question "什麼是氣血兩虛？"

# 不使用RAG的對比查詢
python rag_system/tcm_rag.py --action query --question "什麼是氣血兩虛？" --no_rag
```

### 📊 系統評估

```bash
# 在考試題目上評估RAG效果
python rag_system/tcm_rag.py --action evaluate --exam_file "llm_exam/中醫基礎醫學_106年考題(一).csv"
```

### 🌐 網頁界面

```bash
# 啟動Streamlit網頁界面
streamlit run web_ui.py
```

### 🧪 性能測試

```bash
# 運行性能測試和快取效果驗證
python test_rag_performance.py
```

## ⚙️ 系統配置

### RAG核心參數

在 `rag_system/tcm_rag.py` 中可調整的關鍵參數：

```python
class TCMRAGSystem:
    def __init__(self):
        # 檢索參數 - 已優化
        self.max_context_length = 1500     # 最大上下文長度
        self.retrieval_top_k = 2           # 檢索結果數量
        self.similarity_threshold = 0.5     # 相似度閾值
        
        # 快取參數
        self.max_cache_size = 100          # 最大快取條目
        
        # LLM參數
        self.llm_timeout = 30              # API調用超時（秒）
        self.max_tokens = 400              # 最大生成長度
```

### 嵌入模型配置

系統支持多種中文嵌入模型：

- **主要模型**: `BAAI/bge-base-zh-v1.5` (768維)
- **備用模型**: `paraphrase-multilingual-MiniLM-L12-v2` (384維)

## 🏗️ 架構設計

### 🔄 RAG工作流程

```
用戶查詢 → 快取檢查 → 向量檢索 → 知識整合 → LLM生成 → 快取存儲 → 回答輸出
    ↓           ↓           ↓           ↓           ↓           ↓
  快取命中 ← 實時回答  ← 相關文檔  ← 上下文構建 ← 專業回答  ← 性能優化
```

### 💾 快取機制

- **雙層快取**: 檢索快取 + LLM回答快取
- **LRU策略**: 最近最少使用替換算法
- **線程安全**: 使用鎖機制保證併發安全
- **智能清理**: 自動管理快取容量

### 🛡️ 故障回退

當LLM服務不可用時，系統提供以下備用回答：
- 氣血兩虛、脾胃虛弱、陰虛火旺等常見證候
- 六君子湯、小柴胡湯等經典方劑
- 專業的中醫調理建議

## 📊 性能表現

### 系統統計
- 📚 **知識庫規模**: 7本中醫教科書，3,304個知識片段
- ⚡ **查詢速度**: 快取命中時 < 0.1秒
- 🎯 **檢索準確度**: 相似度閾值0.5，平衡準確性與速度
- 💾 **快取效率**: 支持100個常用查詢的快速響應

### 測試結果示例
```
📋 測試問題: 什麼是氣血兩虛？
⏱️  響應時間: 32.16秒 (含LLM調用)
📖 檢索到知識: 2個片段
📝 上下文長度: 1076字符
💬 專業回答: 氣血兩虛是中醫學中的一種病理狀態...
📚 知識來源: 方劑學承啟版-許濟群(知音出版社)
```

## 🔧 進階功能

### 批量問答

```python
from rag_system.tcm_rag import TCMRAGSystem

# 初始化系統
rag = TCMRAGSystem()
rag.load_system()

# 批量處理問題
questions = [
    "脾胃虛弱的症狀有哪些？",
    "六君子湯的組成是什麼？",
    "如何調理陰虛火旺？"
]

results = rag.batch_qa(questions, "batch_results.json")
```

### 自定義檢索

```python
# 調整檢索參數
results = rag.vector_store.search_with_content(
    query="您的查詢",
    top_k=5,              # 更多結果
    score_threshold=0.3   # 更低閾值
)
```

### 系統統計

```bash
# 查看向量數據庫統計信息
python rag_system/vector_store.py --action stats --store_dir rag_system/vectordb
```

## 🐛 常見問題

### Q: PDF處理失敗怎麼辦？
**A**: 
1. 確認PDF文件完整且可讀
2. 檢查中文編碼支持
3. 嘗試不同的提取方法：`--method pypdf2`

### Q: LLM調用超時？
**A**:
1. 檢查LLM服務是否運行：`http://localhost:8090/v1/models`
2. 確認模型名稱正確：`BianCang-Qwen2.5-7B-Instruct`
3. 系統會自動使用故障回退機制

### Q: 記憶體不足？
**A**:
1. 減少`batch_size`參數
2. 分批處理PDF文件
3. 使用更小的嵌入模型

### Q: 向量化速度慢？
**A**:
1. 首次下載模型需要時間
2. 後續查詢會使用快取加速
3. 考慮使用GPU加速

## 🎯 評估結果

### 評估文件說明

**詳細結果** (`rag_evaluation_detailed_*.json`):
- 每個題目的完整回答
- RAG vs 非RAG準確性對比
- 檢索到的參考資料信息

**總結報告** (`rag_evaluation_summary_*.json`):
- 整體準確率統計
- RAG改進幅度分析
- 性能指標摘要

## 🔬 技術棧

- **向量數據庫**: FAISS
- **嵌入模型**: BGE-base-zh-v1.5
- **PDF處理**: PyPDF2, pdfplumber
- **Web框架**: Streamlit
- **深度學習**: PyTorch, Transformers
- **大語言模型**: 扁倉Qwen2.5-7B

## 📄 License

本項目採用MIT License開源協議。

## ⚠️ 免責聲明

本系統生成的中醫內容僅供學術研究和教育用途，不能替代專業醫療診斷與治療。使用者應當諮詢合格的中醫師獲取專業醫療建議。

## 🤝 貢獻

歡迎提交Issue和Pull Request來改進本項目！

---

**開發團隊**: 宜蘭大學 MIT LAB  
**技術支持**: 如有問題請查看日志文件或提交Issue