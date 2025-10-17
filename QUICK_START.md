# 🚀 快速開始指南

## 5分鐘快速部署扁倉中醫RAG系統

### 1️⃣ 環境檢查 (30秒)
```bash
# 檢查Python版本（需要3.8+）
python --version

# 檢查GPU（可選，用於加速）
nvidia-smi
```

### 2️⃣ 安裝依賴 (2分鐘)
```bash
# 方法一：使用requirements.txt
pip install -r requirements.txt

# 方法二：手動安裝核心包
pip install PyPDF2 pdfplumber faiss-cpu sentence-transformers torch requests streamlit
```

### 3️⃣ 啟動LLM服務 (1分鐘)
```bash
# 啟動扁倉中醫大模型（後台運行）
python swift_deploy_ultrafast.py \
    --model QLUNLP/BianCang-Qwen2.5-7B-Instruct \
    --port 8090 &
    
# 檢查服務狀態
curl http://localhost:8090/v1/models
```

### 4️⃣ 一鍵設置RAG系統 (2分鐘)
```bash
# 自動處理PDF、構建索引、測試系統
python setup_rag.py
```

### 5️⃣ 開始使用！ (0秒)
```bash
# 命令行查詢
python rag_system/tcm_rag.py --action query --question "什麼是氣血兩虛？"

# 網頁界面
streamlit run web_ui.py
```

## 🎯 核心功能演示

### 基礎查詢
```bash
# 中醫證候查詢
python rag_system/tcm_rag.py --action query --question "脾胃虛弱的症狀有哪些？"

# 方劑查詢  
python rag_system/tcm_rag.py --action query --question "六君子湯的組成和功效"

# 治療方法查詢
python rag_system/tcm_rag.py --action query --question "如何調理陰虛火旺？"
```

### 系統評估
```bash
# 在考試題目上測試準確率
python rag_system/tcm_rag.py --action evaluate --exam_file "llm_exam/中醫基礎醫學_106年考題(一).csv"
```

### 性能測試
```bash
# 測試查詢速度和快取效果
python test_rag_performance.py
```

## ⚙️ 重要參數說明

| 參數 | 默認值 | 說明 | 調優建議 |
|------|--------|------|----------|
| `similarity_threshold` | 0.5 | 相似度閾值 | 0.3-0.7，越高越嚴格 |
| `retrieval_top_k` | 2 | 檢索結果數量 | 1-5，越多越全面但越慢 |
| `max_context_length` | 1500 | 上下文長度 | 1000-3000字符 |
| `max_cache_size` | 100 | 快取容量 | 根據記憶體調整 |

## 🔧 故障排除

### 常見問題快速解決

**問題**: PDF處理失敗
```bash
# 解決方案：檢查文件編碼
python rag_system/pdf_processor.py --single_file book/某本書.pdf
```

**問題**: LLM連接失敗  
```bash
# 解決方案：檢查服務狀態
curl http://localhost:8090/v1/models
# 系統會自動使用備用回答
```

**問題**: 記憶體不足
```bash
# 解決方案：減少batch_size
# 編輯 rag_system/vector_store.py，將batch_size改為16
```

## 📊 效果預期

✅ **成功設置後你會看到**:
- RAG系統加載成功，包含3,304個知識片段  
- 查詢能提供專業的中醫回答
- 顯示相關的書籍來源信息
- 快取命中時響應速度 < 1秒

⚠️ **如果LLM服務有問題**:
- 系統仍能正常工作
- 使用內建的專業備用回答
- 所有檢索功能正常

## 🎉 恭喜！

現在你已經擁有一個完整的中醫知識問答系統，可以：
- 回答中醫理論問題  
- 提供方劑組成和功效
- 給出調理建議
- 引用權威教科書資料

開始探索中醫知識的海洋吧！🌊