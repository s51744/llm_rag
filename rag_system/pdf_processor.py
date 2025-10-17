import os
import json
import re
from typing import List, Dict, Tuple
from pathlib import Path
import logging
from datetime import datetime

try:
    import PyPDF2
    import pdfplumber
    HAS_PDF_LIBS = True
except ImportError:
    HAS_PDF_LIBS = False

try:
    from transformers import AutoTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TCMPDFProcessor:
    """中醫教科書PDF處理器"""
    
    def __init__(self, output_dir: str = "rag_system/data"):
        """
        初始化PDF處理器
        
        Args:
            output_dir: 處理後文本的輸出目錄
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 中醫專業詞彙正規化字典
        self.tcm_normalize_dict = {
            '脾': '脾',
            '胃': '胃', 
            '肝': '肝',
            '腎': '腎',
            '心': '心',
            '肺': '肺',
            '氣': '氣',
            '血': '血',
            '陰': '陰',
            '陽': '陽',
            '虛': '虛',
            '實': '實',
            '寒': '寒',
            '熱': '熱',
            '濕': '濕',
            '燥': '燥',
            '風': '風',
            '火': '火'
        }
        
        # 章節標題模式 (中醫教科書常見)
        self.chapter_patterns = [
            r'第[一二三四五六七八九十\d]+章\s*(.+)',
            r'第[一二三四五六七八九十\d]+篇\s*(.+)',
            r'第[一二三四五六七八九十\d]+節\s*(.+)',
            r'第[一二三四五六七八九十\d]+部\s*(.+)',
            r'[一二三四五六七八九十]、\s*(.+)',
            r'\d+\.\s*(.+)',
            r'【.+】',
            r'〖.+〗'
        ]
        
    def check_dependencies(self) -> bool:
        """檢查必要的依賴是否已安裝"""
        missing_deps = []
        
        if not HAS_PDF_LIBS:
            missing_deps.append("PyPDF2 和 pdfplumber")
            
        if missing_deps:
            logger.error(f"缺少必要的依賴包: {', '.join(missing_deps)}")
            logger.info("請運行: pip install PyPDF2 pdfplumber")
            return False
            
        return True
    
    def extract_text_from_pdf(self, pdf_path: str, method: str = "pdfplumber") -> str:
        """
        從PDF提取文本
        
        Args:
            pdf_path: PDF文件路徑
            method: 提取方法 ("pdfplumber" 或 "pypdf2")
            
        Returns:
            提取的文本內容
        """
        if not self.check_dependencies():
            return ""
            
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            logger.error(f"PDF文件不存在: {pdf_path}")
            return ""
            
        logger.info(f"正在處理PDF: {pdf_path.name}")
        text_content = ""
        
        try:
            if method == "pdfplumber" and HAS_PDF_LIBS:
                with pdfplumber.open(pdf_path) as pdf:
                    total_pages = len(pdf.pages)
                    logger.info(f"總頁數: {total_pages}")
                    
                    for i, page in enumerate(pdf.pages):
                        if i % 50 == 0:  # 每50頁顯示進度
                            logger.info(f"處理進度: {i+1}/{total_pages}")
                        
                        page_text = page.extract_text()
                        if page_text:
                            text_content += f"\n--- 第{i+1}頁 ---\n"
                            text_content += page_text + "\n"
                            
            elif method == "pypdf2" and HAS_PDF_LIBS:
                with open(pdf_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    total_pages = len(pdf_reader.pages)
                    logger.info(f"總頁數: {total_pages}")
                    
                    for i, page in enumerate(pdf_reader.pages):
                        if i % 50 == 0:
                            logger.info(f"處理進度: {i+1}/{total_pages}")
                        
                        page_text = page.extract_text()
                        if page_text:
                            text_content += f"\n--- 第{i+1}頁 ---\n"
                            text_content += page_text + "\n"
                            
        except Exception as e:
            logger.error(f"PDF處理出錯 {pdf_path.name}: {str(e)}")
            return ""
            
        logger.info(f"成功提取文本，共 {len(text_content)} 個字符")
        return text_content
    
    def clean_and_normalize_text(self, text: str) -> str:
        """
        清理和標準化文本
        
        Args:
            text: 原始文本
            
        Returns:
            清理後的文本
        """
        if not text:
            return ""
        
        # 移除頁碼標記
        text = re.sub(r'--- 第\d+頁 ---\n?', '', text)
        
        # 移除多餘的空白字符
        text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)  # 多個連續換行變成兩個
        text = re.sub(r'[ \t]+', ' ', text)  # 多個空格變成一個
        
        # 移除頁眉頁腳常見模式
        text = re.sub(r'\n\d+\n', '\n', text)  # 單獨的數字行
        text = re.sub(r'\n[一二三四五六七八九十]+\n', '\n', text)  # 單獨的中文數字
        
        # 標準化中醫術語
        for old_term, new_term in self.tcm_normalize_dict.items():
            text = text.replace(old_term, new_term)
        
        # 移除異常字符
        text = re.sub(r'[^\u4e00-\u9fff\u3000-\u303f\uff00-\uffef\w\s\n.,;:!?()（）【】「」〈〉《》""''…—–-]', '', text)
        
        return text.strip()
    
    def split_into_chapters(self, text: str) -> List[Dict[str, str]]:
        """
        將文本按章節分割
        
        Args:
            text: 輸入文本
            
        Returns:
            章節列表，每個包含標題和內容
        """
        chapters = []
        lines = text.split('\n')
        current_chapter = {"title": "前言", "content": ""}
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # 檢查是否是章節標題
            is_chapter_title = False
            chapter_title = ""
            
            for pattern in self.chapter_patterns:
                match = re.match(pattern, line)
                if match:
                    is_chapter_title = True
                    if match.groups():
                        chapter_title = match.group(1).strip()
                    else:
                        chapter_title = line
                    break
            
            if is_chapter_title and len(current_chapter["content"]) > 100:
                # 保存前一章節
                chapters.append(current_chapter.copy())
                # 開始新章節
                current_chapter = {"title": chapter_title, "content": ""}
            else:
                current_chapter["content"] += line + "\n"
        
        # 添加最後一章
        if current_chapter["content"].strip():
            chapters.append(current_chapter)
        
        logger.info(f"分割出 {len(chapters)} 個章節")
        return chapters
    
    def split_into_chunks(self, text: str, chunk_size: int = 512, overlap: int = 50) -> List[str]:
        """
        將文本分割成適合向量化的塊
        
        Args:
            text: 輸入文本
            chunk_size: 每塊的最大字符數
            overlap: 塊間重疊的字符數
            
        Returns:
            文本塊列表
        """
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # 如果不是最後一塊，嘗試在句號或段落處分割
            if end < len(text):
                # 尋找最近的句號
                sentence_end = text.rfind('。', start, end)
                if sentence_end > start + chunk_size // 2:
                    end = sentence_end + 1
                else:
                    # 尋找最近的段落分隔
                    para_end = text.rfind('\n', start, end)
                    if para_end > start + chunk_size // 2:
                        end = para_end
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            # 計算下一塊的開始位置（考慮重疊）
            start = max(end - overlap, start + 1)
            
        logger.info(f"分割出 {len(chunks)} 個文本塊")
        return chunks
    
    def process_single_pdf(self, pdf_path: str, book_id: str = None) -> Dict:
        """
        處理單個PDF文件
        
        Args:
            pdf_path: PDF文件路徑
            book_id: 書籍ID（可選）
            
        Returns:
            處理結果
        """
        pdf_path = Path(pdf_path)
        if book_id is None:
            book_id = pdf_path.stem
        
        # 提取文本
        raw_text = self.extract_text_from_pdf(pdf_path)
        if not raw_text:
            return {"status": "error", "message": "文本提取失敗"}
        
        # 清理和標準化
        cleaned_text = self.clean_and_normalize_text(raw_text)
        
        # 分章節
        chapters = self.split_into_chapters(cleaned_text)
        
        # 為每章節分塊
        all_chunks = []
        for i, chapter in enumerate(chapters):
            chapter_chunks = self.split_into_chunks(chapter["content"])
            
            for j, chunk in enumerate(chapter_chunks):
                chunk_data = {
                    "book_id": book_id,
                    "book_title": pdf_path.stem,
                    "chapter_id": i,
                    "chapter_title": chapter["title"],
                    "chunk_id": j,
                    "content": chunk,
                    "content_length": len(chunk),
                    "created_at": datetime.now().isoformat()
                }
                all_chunks.append(chunk_data)
        
        # 保存處理結果
        output_file = self.output_dir / f"{book_id}_processed.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                "book_info": {
                    "book_id": book_id,
                    "title": pdf_path.stem,
                    "source_path": str(pdf_path),
                    "total_chapters": len(chapters),
                    "total_chunks": len(all_chunks),
                    "processed_at": datetime.now().isoformat()
                },
                "chapters": chapters,
                "chunks": all_chunks
            }, f, ensure_ascii=False, indent=2)
        
        logger.info(f"處理完成: {book_id}")
        logger.info(f"章節數: {len(chapters)}, 文本塊數: {len(all_chunks)}")
        logger.info(f"結果保存至: {output_file}")
        
        return {
            "status": "success",
            "book_id": book_id,
            "chapters": len(chapters),
            "chunks": len(all_chunks),
            "output_file": str(output_file)
        }
    
    def process_all_pdfs(self, pdf_dir: str) -> List[Dict]:
        """
        處理目錄中的所有PDF文件
        
        Args:
            pdf_dir: PDF文件目錄
            
        Returns:
            處理結果列表
        """
        pdf_dir = Path(pdf_dir)
        if not pdf_dir.exists():
            logger.error(f"目錄不存在: {pdf_dir}")
            return []
        
        pdf_files = list(pdf_dir.glob("*.pdf"))
        if not pdf_files:
            logger.warning(f"目錄中沒有找到PDF文件: {pdf_dir}")
            return []
        
        logger.info(f"找到 {len(pdf_files)} 個PDF文件")
        results = []
        
        for i, pdf_file in enumerate(pdf_files, 1):
            logger.info(f"\n=== 處理第 {i}/{len(pdf_files)} 個文件 ===")
            result = self.process_single_pdf(pdf_file)
            results.append(result)
        
        # 保存總體統計
        summary_file = self.output_dir / "processing_summary.json"
        summary = {
            "processed_at": datetime.now().isoformat(),
            "total_books": len(pdf_files),
            "successful": len([r for r in results if r.get("status") == "success"]),
            "failed": len([r for r in results if r.get("status") == "error"]),
            "results": results
        }
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        logger.info(f"\n=== 處理完成 ===")
        logger.info(f"總計: {len(pdf_files)} 個文件")
        logger.info(f"成功: {summary['successful']} 個")
        logger.info(f"失敗: {summary['failed']} 個")
        logger.info(f"總結報告: {summary_file}")
        
        return results

def main():
    """主函數 - 用於命令行調用"""
    import argparse
    
    parser = argparse.ArgumentParser(description='中醫教科書PDF處理工具')
    parser.add_argument('--input_dir', default='book', help='PDF文件目錄 (默認: book)')
    parser.add_argument('--output_dir', default='rag_system/data', help='輸出目錄 (默認: rag_system/data)')
    parser.add_argument('--single_file', help='處理單個PDF文件')
    parser.add_argument('--chunk_size', type=int, default=512, help='文本塊大小 (默認: 512)')
    
    args = parser.parse_args()
    
    processor = TCMPDFProcessor(output_dir=args.output_dir)
    
    if args.single_file:
        result = processor.process_single_pdf(args.single_file)
        print(f"處理結果: {result}")
    else:
        results = processor.process_all_pdfs(args.input_dir)
        print(f"批量處理完成，共處理 {len(results)} 個文件")

if __name__ == "__main__":
    main()