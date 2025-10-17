import os
import json
import numpy as np
import pickle
from typing import List, Dict, Tuple, Optional, Any
from pathlib import Path
import logging
from datetime import datetime
import hashlib

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False

try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TCMVectorStore:
    """中醫知識向量數據庫"""
    
    def __init__(self, 
                 store_dir: str = "rag_system/vectordb",
                 embedding_model: str = "BAAI/bge-base-zh-v1.5",
                 embedding_dim: int = 768):
        """
        初始化向量數據庫
        
        Args:
            store_dir: 向量數據庫存儲目錄
            embedding_model: 嵌入模型名稱
            embedding_dim: 嵌入維度
        """
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)
        
        self.embedding_model_name = embedding_model
        self.embedding_dim = embedding_dim
        self.embedding_model = None
        
        # 向量索引和元數據
        self.index = None
        self.metadata = []
        self.doc_embeddings = []
        
        # 索引文件路径
        self.index_file = self.store_dir / "tcm_knowledge.index"
        self.metadata_file = self.store_dir / "metadata.json"
        self.embeddings_file = self.store_dir / "embeddings.pkl"
        
        self._load_embedding_model()
        
    def _load_embedding_model(self):
        """加載嵌入模型"""
        if not HAS_SENTENCE_TRANSFORMERS:
            logger.error("未安裝 sentence-transformers，請運行: pip install sentence-transformers")
            return
        
        try:
            logger.info(f"正在加載嵌入模型: {self.embedding_model_name}")
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
            logger.info("嵌入模型加載成功")
        except Exception as e:
            logger.error(f"嵌入模型加載失敗: {e}")
            # 嘗試使用備用模型
            try:
                logger.info("嘗試使用備用模型: paraphrase-multilingual-MiniLM-L12-v2")
                self.embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
                self.embedding_dim = 384  # MiniLM 的維度
                logger.info("備用嵌入模型加載成功")
            except Exception as e2:
                logger.error(f"備用模型也加載失敗: {e2}")
                self.embedding_model = None
    
    def check_dependencies(self) -> bool:
        """檢查依賴包"""
        missing = []
        if not HAS_FAISS:
            missing.append("faiss-cpu")
        if not HAS_SENTENCE_TRANSFORMERS:
            missing.append("sentence-transformers")
        if not HAS_TORCH:
            missing.append("torch")
            
        if missing:
            logger.error(f"缺少依賴包: {', '.join(missing)}")
            logger.info("請運行: pip install faiss-cpu sentence-transformers torch")
            return False
        return True
    
    def encode_texts(self, texts: List[str], show_progress: bool = True) -> np.ndarray:
        """
        文本向量化
        
        Args:
            texts: 文本列表
            show_progress: 是否顯示進度
            
        Returns:
            向量矩陣
        """
        if not self.embedding_model:
            logger.error("嵌入模型未加載")
            return np.array([])
        
        logger.info(f"正在向量化 {len(texts)} 個文本...")
        
        try:
            embeddings = self.embedding_model.encode(
                texts, 
                show_progress_bar=show_progress,
                batch_size=32,
                convert_to_numpy=True
            )
            logger.info(f"向量化完成，形狀: {embeddings.shape}")
            return embeddings
        except Exception as e:
            logger.error(f"文本向量化失敗: {e}")
            return np.array([])
    
    def build_index(self, processed_data_dir: str):
        """
        構建向量索引
        
        Args:
            processed_data_dir: 處理後的數據目錄
        """
        if not self.check_dependencies():
            return
        
        data_dir = Path(processed_data_dir)
        if not data_dir.exists():
            logger.error(f"數據目錄不存在: {data_dir}")
            return
        
        # 收集所有文本塊
        all_chunks = []
        all_metadata = []
        
        json_files = list(data_dir.glob("*_processed.json"))
        logger.info(f"找到 {len(json_files)} 個處理後的文件")
        
        for json_file in json_files:
            logger.info(f"處理文件: {json_file.name}")
            
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            chunks = data.get('chunks', [])
            for chunk in chunks:
                all_chunks.append(chunk['content'])
                
                # 構建元數據
                metadata = {
                    'book_id': chunk['book_id'],
                    'book_title': chunk['book_title'],
                    'chapter_id': chunk['chapter_id'],
                    'chapter_title': chunk['chapter_title'],
                    'chunk_id': chunk['chunk_id'],
                    'content_length': chunk['content_length'],
                    'source_file': json_file.name,
                    'chunk_hash': hashlib.md5(chunk['content'].encode()).hexdigest()
                }
                all_metadata.append(metadata)
        
        logger.info(f"總共收集到 {len(all_chunks)} 個文本塊")
        
        if not all_chunks:
            logger.error("沒有找到可處理的文本塊")
            return
        
        # 向量化
        embeddings = self.encode_texts(all_chunks)
        if embeddings.size == 0:
            logger.error("文本向量化失敗")
            return
        
        # 創建FAISS索引
        logger.info("正在構建FAISS索引...")
        if HAS_FAISS:
            # 使用內積索引，適合歸一化的向量
            self.index = faiss.IndexFlatIP(embeddings.shape[1])
            
            # 歸一化向量
            faiss.normalize_L2(embeddings)
            
            # 添加向量
            self.index.add(embeddings.astype('float32'))
            
            logger.info(f"FAISS索引構建完成，包含 {self.index.ntotal} 個向量")
        
        # 保存數據
        self.metadata = all_metadata
        self.doc_embeddings = embeddings
        
        self.save_index()
        logger.info("向量索引構建並保存完成")
    
    def save_index(self):
        """保存索引到文件"""
        try:
            # 保存FAISS索引
            if self.index and HAS_FAISS:
                faiss.write_index(self.index, str(self.index_file))
                logger.info(f"FAISS索引已保存: {self.index_file}")
            
            # 保存元數據
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'metadata': self.metadata,
                    'embedding_model': self.embedding_model_name,
                    'embedding_dim': self.embedding_dim,
                    'total_documents': len(self.metadata),
                    'created_at': datetime.now().isoformat()
                }, f, ensure_ascii=False, indent=2)
            logger.info(f"元數據已保存: {self.metadata_file}")
            
            # 保存嵌入向量
            with open(self.embeddings_file, 'wb') as f:
                pickle.dump(self.doc_embeddings, f)
            logger.info(f"嵌入向量已保存: {self.embeddings_file}")
            
        except Exception as e:
            logger.error(f"保存索引失敗: {e}")
    
    def load_index(self) -> bool:
        """加載已保存的索引"""
        try:
            # 加載FAISS索引
            if self.index_file.exists() and HAS_FAISS:
                self.index = faiss.read_index(str(self.index_file))
                logger.info(f"FAISS索引已加載: {self.index.ntotal} 個向量")
            
            # 加載元數據
            if self.metadata_file.exists():
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.metadata = data['metadata']
                    logger.info(f"元數據已加載: {len(self.metadata)} 個文檔")
            
            # 加載嵌入向量
            if self.embeddings_file.exists():
                with open(self.embeddings_file, 'rb') as f:
                    self.doc_embeddings = pickle.load(f)
                logger.info(f"嵌入向量已加載: {len(self.doc_embeddings)} 個向量")
            
            return True
            
        except Exception as e:
            logger.error(f"加載索引失敗: {e}")
            return False
    
    def search(self, query: str, top_k: int = 5, score_threshold: float = 0.3) -> List[Dict]:
        """
        搜索相關文檔
        
        Args:
            query: 查詢文本
            top_k: 返回結果數量
            score_threshold: 相似度閾值
            
        Returns:
            搜索結果列表
        """
        if not self.index or not self.embedding_model:
            logger.error("索引或嵌入模型未加載")
            return []
        
        # 查詢向量化
        query_embedding = self.encode_texts([query], show_progress=False)
        if query_embedding.size == 0:
            return []
        
        # 歸一化查詢向量
        if HAS_FAISS:
            faiss.normalize_L2(query_embedding)
        
        # 搜索
        scores, indices = self.index.search(query_embedding.astype('float32'), top_k)
        
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx == -1 or score < score_threshold:  # FAISS用-1表示無效結果
                continue
                
            metadata = self.metadata[idx].copy()
            metadata['similarity_score'] = float(score)
            metadata['rank'] = i + 1
            results.append(metadata)
        
        logger.info(f"搜索查詢: '{query}', 返回 {len(results)} 個結果")
        return results
    
    def get_document_content(self, metadata: Dict) -> str:
        """
        根據元數據獲取文檔內容
        
        Args:
            metadata: 文檔元數據
            
        Returns:
            文檔內容
        """
        try:
            source_file = self.store_dir.parent / "data" / metadata['source_file']
            
            with open(source_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            chunks = data.get('chunks', [])
            for chunk in chunks:
                if (chunk['book_id'] == metadata['book_id'] and 
                    chunk['chapter_id'] == metadata['chapter_id'] and
                    chunk['chunk_id'] == metadata['chunk_id']):
                    return chunk['content']
                    
        except Exception as e:
            logger.error(f"獲取文檔內容失敗: {e}")
        
        return ""
    
    def search_with_content(self, query: str, top_k: int = 5, score_threshold: float = 0.3) -> List[Dict]:
        """
        搜索並返回完整內容
        
        Args:
            query: 查詢文本
            top_k: 返回結果數量  
            score_threshold: 相似度閾值
            
        Returns:
            包含內容的搜索結果
        """
        results = self.search(query, top_k, score_threshold)
        
        for result in results:
            result['content'] = self.get_document_content(result)
        
        return results
    
    def get_stats(self) -> Dict:
        """獲取向量庫統計信息"""
        if not self.metadata:
            return {"error": "向量庫未初始化"}
        
        book_stats = {}
        chapter_stats = {}
        
        for meta in self.metadata:
            book_id = meta['book_id']
            book_title = meta['book_title']
            chapter_title = meta['chapter_title']
            
            if book_id not in book_stats:
                book_stats[book_id] = {
                    'title': book_title,
                    'chunks': 0,
                    'chapters': set()
                }
            
            book_stats[book_id]['chunks'] += 1
            book_stats[book_id]['chapters'].add(chapter_title)
            
            if chapter_title not in chapter_stats:
                chapter_stats[chapter_title] = 0
            chapter_stats[chapter_title] += 1
        
        # 轉換章節統計中的 set 為 count
        for book in book_stats.values():
            book['chapters'] = len(book['chapters'])
        
        return {
            'total_documents': len(self.metadata),
            'total_books': len(book_stats),
            'embedding_model': self.embedding_model_name,
            'embedding_dim': self.embedding_dim,
            'books': book_stats,
            'top_chapters': sorted(chapter_stats.items(), key=lambda x: x[1], reverse=True)[:10]
        }

def main():
    """主函數 - 用於命令行調用"""
    import argparse
    
    parser = argparse.ArgumentParser(description='中醫知識向量數據庫工具')
    parser.add_argument('--action', choices=['build', 'search', 'stats'], 
                       required=True, help='操作類型')
    parser.add_argument('--data_dir', default='rag_system/data', 
                       help='處理後數據目錄')
    parser.add_argument('--store_dir', default='rag_system/vectordb', 
                       help='向量庫存儲目錄')
    parser.add_argument('--query', help='搜索查詢 (用於search操作)')
    parser.add_argument('--top_k', type=int, default=5, help='返回結果數量')
    parser.add_argument('--threshold', type=float, default=0.3, help='相似度閾值')
    
    args = parser.parse_args()
    
    vector_store = TCMVectorStore(store_dir=args.store_dir)
    
    if args.action == 'build':
        vector_store.build_index(args.data_dir)
    
    elif args.action == 'search':
        if not args.query:
            print("搜索操作需要提供 --query 參數")
            return
        
        if not vector_store.load_index():
            print("加載向量索引失敗")
            return
        
        results = vector_store.search_with_content(
            args.query, args.top_k, args.threshold
        )
        
        print(f"\n搜索結果 ('{args.query}'):")
        print("=" * 60)
        for result in results:
            print(f"相似度: {result['similarity_score']:.3f}")
            print(f"書籍: {result['book_title']}")
            print(f"章節: {result['chapter_title']}")
            print(f"內容: {result['content'][:200]}...")
            print("-" * 40)
    
    elif args.action == 'stats':
        if not vector_store.load_index():
            print("加載向量索引失敗")
            return
            
        stats = vector_store.get_stats()
        print(f"\n向量庫統計信息:")
        print("=" * 50)
        print(f"總文檔數: {stats['total_documents']}")
        print(f"總書籍數: {stats['total_books']}")
        print(f"嵌入模型: {stats['embedding_model']}")
        print(f"向量維度: {stats['embedding_dim']}")
        
        print(f"\n書籍統計:")
        for book_id, info in stats['books'].items():
            print(f"  {info['title']}: {info['chunks']} 個文本塊, {info['chapters']} 個章節")

if __name__ == "__main__":
    main()