import os
import glob
import time
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union

from sentence_transformers import SentenceTransformer


class LocalEmbeddingModel:
    def __init__(self, model_name):
        start_time = time.time()
        self.embedding_model = SentenceTransformer(model_name)
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        end_time = time.time()
        print(f"✅ Embedding model loaded in {end_time - start_time:.2f}s")
        print(f"📊 Embedding dimension: {self.embedding_dim}")
        
    def encode(self, query, documents):
        query_embeddings = self.embedding_model.encode(query, prompt_name="query", normalize_embeddings=True)
        document_embeddings = self.embedding_model.encode(documents, normalize_embeddings=True)
        return query_embeddings, document_embeddings
            

class RealTimeRAG:
    def __init__(self, 
                 embedding_model_name: str = "./models/Qwen3-0.6B-embedding",
                 embedding_model = None,
                 chunk_size: int = 500,
                 chunk_overlap: int = 50
                 ):
        """
        Args:
            embedding_model_name: SentenceTransformer模型名称
            chunk_size: 文本分割块大小
            chunk_overlap: 文本分割重叠大小
            persist_directory: FAISS索引存储根目录
        """
        self.embedding_model_name = embedding_model_name
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        self._initialize_components()
    
    def _initialize_components(self):
        """初始化各个组件"""
        print("🔧 Initializing RAG components...")
        if self.embedding_model is None:
            print(f"🔢 Loading Local embedding model: {self.embedding_model_name}")
            self.embedding_model = LocalEmbeddingModel(self.embedding_model_name)
        self.text_spliter = SimpleTextSplitter(self.chunk_size, self.chunk_overlap)
        print("✅ Real-time RAG components initialized successfully!")
    
    
    def execute(self, 
                query: str, 
                documents: List[str],
                k: int = 5,
                return_scores: bool = False,
                score_threshold: Optional[float] = None) -> Union[List[str], Tuple[List[str], List[float]]]:
        """
        执行检索，返回与query最相似的前k个文档
        
        Args:
            query: 查询文本
            documents: 文档列表
            k: 返回的文档数量
            return_scores: 是否返回相似度分数
            score_threshold: 分数阈值，只返回分数高于该值的文档
            
        Returns:
            如果 return_scores=True: 返回 (documents, scores)
            如果 return_scores=False: 返回 documents
        """
        if not documents:
            return ([], []) if return_scores else []
        
        if isinstance(query, str):
            query = [query]
            
        if self.text_spliter:
            chunks_list = []
            for doc in documents:
                chunks = self.text_spliter.split_text(doc)
                chunks_list += chunks
            documents = chunks_list[:10] # only use top chunk for debug and for the limitation of API
                    
        # Encode query and documents
        query_embeddings, document_embeddings = self.embedding_model.encode(query, documents)
        
        # Compute cosine similarity
        # similarity_matrix's shape is (1, num_documents)
        similarity_matrix = np.matmul(query_embeddings, document_embeddings.T)
        similarity_scores = similarity_matrix[0]
        
        # Sort by scores
        if k > len(documents):
            k = len(documents)
        top_k_indices = np.argsort(similarity_scores)[-k:][::-1]
        
        # Apply score threshold (if provided)
        if score_threshold is not None:
            valid_indices = [idx for idx in top_k_indices if similarity_scores[idx] >= score_threshold]
            if not valid_indices:
                return ([], []) if return_scores else []
            top_k_indices = np.array(valid_indices)
        
        # Get corresponding docs and scores
        top_documents = [documents[idx] for idx in top_k_indices]
        top_scores = [float(similarity_scores[idx]) for idx in top_k_indices]
        
        if return_scores:
            return top_documents, top_scores
        else:
            return top_documents
        
    
import re
from typing import List


class SimpleTextSplitter:
    """
    轻量级文本分割器（无额外依赖）
    """
    
    def __init__(self, 
                 chunk_size: int = 500,
                 chunk_overlap: int = 50,
                 separators: List[str] = None):
        """
        Args:
            chunk_size: 块大小（字符数）
            chunk_overlap: 重叠大小
            separators: 分隔符列表，按优先级排序
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        if separators is None:
            # 中文友好的分隔符
            self.separators = [
                # "\n\n",      # 双换行（段落）
                # "\n",        # 单换行
                # "。",        # 句号
                # "！",        # 感叹号
                # "？",        # 问号
                # "；",        # 分号
                # "，",        # 逗号
                # " ",         # 空格
                # ""           # 最后按字符分割
            ]
        else:
            self.separators = separators
    
    def split_text(self, text: str) -> List[str]:
        """
        分割文本
        
        Args:
            text: 输入文本
            
        Returns:
            分割后的文本块列表
        """
        if not text:
            return []
        
        # 递归分割函数
        def recursive_split(current_text: str, current_separators: List[str]) -> List[str]:
            # 如果文本已经足够小，直接返回
            if len(current_text) <= self.chunk_size:
                return [current_text]
            
            # 如果没有更多分隔符，按字符分割
            if not current_separators:
                return self._split_by_length(current_text)
            
            # 获取当前分隔符
            separator = current_separators[0]
            remaining_separators = current_separators[1:]
            
            # 使用当前分隔符分割
            if separator:
                parts = current_text.split(separator)
            else:
                # 空字符串分隔符表示按字符
                return self._split_by_length(current_text)
            
            # 合并小片段
            merged_parts = []
            current_part = ""
            
            for part in parts:
                # 如果当前部分为空，直接添加分隔符
                if not current_part:
                    current_part = part + (separator if separator != "" else "")
                # 如果添加新部分后仍小于块大小，合并
                elif len(current_part) + len(separator) + len(part) <= self.chunk_size:
                    current_part += separator + part
                # 否则，保存当前部分，开始新的部分
                else:
                    if current_part:
                        merged_parts.append(current_part)
                    current_part = part
            
            # 添加最后的部分
            if current_part:
                merged_parts.append(current_part)
            
            # 如果分割结果只有1个，尝试下一个分隔符
            if len(merged_parts) == 1:
                return recursive_split(current_text, remaining_separators)
            
            # 递归处理每个部分
            final_chunks = []
            for part in merged_parts:
                chunks = recursive_split(part, self.separators)
                final_chunks.extend(chunks)
            
            return final_chunks
        
        # 开始递归分割
        chunks = recursive_split(text, self.separators)
        
        # 应用重叠
        if self.chunk_overlap > 0 and len(chunks) > 1:
            chunks = self._apply_overlap(chunks)
        
        return chunks
    
    def _split_by_length(self, text: str) -> List[str]:
        """按固定长度分割"""
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            chunks.append(text[start:end])
            start = end
        
        return chunks
    
    def _apply_overlap(self, chunks: List[str]) -> List[str]:
        """应用重叠"""
        if len(chunks) <= 1:
            return chunks
        
        overlapped_chunks = []
        
        for i in range(len(chunks)):
            current_chunk = chunks[i]
            
            # 添加上一个块的重叠部分
            if i > 0 and self.chunk_overlap > 0:
                prev_chunk = chunks[i-1]
                overlap_start = max(0, len(prev_chunk) - self.chunk_overlap)
                overlap_text = prev_chunk[overlap_start:]
                current_chunk = overlap_text + current_chunk
            
            # 添加下一个块的重叠部分
            if i < len(chunks) - 1 and self.chunk_overlap > 0:
                next_chunk = chunks[i+1]
                overlap_text = next_chunk[:min(self.chunk_overlap, len(next_chunk))]
                current_chunk = current_chunk + overlap_text
            
            overlapped_chunks.append(current_chunk)
        
        return overlapped_chunks
    
    def split_by_sentences(self, text: str) -> List[str]:
        """
        按句子分割（适合中英文混合文本）
        """
        # 中英文句子分割正则表达式
        sentence_pattern = r'(?<=[。！？.!?])\s+'
        sentences = re.split(sentence_pattern, text)
        
        # 合并短句子
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if not sentence.strip():
                continue
                
            if len(current_chunk) + len(sentence) <= self.chunk_size:
                current_chunk += " " + sentence if current_chunk else sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = sentence
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks