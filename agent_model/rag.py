import os
import glob
import time
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
from sentence_transformers import SentenceTransformer


class RealTimeRAG:
    
    def __init__(self, 
                 embedding_model_name: str = "./Qwen3-0.6B-embedding",
                 chunk_size: int = 500,
                 chunk_overlap: int = 50,
                 persist_directory: str = "./multi_faiss_db"):
        """
        Args:
            embedding_model_name: SentenceTransformer模型名称
            chunk_size: 文本分割块大小
            chunk_overlap: 文本分割重叠大小
            persist_directory: FAISS索引存储根目录
        """
        self.embedding_model_name = embedding_model_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.persist_directory = persist_directory
        
        # 确保存储目录存在
        # os.makedirs(self.persist_directory, exist_ok=True)
        
        # 初始化组件
        self._initialize_components()
        
        # 存储状态
        # self.vectorstores = {}  # 文件名 -> FAISS向量库
        # self.document_info = {}  # 文件信息
        # self.is_loaded = False
        # self.document_count = 0
        # self.loaded_files = set()
    
    def _initialize_components(self):
        """初始化各个组件"""
        print("🔧 Initializing Episodic RAG components...")
        
        # 1. 加载embedding模型
        print(f"🔢 Loading embedding model: {self.embedding_model_name}")
        start_time = time.time()
        self.embedding_model = SentenceTransformer(self.embedding_model_name)
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        self.text_spliter = SimpleTextSplitter(self.chunk_size, self.chunk_overlap)
        end_time = time.time()
        print(f"✅ Embedding model loaded in {end_time - start_time:.2f}s")
        print(f"📊 Embedding dimension: {self.embedding_dim}")
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
                # print("doc:", doc)
                # print("chunk:", chunks)
                chunks_list += chunks
            documents = chunks_list
                    
        # 1. 编码query和documents
        query_embeddings = self.embedding_model.encode(query, prompt_name="query", normalize_embeddings=True)
        document_embeddings = self.embedding_model.encode(documents, normalize_embeddings=True)
        
        # 2. 计算相似度（cosine similarity）
        # SentenceTransformer的similarity返回的是cosine相似度矩阵
        similarity_matrix = np.matmul(query_embeddings, document_embeddings.T)
        # print(similarity_matrix, similarity_matrix.shape)
        
        # 3. 获取query与每个document的相似度分数
        # similarity_matrix形状为 (1, num_documents)
        similarity_scores = similarity_matrix[0]
        
        # 4. 按分数降序排序，获取前k个索引
        if k > len(documents):
            k = len(documents)
        
        # 获取前k个最高分的索引
        top_k_indices = np.argsort(similarity_scores)[-k:][::-1]
        
        # 5. 应用分数阈值（如果提供）
        if score_threshold is not None:
            valid_indices = [idx for idx in top_k_indices if similarity_scores[idx] >= score_threshold]
            if not valid_indices:
                return ([], []) if return_scores else []
            top_k_indices = np.array(valid_indices)
        
        # 6. 获取对应的文档和分数
        top_documents = [documents[idx] for idx in top_k_indices]
        top_scores = [float(similarity_scores[idx]) for idx in top_k_indices]
        
        # 7. 返回结果
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


if __name__ == "__main__":
    spliter = SimpleTextSplitter()
    # rager = RealTimeRAG()
    query = "deepfake"
    docs = ["AI-Generated or Real? Please identify whether the image is real or fake and how confident you are. Scroll Down. ↓ Advice: 🧍‍♂️ Notice the posture and overall appearance of the e people—do they look consistent and realistic? 🚪 Check the cabinets and objects—do the door handles make sense and appear properly placed? Real: This is a real image. Fake: This is an AI-generated image. How confident are you? Not at all Slightly Moderately Very Perfectly Submit your initial guess Real or AI-Generated? Real: This is a real image. Fake: This is an AI-generated image. How confident are you? Not at all Slightly Moderately Very Perfectly How much do you think others would agree with your judgment? Almost no one Few About half Most Almost everyone Submit your final guess Real: This is a real image Next Image", 
            "一文速览深度伪造检测（Detection of Deepfakes）：未来技术的守门人 前言 一、Deepfakes技术原理 卷积神经网络（CNN）：细致的艺术学徒 生成对抗网络（GAN）：画家与评审的双重角色 训练过程：技艺的磨练 应用和挑战 二、Detection of Deepfakes技术原理：解密数字伪装 特征提取：寻找数字足迹 异常检测：寻找不和谐的旋律 深度学习模型：构建智能的守门人 多模态分析：全方位的监控系统 未来展望：挑战与机遇并存 🌈你好呀！我是 是Yu欸 🌌 2024每日百字篆刻时光，感谢你的陪伴与支持 ~ 🚀 欢迎一起踏上探险之旅，挖掘无限可能，共同成长！ 前些天发现了一个人工智能学习网站，内容深入浅出、易于理解。如果对人工智能感兴趣，不妨 点击查看 。 前言 在数字化时代的高速公路上，深度伪造技术（Deepfake）如同一辆无人驾驶的跑车，其速度惊人，潜力巨大，同时也带来了潜在的危险。 深度伪造检测（Detection of Deepfakes）不仅是一场科技界的军备竞赛，更是未来数字内容安全领域的黄金矿脉。本文将探讨这一技术的核心原理，揭示其如何成为数字时 代守门人的角色。 重现和替换的对比 编辑 合成： 参考：https://zhuanlan.zhihu.com/p/139489768 https://zhuanlan.zhihu.com/p/564661269 一、Deepfakes技术原理 Deepfakes技术，是一种基 于深度学习的图像、视频和音频合成技术，能够创建看起来非常真实的假象。这项技术的名字来源于“深度学习（Deep Learning）”和“假冒（Fake）”的结合，它利用了深度学习的一种特殊形式——卷积 神经网络（CNN）和生成对抗网络（GAN）来实现其核心功能。 将Deepfakes技术比喻为一位高超的画家和他的挑剔评审，可以形象地解释这项技术背后的专业术语和原理。在这个比喻中，深度学习的复杂世界被简化为艺术创作的过程，旨在创造出足以欺骗观众眼睛的作品。 以下是Deepfakes技术原理的简要介绍： 卷积神经网络（CNN）：细致的艺术学徒 CNN是一类特别设计来识别和处理图像的深度神经网络。在Deepfakes技术中，CNN用于分析和理解输入的图像或视频帧，如人脸的特征和表情。CNN通过从大量的数据中学习，能够识别不同人脸的细微差异，并提取出关键特征，为后续的处理步骤 打下基础。 想象一位年轻的艺术学徒（CNN），他正在学习如何精确地捕捉人物的面部特征和表情。通过观察成千上万的肖像画，这位学徒学会了如何识别面部的每一条线条和阴影，就像CNN通过分析 大量图像数据学习识别和处理图像特征一样。学徒的目标是掌握复制任何人物面部特征的技艺，以至于他的作品可以与原作媲美。 生成对抗网络（GAN）：画家与评审的双重角色 GAN是由两部分组成的深度学习模型：一个生成器（Generator）和一个鉴别器（Discriminator）。在Deepfakes中，生成器的任务是创建尽可能真实的假图像或视频帧，而鉴别器的任务则是区分生成的图像与真实图像之间 的差异。这两个网络在训练过程中相互竞争，生成器不断学习如何改进其生成的图像，以使其更难被鉴别器识别，而鉴别器则不断提高其识别真伪的能力。这个过程最终会导致生成的图像质量显著提高，足以以假乱真。 在这个艺术世界里，有一位天才画家（生成器）和一位极其挑剔的艺术评论家（鉴别器）不断地较量。画家的目标是创作出极其逼真的肖像画，以至于连最细微的细节都能欺骗观众 。每次画家完成一幅作品时，评论家都会仔细审查，试图找出任何可能揭示作品为复制品的线索。如果评论家指出了作品的瑕疵，画家就会根据这些反馈回去修正，每次都试图创作出更加完美的作品。这个过程不断重复，画家的技艺（生成器的生成能力）和评论家的鉴赏眼光（鉴别器的辨别能力）都在不断提高。 训练过程：技艺的磨练 在创建Deepfakes时，首先需要收集大量的目标人物的图像或 视频资料，作为训练数据。这些数据被用来训练GAN，特别是生成器，以学习如何产生目标人物的准确和真实的面部特征。训练过程中，生成器尝试创建越来越真实的图像，而鉴别器则尝试准确地区分 真实图像和生成图像。通过这种方式，模型逐渐学会生成高质量的假图像或视频。 在Deepfakes技术的背后，这场艺术的较量实际上是一个复杂的训练过程，其中包括了无数次的尝试和错误，画家（生成器）不断尝试创作出新的作品，而评论家（鉴别器）则持续提供关键的反馈。这个过程需要大量的“艺术作品”（图像数据）作为训练材料，以确保画家能够学习到制作各种不同风格和表情的技巧。随着时间的推移，画家变得足够熟练，以至于他的作品可以轻易地与真实的肖像画混淆。 应用和挑战 Deepfakes技术的发展，虽然在娱乐、电影制作、个人隐私保护等领域提供了新的可能性，但同时也 引发了伦理、法律和社会安全方面的重大关切。因为它可以被用来制作误导性的内容，影响公众舆论，甚至损害个人声誉。 虽然这位画家（Deepfakes生成器）的技艺令人钦佩，但他的能力也引发了一系列伦理和道德上的问题。在这个数字化的艺术世界中，他的作品可能被用于创造误导性的内容，影响公众意见或损害个人声誉。因此，虽然这项技术展示了深度学习的巨大潜力，但同时也提醒我们需要谨慎地考虑其应用的界限和后果。 总之，Deepfakes技术的原理涉及到复杂的深度学习算法，尤其是CNN和GAN，它们共同作用于生成难以区分真伪的图像和视频。随着技术的发展，如何平衡其创新应用与潜在风险，成为了一个亟待解决的问题。 二、Detection of Deepfakes技术原理：解密数字伪装 特征提取：寻找数字足迹 想象一下，如果将每个视频比作一个复杂的迷宫，那么深度伪造视频检 测技术就是那些试图找到出口的探险者。这些探险者（检测算法）首先需要识别迷宫中的关键线索（视频特征），这包括了面部的微妙变化、眼睛的闪烁频率，甚至是光线投射的方式。通过精确分析这些细微的线索，检测算法可以开始判断这个迷宫是真实存在的，还是某种技术创造出来的幻象。 异常检测：寻找不和谐的旋律 将每个视频比作一首曲子，那么异常检测就在于辨识出其中的不和谐音符。深度学习模型通过大量的训练，学会了识别哪些音符（视频特征）属于正常的旋律，哪些则暗示着曲子被人为篡改。这就像一位经验丰富的音乐家能够凭借细腻的听觉察觉出演奏中的微小失误。 深 度学习模型：构建智能的守门人 深度学习模型是深度虚假视频检测技术的核心，它们就像是训练有素的守门人，守护着数字内容的真实性。通过对大量真实和伪造视频的学习，这些守门人逐渐掌握了 区分二者的能力。无论伪造技术如何进步，只要持续对这些守门人进行训练，它们就能适应新的挑战，保护数字世界的安全。 多模态分析：全方位的监控系统 在深度虚假视频检测中，仅仅分析视频是不够的，就像一座要塞不可能只依靠一道防线。多模态分析允许检测系统同时监控视频和音频，甚至是它们之间的关联，从而构建起一套更为全面的防御机制。这就像是在要塞的每个角落都部署了哨兵，无论敌人从哪个方向来袭，都能被及时发现和拦截。 未来展望：挑战与机遇并存 随着深度伪造技术的不断进化，深度虚假视频检测面临着前所未有的挑战。然而，正是这种挑战，提供了独特的机遇。 这一领域的先进技术和解决方案，不仅可以保护社会免受虚假信息的侵害，也能在未来的数字安全领域占据有利地位。 作为未来技术的守门人，深度虚假视频检测技术正站在风口浪尖，共同守护数字世界的真实性和安全性。"]
    
    chunks_list = []
    for doc in docs:
        chunks = spliter.split_text(doc)
        chunks_list += chunks
    print(len(chunks_list))
    # top_k = rager.execute(query, chunks_list)
    # print(top_k)