import dashscope
import numpy as np
import os


class DashTextEmbeddingModel:
    def __init__(self):
        print("Initializing DashTextEmbeddingModel with Ali-Clound...")
        # 设置DashScope API密钥
        api_key = os.getenv('DASHSCOPE_API_KEY')
        if api_key:
            dashscope.api_key = api_key
        else:
            raise ValueError("DASHSCOPE_API_KEY environment variable not set")
        
    def encode(self, query, docs):
        # 场景：为搜索引擎构建文档向量时，可以添加指令以优化用于检索的向量质量。
        query_embeddings = dashscope.TextEmbedding.call(
            model="text-embedding-v4",
            input=query,
            text_type="query",
            instruct="Given a query, retrieve relevant research paper"
        ).output['embeddings'][0]['embedding']
        query_embeddings = np.asarray(query_embeddings).reshape(1, -1)
        
        document_embeddings = dashscope.TextEmbedding.call(
            model="text-embedding-v4",
            input=docs,
            text_type="document"
        ).output['embeddings']
        document_emb_list = []
        for doc_emb in document_embeddings:
            document_emb_list.append(doc_emb['embedding'])
        document_embeddings = np.asarray(document_emb_list)

        return query_embeddings, document_embeddings