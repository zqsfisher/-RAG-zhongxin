import logging
from pathlib import Path
from sentence_transformers import CrossEncoder

_log = logging.getLogger(__name__)


# JinaReranker：基于本地 Jina Reranker v2 模型的重排器，无需网络
class JinaReranker:
    def __init__(
        self,
        model_path: str = None,
        model_name: str = "jinaai/jina-reranker-v2-base-multilingual",
        trust_remote_code: bool = True,
        max_length: int = 512,
    ):
        """
        初始化本地 Jina 重排器。
        优先使用本地模型路径，若未提供则从 HuggingFace 自动下载。

        参数：
            model_path: 本地模型文件夹路径（如 models/jina-reranker）
            model_name: HuggingFace 模型名称（本地路径不存在时自动下载）
            trust_remote_code: 是否信任远程代码（jina 模型需要 True）
            max_length: 重排时文档最大 token 数（截断以加速 CPU 推理）
        """
        if model_path:
            local_path = Path(model_path)
            if local_path.exists():
                _log.info(f"Loading Jina reranker from local path: {local_path}")
                self.model = CrossEncoder(
                    str(local_path),
                    trust_remote_code=trust_remote_code,
                    max_length=max_length,
                    model_kwargs={"use_safetensors": True},
                )
            else:
                _log.warning(f"Local path {model_path} not found, downloading {model_name}")
                self.model = CrossEncoder(
                    model_name,
                    trust_remote_code=trust_remote_code,
                    max_length=max_length,
                    model_kwargs={"use_safetensors": True},
                )
        else:
            _log.info(f"Loading Jina reranker: {model_name}")
            self.model = CrossEncoder(
                model_name,
                trust_remote_code=trust_remote_code,
                max_length=max_length,
                model_kwargs={"use_safetensors": True},
            )

    def rerank_documents(self, query: str, documents: list, top_n: int = None):
        """
        使用本地 Jina 模型对文档进行重排。
        参数：
            query: 查询语句
            documents: 待重排的文档列表，每个元素需包含'text'
            top_n: 返回的文档数量，默认返回全部
        返回：
            按相关性分数降序排序的文档列表
        """
        if not documents:
            return []

        # 提取文档文本
        doc_texts = [doc['text'] for doc in documents]
        n_docs = len(doc_texts)

        # 构建 (query, document) 对，批量推理
        pairs = [(query, text) for text in doc_texts]
        scores = self.model.predict(pairs, show_progress_bar=False)

        # 处理单文档情况（predict 可能返回标量）
        if n_docs == 1:
            scores = [float(scores)]

        # 将分数合并回原始文档
        for i, doc in enumerate(documents):
            doc["relevance_score"] = round(float(scores[i]), 4)
            doc["combined_score"] = round(float(scores[i]), 4)

        # 按分数降序排序
        documents.sort(key=lambda x: x["combined_score"], reverse=True)

        if top_n:
            return documents[:top_n]
        return documents
