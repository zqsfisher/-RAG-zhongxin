import json
import logging
from typing import List, Tuple, Dict, Union
from rank_bm25 import BM25Okapi
import pickle
from pathlib import Path
import faiss
from openai import OpenAI
from dotenv import load_dotenv
import os
import numpy as np
from src.reranking import JinaReranker
import hashlib
import pandas as pd
import time

_log = logging.getLogger(__name__)

class BM25Retriever:
    def __init__(self, bm25_db_dir: Path, documents_dir: Path):
        # 初始化BM25检索器，指定BM25索引和文档目录
        self.bm25_db_dir = bm25_db_dir
        self.documents_dir = documents_dir
        
    def retrieve_by_company_name(self, company_name: str, query: str, top_n: int = 3, return_parent_pages: bool = False) -> List[Dict]:
        # 按公司名检索相关文本块，返回BM25分数最高的top_n个块
        document_path = None
        for path in self.documents_dir.glob("*.json"):
            with open(path, 'r', encoding='utf-8') as f:
                doc = json.load(f)
                if doc["metainfo"]["company_name"] == company_name:
                    document_path = path
                    document = doc
                    break
        if document_path is None:
            raise ValueError(f"No report found with '{company_name}' company name.")
        # 加载对应的BM25索引，文件名用 sha1
        bm25_path = self.bm25_db_dir / f"{document['metainfo']['sha1']}.pkl"
        with open(bm25_path, 'rb') as f:
            bm25_index = pickle.load(f)
            
        # 获取文档内容和BM25索引
        document = document
        chunks = document["content"]["chunks"]
        pages = document["content"]["pages"]
        
        # 计算BM25分数
        tokenized_query = query.split()
        scores = bm25_index.get_scores(tokenized_query)
        
        actual_top_n = min(top_n, len(scores))
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:actual_top_n]
        
        retrieval_results = []
        seen_pages = set()
        
        for index in top_indices:
            score = round(float(scores[index]), 4)
            chunk = chunks[index]
            parent_page = next(page for page in pages if page["page"] == chunk["page"])
            
            if return_parent_pages:
                if parent_page["page"] not in seen_pages:
                    seen_pages.add(parent_page["page"])
                    result = {
                        "distance": score,
                        "page": parent_page["page"],
                        "text": parent_page["text"]
                    }
                    retrieval_results.append(result)
            else:
                result = {
                    "distance": score,
                    "page": chunk["page"],
                    "text": chunk["text"]
                }
                retrieval_results.append(result)
        
        return retrieval_results



class VectorRetriever:
    def __init__(self, vector_db_dir: Path, documents_dir: Path, embedding_provider: str = "dashscope"):
        # 初始化向量检索器，加载所有向量库和文档
        self.vector_db_dir = vector_db_dir
        self.documents_dir = documents_dir
        self.all_dbs = self._load_dbs()
        # 默认使用 dashscope 作为 embedding provider
        self.embedding_provider = embedding_provider.lower()
        self.llm = self._set_up_llm()

    def _set_up_llm(self):
        # 根据 embedding_provider 初始化对应的 LLM 客户端
        load_dotenv()
        if self.embedding_provider == "openai":
            llm = OpenAI(
                api_key=os.getenv("OPENAI_API_KEY"),
                timeout=None,
                max_retries=2
            )
            return llm
        elif self.embedding_provider == "dashscope":
            import dashscope
            dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")
            return None  # dashscope 不需要 client 对象
        else:
            raise ValueError(f"不支持的 embedding provider: {self.embedding_provider}")

    def _get_embedding(self, text: str):
        # 根据 embedding_provider 获取文本的向量表示
        if self.embedding_provider == "openai":
            embedding = self.llm.embeddings.create(
                input=text,
                model="text-embedding-3-large"
            )
            return embedding.data[0].embedding
        elif self.embedding_provider == "dashscope":
            import dashscope
            rsp = dashscope.TextEmbedding.call(
                model="text-embedding-v1",
                input=[text]
            )
            # 兼容 dashscope 返回格式，不能用 resp.output，需用 resp['output']
            if 'output' in rsp and 'embeddings' in rsp['output']:
                # 多条输入（本处只有一条）
                emb = rsp['output']['embeddings'][0]
                if emb['embedding'] is None or len(emb['embedding']) == 0:
                    raise RuntimeError(f"DashScope返回的embedding为空，text_index={emb.get('text_index', None)}")
                return emb['embedding']
            elif 'output' in rsp and 'embedding' in rsp['output']:
                # 兼容单条输入格式
                if rsp['output']['embedding'] is None or len(rsp['output']['embedding']) == 0:
                    raise RuntimeError("DashScope返回的embedding为空")
                return rsp['output']['embedding']
            else:
                raise RuntimeError(f"DashScope embedding API返回格式异常: {rsp}")
        else:
            raise ValueError(f"不支持的 embedding provider: {self.embedding_provider}")

    @staticmethod
    def set_up_llm():
        # 静态方法，初始化OpenAI LLM
        load_dotenv()
        llm = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            timeout=None,
            max_retries=2
        )
        return llm

    def _load_dbs(self):
        # 加载所有向量库和对应文档，建立映射
        all_dbs = []
        all_documents_paths = list(self.documents_dir.glob('*.json'))
        for document_path in all_documents_paths:
            try:
                with open(document_path, 'r', encoding='utf-8') as f:
                    document = json.load(f)
            except Exception as e:
                _log.error(f"Error loading JSON from {document_path.name}: {e}")
                continue
            # 用 metainfo['sha1'] 拼接 faiss 文件名
            sha1 = document.get('metainfo', {}).get('sha1', None)
            if not sha1:
                _log.warning(f"No sha1 found in metainfo for document {document_path.name}")
                continue
            faiss_path = self.vector_db_dir / f"{sha1}.faiss"
            if not faiss_path.exists():
                _log.warning(f"No matching vector DB found for document {document_path.name} (sha1={sha1})")
                continue
            try:
                # FAISS 在 Windows 上无法处理含中文的路径。
                # 使用 Python open 读取字节，转为 numpy array 再反序列化，完全绕过 FAISS 的文件 I/O。
                with open(str(faiss_path), 'rb') as f:
                    index_bytes = f.read()
                data = np.frombuffer(index_bytes, dtype=np.uint8)
                vector_db = faiss.deserialize_index(data)
            except Exception as e:
                _log.error(f"Error reading vector DB for {document_path.name}: {e}")
                continue
            report = {
                "name": sha1,
                "vector_db": vector_db,
                "document": document
            }
            all_dbs.append(report)
        return all_dbs

    @staticmethod
    def get_strings_cosine_similarity(str1, str2):
        # 计算两个字符串的余弦相似度（通过嵌入）
        llm = VectorRetriever.set_up_llm()
        embeddings = llm.embeddings.create(input=[str1, str2], model="text-embedding-3-large")
        embedding1 = embeddings.data[0].embedding
        embedding2 = embeddings.data[1].embedding
        similarity_score = np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
        similarity_score = round(similarity_score, 4)
        return similarity_score

    def retrieve_by_company_name(self, company_name: str, query: str, top_n: int = 3, return_parent_pages: bool = False) -> List[Dict]:
        # 按公司名检索相关文本块，返回向量距离最近的top_n个块
        # 直接遍历所有分段 JSON，找到 company_name 匹配的文档
        target_report = None
        for report in self.all_dbs:
            document = report.get("document", {})
            metainfo = document.get("metainfo", {})
            # 优先 company_name 字段匹配，否则 fallback 到 sha1 包含关系
            if metainfo.get("company_name") == company_name:
                target_report = report
                break
            elif company_name in metainfo.get("file_name", ""):
                target_report = report
                break
        if target_report is None:
            _log.error(f"No report found with '{company_name}' company name.")
            raise ValueError(f"No report found with '{company_name}' company name.")
        # 取 sha1，直接查找 faiss 文件（不再用 sha1_name，也不做 md5 编码）
        sha1 = target_report["document"]["metainfo"].get("sha1")
        if not sha1:
            raise ValueError(f"No sha1 found in metainfo for company '{company_name}'")
        faiss_path = self.vector_db_dir / f"{sha1}.faiss"
        if not faiss_path.exists():
            raise ValueError(f"No vector DB found for '{company_name}' (sha1: {sha1})")
        document = target_report["document"]
        vector_db = target_report["vector_db"]
        chunks = document["content"]["chunks"]
        pages = document["content"].get("pages", [])
        actual_top_n = min(top_n, len(chunks))
        # 获取 query 的 embedding，支持 openai/dashscope
        embedding = self._get_embedding(query)
        embedding_array = np.array(embedding, dtype=np.float32).reshape(1, -1)
        distances, indices = vector_db.search(x=embedding_array, k=actual_top_n)
        retrieval_results = []
        seen_pages = set()
        for distance, index in zip(distances[0], indices[0]):
            distance = round(float(distance), 4)
            chunk = chunks[index]
            parent_page = None
            if pages:
                parent_page = next((page for page in pages if page["page"] == chunk.get("page")), None)
            if return_parent_pages and parent_page:
                if parent_page["page"] not in seen_pages:
                    seen_pages.add(parent_page["page"])
                    result = {
                        "distance": distance,
                        "page": parent_page["page"],
                        "text": parent_page["text"]
                    }
                    retrieval_results.append(result)
            else:
                result = {
                    "distance": distance,
                    "page": chunk.get("page", 0),
                    "text": chunk["text"]
                }
                retrieval_results.append(result)
        return retrieval_results

    def retrieve_all(self, company_name: str) -> List[Dict]:
        # 检索公司所有文本块，返回全部内容
        target_report = None
        for report in self.all_dbs:
            document = report.get("document", {})
            metainfo = document.get("metainfo")
            if not metainfo:
                continue
            if metainfo.get("company_name") == company_name:
                target_report = report
                break
        
        if target_report is None:
            _log.error(f"No report found with '{company_name}' company name.")
            raise ValueError(f"No report found with '{company_name}' company name.")
        
        document = target_report["document"]
        pages = document["content"]["pages"]
        
        all_pages = []
        for page in sorted(pages, key=lambda p: p["page"]):
            result = {
                "distance": 0.5,
                "page": page["page"],
                "text": page["text"]
            }
            all_pages.append(result)
            
        return all_pages

    def retrieve_by_sha1_list(
        self,
        sha1_list: List[str],
        query: str,
        top_n: int = 10,
        return_parent_pages: bool = False
    ) -> List[Dict]:
        """Search across multiple documents' FAISS DBs.

        Retrieves top_n results from each document, then merges and re-ranks
        by vector distance, returning the overall top_n.
        """
        query_embedding = self._get_embedding(query)
        query_vec = np.array(query_embedding, dtype=np.float32).reshape(1, -1)

        all_results = []
        for sha1 in sha1_list:
            target = None
            for report in self.all_dbs:
                doc_sha1 = report.get("document", {}).get("metainfo", {}).get("sha1", "")
                if doc_sha1 == sha1:
                    target = report
                    break
            if target is None:
                continue

            document = target["document"]
            vector_db = target["vector_db"]
            chunks = document["content"]["chunks"]
            pages = document["content"].get("pages", [])

            k = min(top_n, len(chunks))
            if k == 0:
                continue

            distances, indices = vector_db.search(x=query_vec, k=k)

            for distance, idx in zip(distances[0], indices[0]):
                chunk = chunks[idx]
                parent_page = None
                if pages:
                    parent_page = next(
                        (p for p in pages if p["page"] == chunk.get("page")), None
                    )

                # 提取行号范围（默认 [0, 0]）
                chunk_lines = chunk.get("lines", [0, 0])
                file_name = document.get("metainfo", {}).get("file_name", "")

                if return_parent_pages and parent_page:
                    result = {
                        "distance": round(float(distance), 4),
                        "page": parent_page["page"],
                        "chunk_lines": chunk_lines,
                        "text": parent_page["text"],
                        "sha1": sha1,
                        "file_name": file_name,
                    }
                else:
                    result = {
                        "distance": round(float(distance), 4),
                        "page": chunk_lines[0],  # 用起始行号代替 page
                        "chunk_lines": chunk_lines,
                        "text": chunk["text"],
                        "sha1": sha1,
                        "file_name": file_name,
                    }
                all_results.append(result)

        # Sort by distance (higher inner product = more similar for IP index)
        all_results.sort(key=lambda x: x["distance"], reverse=True)

        # Deduplicate: keep best distance for each unique text (by content hash)
        import hashlib
        seen = set()
        deduped = []
        for r in all_results:
            # 使用文本内容哈希作为去重键（避免因 page 缺失导致全部去重）
            text_hash = hashlib.md5(r["text"].encode('utf-8')).hexdigest()[:8]
            if text_hash not in seen:
                seen.add(text_hash)
                deduped.append(r)

        return deduped[:top_n]


class HybridRetriever:
    def __init__(self, vector_db_dir: Path, documents_dir: Path, reranker_model_path: str = None):
        self.vector_retriever = VectorRetriever(vector_db_dir, documents_dir)
        # 自动查找项目根目录下的本地模型
        if reranker_model_path is None:
            default_path = Path(__file__).parent.parent / "models" / "jina-reranker"
            if default_path.exists():
                reranker_model_path = str(default_path)
        self.reranker = JinaReranker(model_path=reranker_model_path)

    def retrieve_by_company_name(
        self,
        company_name: str,
        query: str,
        reranking_sample_size: int = 30,
        top_n: int = 10,
        return_parent_pages: bool = False
    ) -> List[Dict]:
        """
        使用Jina Reranker进行检索和重排。

        参数：
            company_name: 需要检索的公司名称
            query: 检索查询语句
            reranking_sample_size: 首轮向量检索返回的候选数量
            top_n: 最终返回的重排结果数量
            return_parent_pages: 是否返回完整页面（而非分块）

        返回：
            经过重排的文档字典列表，包含分数
        """
        t0 = time.time()
        # 首先用向量检索器获取初步结果
        print("[计时] [HybridRetriever] 开始向量检索 ...")
        vector_results = self.vector_retriever.retrieve_by_company_name(
            company_name=company_name,
            query=query,
            top_n=reranking_sample_size,
            return_parent_pages=return_parent_pages
        )
        t1 = time.time()
        print(f"[计时] [HybridRetriever] 向量检索耗时: {t1-t0:.2f} 秒")
        # 使用Jina对结果进行重排
        print("[计时] [HybridRetriever] 开始Jina重排 ...")
        reranked_results = self.reranker.rerank_documents(
            query=query,
            documents=vector_results,
            top_n=top_n
        )
        t2 = time.time()
        print(f"[计时] [HybridRetriever] Jina重排耗时: {t2-t1:.2f} 秒")
        print(f"[计时] [HybridRetriever] 总耗时: {t2-t0:.2f} 秒")
        return reranked_results

    def retrieve_by_sha1_list(
        self,
        sha1_list: List[str],
        query: str,
        reranking_sample_size: int = 30,
        top_n: int = 10,
        return_parent_pages: bool = False,
    ) -> List[Dict]:
        """Hybrid retrieval across multiple documents with Jina reranking."""
        t0 = time.time()
        print("[计时] [HybridRetriever] 开始多文档向量检索 ...")
        vector_results = self.vector_retriever.retrieve_by_sha1_list(
            sha1_list=sha1_list,
            query=query,
            top_n=reranking_sample_size,
            return_parent_pages=return_parent_pages,
        )
        t1 = time.time()
        print(f"[计时] [HybridRetriever] 向量检索耗时: {t1-t0:.2f} 秒")
        print("[计时] [HybridRetriever] 开始Jina重排 ...")
        reranked_results = self.reranker.rerank_documents(
            query=query,
            documents=vector_results,
            top_n=top_n,
        )
        t2 = time.time()
        print(f"[计时] [HybridRetriever] Jina重排耗时: {t2-t1:.2f} 秒")
        print(f"[计时] [HybridRetriever] 总耗时: {t2-t0:.2f} 秒")
        return reranked_results
