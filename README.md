# 🏭 中芯国际 · 智能研报问答 (RAG)

基于 RAG（检索增强生成）的金融研报智能问答系统，覆盖 9 份中芯国际相关文档。

**完整流程：** 文档摘要路由 → FAISS 向量检索 → 本地 Jina Reranker 重排 → LLM 推理

## 快速开始

```bash
# 1. 环境
conda create -n rag python=3.11
conda activate rag
pip install -r requirements.txt

# 2. 下载 Jina Reranker 模型
# 从 https://huggingface.co/jinaai/jina-reranker-v2-base-multilingual
# 下载所有文件到 models/jina-reranker/，删除 pytorch_model.bin

# 3. 配置 API Key
# 创建 .env 文件：
#   DASHSCOPE_API_KEY=sk-xxx

# 4. 生成向量库（首次必须）
python -c "
from pathlib import Path; from src.pipeline import Pipeline, routing_config
p = Pipeline(Path('data/stock_data'), run_config=routing_config)
p.chunk_reports(); p.create_vector_dbs(); p.generate_document_summaries()
"

# 5. 启动 Web UI
$env:PYTHONPATH='.'
streamlit run app_streamlit.py

# 或命令行批量处理
python src/pipeline.py
```

## 项目结构

```
├── app_streamlit.py          # Streamlit Web UI
├── src/
│   ├── pipeline.py           # 主流程编排 & 运行配置
│   ├── reranking.py          # Jina Reranker（本地模型）
│   ├── retrieval.py          # FAISS 向量检索 + HybridRetriever
│   ├── ingestion.py          # 向量库 & BM25 生成
│   ├── questions_processing.py  # 问题处理 & 答案生成
│   ├── document_router.py    # 文档摘要路由
│   ├── document_summarizer.py   # 文档摘要生成
│   ├── text_splitter.py      # Markdown 分块
│   ├── prompts.py            # 提示词模板
│   └── api_requests.py       # LLM API 调用
├── data/stock_data/
│   ├── subset.csv            # 文档索引
│   ├── questions.json         # 示例问题
│   ├── example_answers.json   # 示例答案参考
│   └── databases/
│       └── document_summaries.json  # 文档摘要
├── models/jina-reranker/     # （需自行下载）
├── RAG_CHANGELOG.md          # 改动记录 & 流程图
└── requirements.txt
```

## 技术栈

| 层级 | 技术 |
|------|------|
| PDF 解析 | MinerU → Markdown |
| 文本分块 | 自定义 TextSplitter（按行分块，30行/块，5行重叠） |
| 向量嵌入 | DashScope `text-embedding-v1`（1024维） |
| 向量索引 | FAISS `IndexFlatIP`（内积距离） |
| 文档路由 | qwen-turbo-latest + 文档摘要 JSON |
| 重排序 | Jina Reranker v2 Multilingual（本地 CrossEncoder） |
| 答案生成 | qwen-turbo-latest |
| 前端 | Streamlit |

## 提示词类型

| kind | 说明 | 输出格式 |
|------|------|----------|
| `string` | 文本描述类（总结、分析） | 自由文本 |
| `number` | 数值指标类（营收、利润率） | 精确数值 |
| `boolean` | 是/否判断类 | True/False |
| `names` | 名单/实体类（高管、产品） | 字符串列表 |

## License

MIT