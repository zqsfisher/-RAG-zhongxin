# RAG 企业知识库 — 改动记录 & 流程图

> 项目路径：`18-项目实战：企业知识库/RAG-cy`  
> 知识库：9 份中芯国际文档（7 份券商研报 + 1 份年报 + 1 份调研纪要）  
> 最后更新：2026-05-29

---

## 一、代码改动总览

### 1. 重排器：LLM → Jina 本地模型

| 文件 | 改动 |
|------|------|
| `src/reranking.py` | 删除 `LLMReranker` 类（~180行），保留 `JinaReranker`。改为本地 `sentence-transformers` + `CrossEncoder` 加载 `models/jina-reranker/`，新增 `max_length=512` 截断参数以加速 CPU 推理 |
| `src/retrieval.py` | `HybridRetriever` 从 `LLMReranker` 切换为 `JinaReranker`，自动查找项目根目录 `models/jina-reranker/` |

**优势**：无需网络 → 无超时风险；`max_length=512` 将 CPU 推理从 250s 降至 7~20s。

### 2. FAISS 向量库：中文路径兼容

| 文件 | 改动 |
|------|------|
| `src/retrieval.py` | `_load_dbs()` 改为 Python `open().read()` → `np.frombuffer()` → `faiss.deserialize_index()`，完全绕过 FAISS 的 C++ 文件 I/O |
| `src/ingestion.py` | 写入端原本就是 `faiss.serialize_index()` + Python `open().write()`，无需修改 |

**原因**：FAISS 的 C++ 底层 `read_index()` 不支持含中文的 Windows 路径（如 `E:\...\18-项目实战：企业知识库\...`）。

### 3. 文档路由：Regex 回退

| 文件 | 改动 |
|------|------|
| `src/document_router.py` | 新增 `_extract_indices_from_text()` 方法，从 LLM 返回的自然语言文本中正则提取文档编号。`route_question()` 返回值改为 `(sha1_list, reasoning)` 元组 |

**原因**：`qwen-turbo-latest` 不支持原生 JSON 模式，返回的自然语言无法被解析为 `RouteSelection` schema。正则匹配 `**答案：1, 4, 8**` 等格式作为回退。

### 4. 检索去重 Bug 修复

| 文件 | 改动 |
|------|------|
| `src/retrieval.py` | `retrieve_by_sha1_list()` 中去重键从 `(sha1, page)` 改为文本内容 MD5 哈希 |

**原因**：chunks 缺少 `page` 字段，默认值全为 0，导致同文档 6 个 chunk 被去重成 1 个。

### 5. 流程追踪（Trace）

| 文件 | 改动 |
|------|------|
| `src/questions_processing.py` | 新增 `enable_trace` 参数和 `self.trace` 字典，在 `get_answer_with_routing()` 中记录路由→检索→提示词→LLM 响应 |
| `src/pipeline.py` | `answer_single_question()` 新增 `with_trace=True`，返回 `(answer, trace)` |
| `src/document_router.py` | `route_question()` 返回值改为元组，包含路由推理过程 |

### 6. 前端重构

| 文件 | 改动 |
|------|------|
| `app_streamlit.py` | 全新 UI：渐变 Banner、示例问题快捷按钮（🟢🟡🔴 标注难度）、4 个结果 Tab（最终答案/推理过程/引用来源/RAG 流程追踪）。流程追踪 Tab 展示路由结果、检索排名、最终提示词全文 |
| `data/stock_data/example_answers.json` | 新增 6 道由易到难的示例问题及参考要点 |

### 7. 参数重命名

| 旧名 | 新名 | 涉及文件 |
|------|------|----------|
| `llm_reranking` | `use_reranking` | `pipeline.py`, `questions_processing.py` |
| `llm_reranking_sample_size` | `reranking_sample_size` | `pipeline.py`, `questions_processing.py`, `retrieval.py` |

### 8. 环境依赖

```bash
pip install sentence-transformers einops
```

模型文件：从 https://huggingface.co/jinaai/jina-reranker-v2-base-multilingual 下载到 `models/jina-reranker/`，删除 `pytorch_model.bin`（只保留 `model.safetensors`）以避免 meta tensor 错误。

---

## 二、RAG 完整流程图

```
┌─────────────────────────────────────────────────────────────────────┐
│                       用户输入问题                                    │
│         例："中原证券对中芯国际2025年一季度的投资评级是什么？"           │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Step 1  文档路由 (DocumentRouter)                                    │
│  ─────────────────────────────────────                               │
│  • 加载 document_summaries.json（9 篇文档的摘要+关键主题+时间范围）      │
│  • 组装 Prompt → 调用 qwen-turbo-latest 选出最相关文档                 │
│  • Regex 回退：从 LLM 文本中提取 "**答案：3, 2, 4**" 模式              │
│  • 输出：2~3 个 sha1（如 0670cee, 9cc35c8, 9cb72bb）                  │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Step 2  向量检索 (VectorRetriever)                                   │
│  ─────────────────────────────────────                               │
│  • 对每个选中的 sha1，加载对应 FAISS 索引                              │
│  • Python open() + np.frombuffer() + faiss.deserialize_index()       │
│    （绕过 FAISS C++ 文件 I/O 的中文路径兼容问题）                      │
│  • query → dashscope text-embedding-v1 → 向量                        │
│  • FAISS IndexFlatIP 内积搜索 → 每个文档返回 top-k 个片段             │
│  • 输出：~15 个候选片段（合并后去重）                                  │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Step 3  Jina 重排 (JinaReranker, 本地模型)                           │
│  ─────────────────────────────────────                               │
│  • 加载 models/jina-reranker/ (CrossEncoder, max_length=512)         │
│  • 对每个 (query, chunk_text) 对打分                                 │
│  • 按相关性分数降序排列 → 取 top_n=10                                 │
│  • 输出：10 个重排后的片段（含 relevance_score）                       │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Step 4  拼接最终提示词                                               │
│  ─────────────────────────────────────                               │
│  System Prompt (~570 字符):                                          │
│    "你是一个RAG问答系统。仅基于检索到的上下文回答..."                    │
│                                                                       │
│  User Prompt (~6000 字符):                                           │
│    "以下是上下文:                                                     │
│     \"\"\"                                                           │
│     Text retrieved from page 0: ...                                  │
│     ---                                                              │
│     Text retrieved from page 0: ...                                  │
│     \"\"\"                                                           │
│     以下是问题：                                                      │
│     \"中原证券对中芯国际2025年一季度的投资评级是什么？\"                │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Step 5  LLM 推理 (qwen-turbo-latest, DashScope)                     │
│  ─────────────────────────────────────                               │
│  • 发送 System + User Prompt 到 qwen-turbo-latest                    │
│  • 结构化输出：step_by_step_analysis / reasoning_summary /            │
│                relevant_pages / final_answer                          │
│  • 输出：{"final_answer": "买入", "relevant_pages": [0], ...}         │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       返回最终答案                                    │
│                  "中原证券维持'买入'评级。"                            │
└─────────────────────────────────────────────────────────────────────┘
```

### 关键技术栈

| 层级 | 技术 |
|------|------|
| 文档解析 | MinerU PDF → Markdown |
| 文本分块 | 自定义 `TextSplitter`（按 Markdown 标题分块） |
| 向量嵌入 | DashScope `text-embedding-v1` |
| 向量索引 | FAISS `IndexFlatIP`（内积/余弦距离） |
| 文档路由 | qwen-turbo-latest + 文档摘要 JSON |
| 重排序 | Jina Reranker v2 Multilingual（本地 CrossEncoder, `max_length=512`） |
| 答案生成 | qwen-turbo-latest（DashScope API） |
| 前端 | Streamlit |

---

## 三、运行方式

```powershell
conda activate rag
cd "E:\Python\llm\AI_course\18-项目实战：企业知识库\RAG-cy"
$env:PYTHONPATH="."
streamlit run app_streamlit.py
```

### 首次运行前需完成的步骤（已配置好则跳过）：

```powershell
# 1. 安装依赖
pip install sentence-transformers einops

# 2. 下载 Jina Reranker 模型到 models/jina-reranker/
#    从 https://huggingface.co/jinaai/jina-reranker-v2-base-multilingual
#    删除 pytorch_model.bin，只保留 model.safetensors

# 3. 确保 .env 中有:
#    DASHSCOPE_API_KEY=sk-xxx
#    JINA_API_KEY=jina_xxx  (本地模型不需要，可选)

# 4. 生成向量库和摘要（如果还没有）
python -c "
from pathlib import Path; from src.pipeline import Pipeline, routing_config
p = Pipeline(Path('data/stock_data'), run_config=routing_config)
p.chunk_reports()
p.create_vector_dbs()
p.generate_document_summaries()
"
```
