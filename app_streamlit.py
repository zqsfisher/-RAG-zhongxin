import streamlit as st
from pathlib import Path
from src.pipeline import Pipeline, routing_config
from src import prompts
import json

# 使用脚本所在目录定位项目根目录，避免相对路径问题
SCRIPT_DIR = Path(__file__).parent.resolve()
root_path = SCRIPT_DIR / "data" / "stock_data"
pipeline = Pipeline(root_path, run_config=routing_config)

EXAMPLE_QUESTIONS_PATH = root_path / "example_answers.json"
with open(EXAMPLE_QUESTIONS_PATH, "r", encoding="utf-8") as f:
    EXAMPLE_DATA = json.load(f)

EXAMPLE_QUESTIONS = [q["question"] for q in EXAMPLE_DATA["questions"]]

# ==================== 提示词模版数据 ====================
PROMPT_TEMPLATES = {
    "string": {
        "label": "📝 string — 文本描述类",
        "system_prompt": prompts.AnswerWithRAGContextStringPrompt.system_prompt,
        "user_prompt_template": prompts.AnswerWithRAGContextStringPrompt.user_prompt,
        "example": prompts.AnswerWithRAGContextStringPrompt.example,
        "description": "适用于需要文本描述回答的问题，如总结、分析、解释等。",
    },
    "number": {
        "label": "🔢 number — 数值指标类",
        "system_prompt": prompts.AnswerWithRAGContextNumberPrompt.system_prompt,
        "user_prompt_template": prompts.AnswerWithRAGContextNumberPrompt.user_prompt,
        "example": prompts.AnswerWithRAGContextNumberPrompt.example,
        "description": "适用于需要精确数值回答的问题，如营收、利润率、增长率等。严格匹配指标定义，不允许计算推导。",
    },
    "boolean": {
        "label": "✅ boolean — 是/否判断类",
        "system_prompt": prompts.AnswerWithRAGContextBooleanPrompt.system_prompt,
        "user_prompt_template": prompts.AnswerWithRAGContextBooleanPrompt.user_prompt,
        "example": prompts.AnswerWithRAGContextBooleanPrompt.example,
        "description": "适用于需要 True/False 回答的问题，如是否发生某事、是否包含某项等。",
    },
    "names": {
        "label": "📋 names — 名单/实体类",
        "system_prompt": prompts.AnswerWithRAGContextNamesPrompt.system_prompt,
        "user_prompt_template": prompts.AnswerWithRAGContextNamesPrompt.user_prompt,
        "example": prompts.AnswerWithRAGContextNamesPrompt.example,
        "description": "适用于需要返回名单/列表的问题，如高管姓名、产品名称、子公司等。",
    },
}

st.set_page_config(page_title="中芯国际 · 智能研报问答", page_icon="🤖", layout="wide")

# ==================== 顶部 Banner ====================
st.markdown("""
<div style='background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); padding: 24px; border-radius: 16px; text-align: center; margin-bottom: 20px;'>
    <h1 style='color: white; margin: 0; font-size: 32px;'>🏭 中芯国际 · 智能研报问答</h1>
    <p style='color: rgba(255,255,255,0.85); font-size: 15px; margin-top: 8px;'>
    基于多份券商研报 + 年报 + 调研纪要 | 文档摘要路由 → FAISS 检索 → 本地 Jina Reranker 重排 → LLM 推理
    </p>
</div>
""", unsafe_allow_html=True)

# ==================== 侧边栏 ====================
with st.sidebar:
    st.header("📋 查询设置")

    # 快捷问题按钮
    st.markdown("**🔥 示例问题（由易到难）：**")
    selected_example = None
    for i, q in enumerate(EXAMPLE_QUESTIONS):
        level = ["🟢 简单", "🟡 中等", "🔴 困难"][i // 2] if i < 6 else "🔴 困难"
        label = f"{level} {q[:35]}..."
        if st.button(label, key=f"example_{i}", use_container_width=True):
            selected_example = q
            st.session_state["question"] = q

    st.divider()

    # 问题类型选择
    st.markdown("**📐 问题类型 (kind)：**")
    kind_options = {
        "string": "📝 string — 文本描述类",
        "number": "🔢 number — 数值指标类",
        "boolean": "✅ boolean — 是/否判断类",
        "names":  "📋 names  — 名单/实体类",
    }
    selected_kind = st.selectbox(
        "选择匹配你问题的类型",
        options=list(kind_options.keys()),
        format_func=lambda x: kind_options[x],
        index=0
    )

    # 输入框
    default_q = st.session_state.get("question", EXAMPLE_QUESTIONS[0])
    user_question = st.text_area("✏️ 输入你的问题", default_q, height=100)

    col1, col2 = st.columns([2, 1])
    with col1:
        submit_btn = st.button("🚀 生成答案", use_container_width=True)
    with col2:
        clear_btn = st.button("清空", use_container_width=True)
        if clear_btn:
            st.session_state["question"] = ""
            st.rerun()

    st.divider()
    st.caption("模型：qwen-turbo-latest | 重排：jina-reranker-v2 (本地)")
    st.caption(f"知识库：{len(list(pipeline.paths.documents_dir.glob('*.json')))} 份中芯国际文档")

# ==================== 主区域 ====================
if submit_btn and user_question.strip():
    with st.spinner("🔍 正在分析问题 → 路由文档 → 检索 → 重排 → 生成答案..."):
        try:
            result = pipeline.answer_single_question(user_question, kind=selected_kind, with_trace=True)
            answer, trace = result if isinstance(result, tuple) else (result, {})

            # 解析答案
            if isinstance(answer, dict):
                answer_dict = answer
            elif isinstance(answer, str):
                try:
                    answer_dict = json.loads(answer.strip().lstrip("```json").rstrip("```").strip())
                except json.JSONDecodeError:
                    answer_dict = {"final_answer": answer, "reasoning_summary": "", "relevant_pages": []}
            else:
                answer_dict = {"final_answer": str(answer), "reasoning_summary": "", "relevant_pages": []}

            final_answer = answer_dict.get("final_answer", "-")
            step_by_step = answer_dict.get("step_by_step_analysis", "")
            reasoning_summary = answer_dict.get("reasoning_summary", "")
            relevant_pages = answer_dict.get("relevant_pages", [])

            # 结果展示
            tab1, tab2, tab3, tab4 = st.tabs(["📝 最终答案", "🧠 推理过程", "📄 引用来源", "🔍 RAG 流程追踪"])

            with tab1:
                st.markdown(f"""
                <div style='background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
                            padding: 24px; border-radius: 12px; border-left: 5px solid #667eea;'>
                    <p style='font-size: 17px; line-height: 1.8; color: #1a1a2e; margin: 0;'>{final_answer}</p>
                </div>
                """, unsafe_allow_html=True)

                matched = next((q for q in EXAMPLE_DATA["questions"] if q["question"] == user_question), None)
                if matched:
                    with st.expander("📌 查看参考要点"):
                        st.info(matched["reference_key_points"])

            with tab2:
                if step_by_step:
                    st.markdown("##### 分步推理")
                    st.info(step_by_step)
                if reasoning_summary:
                    st.markdown("##### 推理摘要")
                    st.success(reasoning_summary)
                if not step_by_step and not reasoning_summary:
                    st.caption("（模型未返回推理过程）")

            with tab3:
                references = answer_dict.get("references", [])
                if references:
                    st.write(f"涉及 {len(references)} 条引用：")
                    for i, ref in enumerate(references):
                        fname = ref.get("file_name", "?")
                        lines = ref.get("chunk_lines", [0, 0])
                        sha1 = ref.get("pdf_sha1", "")[:12]
                        st.markdown(
                            f"**{i+1}.** 📄 `{fname}`  · 行 {lines[0]}–{lines[1]}  · `{sha1}...`"
                        )
                elif relevant_pages:
                    st.write(f"涉及 {len(relevant_pages)} 个行号范围：")
                    st.json(relevant_pages)
                else:
                    st.caption("（未返回具体引用来源）")

            # ==================== RAG 流程追踪 Tab ====================
            with tab4:
                if trace and trace.get("steps"):
                    st.markdown("### 🎯 完整 RAG 流程")
                    for step in trace["steps"]:
                        step_name = step["step"]
                        with st.expander(f"**{step_name}** — {step.get('detail', '')}", expanded=(step_name.startswith("1"))):
                            if "routed_documents" in step:
                                st.caption(f"路由分析理由：")
                                st.text(step.get("router_reasoning", "")[:800])
                                st.caption(f"选中的文档（共 {step.get('all_documents_count', '?')} 篇候选）：")
                                for d in step["routed_documents"]:
                                    st.markdown(f"- 📄 {d['institution']} | {d['type']} | `{d['sha1']}...`")

                            if "top_results" in step:
                                st.caption(f"Top-{len(step['top_results'])} 重排结果：")
                                for r in step["top_results"]:
                                    src = r.get("source", "?")
                                    st.markdown(f"- {src} [score={r['score']:.4f}] _{r['text_preview']}_")

                            if "final_prompt" in step or step_name == "3. 拼接最终提示词":
                                pass  # 下面单独展示

                    # 最终提示词独立展示
                    if trace.get("final_prompt"):
                        st.divider()
                        st.markdown("### 📨 发送给 LLM 的最终提示词")

                        prompt_data = trace["final_prompt"]
                        current_kind = trace.get("schema", "string")

                        col_a, col_b, col_c = st.columns(3)
                        with col_a:
                            st.metric("System Prompt", f"{len(prompt_data['system_prompt'])} 字符")
                        with col_b:
                            st.metric("User Prompt", f"{len(prompt_data['user_prompt'])} 字符")
                        with col_c:
                            st.metric("问题类型", f"`{current_kind}`")

                        with st.expander("📋 System Prompt（当前类型的系统指令）", expanded=True):
                            st.code(prompt_data["system_prompt"], language="markdown")

                        with st.expander("📋 User Prompt（{context} + {question} 拼装后）", expanded=True):
                            st.code(prompt_data["user_prompt"], language="markdown")

                    # 提示词模版参考（所有类型）
                    st.divider()
                    st.markdown("### 📐 提示词模版参考")

                    # 当前类型高亮
                    current_kind = trace.get("schema", selected_kind) if trace else selected_kind
                    if current_kind in PROMPT_TEMPLATES:
                        t = PROMPT_TEMPLATES[current_kind]
                        st.info(f"**当前类型：{t['label']}** — {t['description']}")

                    # 全部类型展示
                    for kind, t in PROMPT_TEMPLATES.items():
                        is_current = (kind == current_kind)
                        title = f"{t['label']} {'⭐ 当前' if is_current else ''}"
                        with st.expander(title, expanded=is_current):
                            st.caption(t["description"])
                            st.markdown("**System Prompt:**")
                            st.code(t["system_prompt"], language="markdown")
                            st.markdown("**User Prompt 模板:**")
                            st.code(t["user_prompt_template"], language="markdown")

                    # 检索上下文原始数据
                    if trace.get("retrieval_context"):
                        with st.expander("📦 检索到的原始上下文文本", expanded=False):
                            ctx = trace["retrieval_context"]
                            st.text_area("RAG Context", ctx, height=300)

                    # LLM 响应元数据
                    if trace.get("llm_response"):
                        st.divider()
                        st.markdown("### 📊 LLM 调用信息")
                        st.json(trace["llm_response"])

                else:
                    st.info("当前查询未启用流程追踪。")

            # 原始响应
            with st.expander("🔧 原始响应 JSON (调试)", expanded=False):
                st.json(answer_dict)

        except Exception as e:
            import traceback
            st.error(f"❌ 生成答案时出错: {e}")
            with st.expander("详细错误信息"):
                st.code(traceback.format_exc())

elif not submit_btn:
    # ==================== 欢迎页 ====================
    st.markdown("### 👋 欢迎使用中芯国际智能研报问答系统")
    st.markdown("""
    本系统基于 **RAG（检索增强生成）** 技术，覆盖中芯国际相关的 **9 份** 专业文档：

    | 类型 | 来源 |
    |------|------|
    | 券商研报 | 上海证券、东方证券、中原证券、光大证券、兴业证券、华泰证券、国信证券 |
    | 年度报告 | 中芯国际 2024 年年度报告 |
    | 调研纪要 | 多机构联合调研纪要 |

    **工作流程：** 文档摘要路由 → 向量检索 → Jina 本地重排 → LLM 推理
    """)

    st.divider()

    st.markdown("### 🔥 试试这些问题（由易到难）")
    cols = st.columns(3)
    levels = [
        ("🟢 简单 · 事实查找", ["单文档直接可答", "单一指标/数据"]),
        ("🟡 中等 · 跨文档综合", ["需综合多文档信息", "对比或归纳"]),
        ("🔴 困难 · 因果推理", ["需多步推理", "政策/趋势分析"]),
    ]
    for i, (title, descs) in enumerate(levels):
        with cols[i]:
            st.markdown(f"**{title}**")
            for d in descs:
                st.caption(f"• {d}")
            start_idx = i * 2
            for j in range(start_idx, min(start_idx + 2, len(EXAMPLE_QUESTIONS))):
                q = EXAMPLE_QUESTIONS[j]
                st.markdown(f"*{j+1}.* {q[:50]}...")

    st.divider()
    st.caption("💡 在左侧选择示例问题或输入自定义问题，点击「生成答案」开始分析。")
    st.caption("⚡ 首次加载需 2-5 秒初始化本地 Jina Reranker 模型。")