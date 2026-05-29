import json
import re
from pathlib import Path
from typing import List

from src.api_requests import APIProcessor
from src.prompts import DocumentRoutingPrompt


class DocumentRouter:
    def __init__(
        self,
        summaries_path: Path,
        api_provider: str = "dashscope",
        model: str = "qwen-turbo-latest",
    ):
        self.summaries_path = Path(summaries_path)
        if not self.summaries_path.exists():
            raise FileNotFoundError(
                f"document_summaries.json not found at {summaries_path}. "
                "Run generate_document_summaries() first."
            )

        with open(self.summaries_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.documents = data.get("documents", {})

        self.sha1_list = list(self.documents.keys())
        self.api_processor = APIProcessor(provider=api_provider)
        self.model = model

    def _build_summaries_context(self) -> str:
        """Format all document summaries as a numbered list for the LLM."""
        parts = []
        for i, sha1 in enumerate(self.sha1_list, start=1):
            doc = self.documents[sha1]
            parts.append(
                f"文档 {i}:\n"
                f"  - 名称: {doc.get('file_name', '')}\n"
                f"  - 类型: {doc.get('document_type', '')}\n"
                f"  - 机构: {doc.get('issuing_institution', '')}\n"
                f"  - 时间范围: {doc.get('time_period', '')}\n"
                f"  - 关键主题: {', '.join(doc.get('key_topics', []))}\n"
                f"  - 评级: {doc.get('investment_rating', '无')}\n"
                f"  - 摘要: {doc.get('summary', '')}"
            )
        return "\n\n".join(parts)

    def _extract_indices_from_text(self, text: str, max_documents: int) -> List[int]:
        """从 LLM 返回的文本中提取文档编号（当 JSON 解析失败时的回退方案）。"""
        indices = []
        # 匹配 "答案：1, 2, 3" 或 "**答案：1, 4, 8**" 或 "答案: 9,7,6" 等格式
        patterns = [
            r'\*\*答案[：:]\s*([\d,\s]+)\*\*',
            r'答案[：:]\s*([\d,\s]+)',
            r'推荐[：:]\s*([\d,\s]+)',
            r'选择[：:]\s*([\d,\s]+)',
            r'文档\s*([\d]+)',
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                nums_str = match.group(1)
                # 提取所有数字
                found = [int(n) for n in re.findall(r'\d+', nums_str)]
                if found:
                    indices = found
                    break

        # 去重、限制数量、过滤有效范围
        seen = set()
        valid = []
        for idx in indices:
            if idx not in seen and 1 <= idx <= len(self.sha1_list):
                seen.add(idx)
                valid.append(idx)
        return valid[:max_documents]

    def route_question(self, question: str, max_documents: int = 3) -> tuple:
        """Select the most relevant documents for a given question.

        Returns a tuple of (list of sha1 values, router reasoning text).
        Falls back to returning all sha1s if routing produces an empty result.
        """
        if not self.documents:
            raise ValueError("No document summaries available for routing.")

        if len(self.sha1_list) <= max_documents:
            return list(self.sha1_list), "文档总数不超过路由上限，全部选用"

        instruction = DocumentRoutingPrompt.instruction.format(
            max_documents=max_documents
        )
        summaries_text = self._build_summaries_context()

        try:
            result = self.api_processor.send_message(
                model=self.model,
                temperature=0.1,
                system_content=instruction,
                human_content=DocumentRoutingPrompt.user_prompt.format(
                    summaries=summaries_text, question=question
                ),
                is_structured=True,
                response_format=DocumentRoutingPrompt.RouteSelection,
            )
        except Exception as e:
            print(f"Document routing failed: {e}, falling back to all documents")
            return list(self.sha1_list), f"路由异常: {e}"

        indices = result.get("selected_indices", [])
        reasoning = result.get("reasoning", "")
        raw_text = result.get("final_answer", "") or str(result)

        # 如果 JSON 解析失败，从 LLM 返回的文本中提取文档编号
        if not indices:
            indices = self._extract_indices_from_text(raw_text, max_documents)
            if indices:
                reasoning = raw_text[:500]  # 用原始文本作为推理说明
                print(f"Router (fallback text parse): extracted indices {indices}")

        print(f"Router reasoning: {reasoning}")

        # Convert 1-based indices to sha1 list
        selected = []
        for idx in indices:
            if 1 <= idx <= len(self.sha1_list):
                selected.append(self.sha1_list[idx - 1])

        if not selected:
            print("Router returned no documents, falling back to all documents")
            return list(self.sha1_list), reasoning or "未选出特定文档，回退到全部文档"

        print(f"Router selected: {selected}")
        return selected, reasoning

    def get_document_meta(self, sha1: str) -> dict:
        """Get metadata for a document by sha1."""
        return self.documents.get(sha1, {})
