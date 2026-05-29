import json
from pathlib import Path
from typing import Optional
from datetime import datetime

from src.api_requests import APIProcessor
from src.prompts import DocumentSummaryPrompt


class DocumentSummarizer:
    def __init__(
        self,
        documents_dir: Path,
        api_provider: str = "dashscope",
        model: str = "qwen-turbo-latest",
    ):
        self.documents_dir = Path(documents_dir)
        self.api_processor = APIProcessor(provider=api_provider)
        self.model = model

    def _build_document_text(self, document: dict, max_chars: int = 4000) -> str:
        """Concatenate the leading chunks of a document up to max_chars."""
        chunks = document.get("content", {}).get("chunks", [])
        text = ""
        for chunk in chunks:
            chunk_text = chunk.get("text", "")
            if len(text) + len(chunk_text) > max_chars:
                remaining = max_chars - len(text)
                if remaining > 0:
                    text += chunk_text[:remaining]
                break
            text += chunk_text
        return text

    def _summarize_document(self, sha1: str, document: dict) -> Optional[dict]:
        """Generate a structured summary for a single document."""
        text = self._build_document_text(document)
        if not text.strip():
            print(f"Warning: empty text for document {sha1}, skipping")
            return None

        try:
            result = self.api_processor.send_message(
                model=self.model,
                temperature=0.1,
                system_content=DocumentSummaryPrompt.system_prompt,
                human_content=DocumentSummaryPrompt.user_prompt.format(
                    document_text=text
                ),
                is_structured=True,
                response_format=DocumentSummaryPrompt.SummarySchema,
            )
            return {
                "file_name": document.get("metainfo", {}).get("file_name", ""),
                "company_name": document.get("metainfo", {}).get("company_name", ""),
                "summary": result.get("summary", ""),
                "key_topics": result.get("key_topics", []),
                "time_period": result.get("time_period", ""),
                "document_type": result.get("document_type", ""),
                "issuing_institution": result.get("issuing_institution", ""),
                "investment_rating": result.get("investment_rating", "无"),
                "generated_at": datetime.now().isoformat(),
            }
        except Exception as e:
            print(f"Error summarizing document {sha1}: {e}")
            return None

    def generate_all_summaries(self, output_path: Path) -> dict:
        """Generate summaries for all chunked report JSON files.

        Skips documents already present in existing summaries (incremental update).
        """
        existing = {}
        if output_path.exists():
            with open(output_path, "r", encoding="utf-8") as f:
                existing = json.load(f).get("documents", {})

        all_report_paths = sorted(self.documents_dir.glob("*.json"))
        new_summaries = dict(existing)

        for report_path in all_report_paths:
            with open(report_path, "r", encoding="utf-8") as f:
                document = json.load(f)

            sha1 = document.get("metainfo", {}).get("sha1", "")
            if not sha1:
                print(f"Warning: no sha1 in {report_path.name}, skipping")
                continue

            if sha1 in new_summaries:
                print(f"Summary already exists for {report_path.name}, skipping")
                continue

            print(f"Generating summary for {report_path.name}...")
            summary_entry = self._summarize_document(sha1, document)
            if summary_entry:
                new_summaries[sha1] = summary_entry
                print(f"  -> Done: {summary_entry.get('issuing_institution', '')} - {summary_entry.get('document_type', '')}")

        result = {"documents": new_summaries}
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        print(f"Saved {len(new_summaries)} summaries to {output_path}")
        return result
