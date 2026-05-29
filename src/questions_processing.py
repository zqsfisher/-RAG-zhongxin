import json
from typing import Union, Dict, List, Optional
import re
from pathlib import Path
from src.retrieval import VectorRetriever, HybridRetriever
from src.api_requests import APIProcessor
from tqdm import tqdm
import pandas as pd
import threading
import concurrent.futures
import time


class QuestionsProcessor:
    def __init__(
        self,
        vector_db_dir: Union[str, Path] = './vector_dbs',
        documents_dir: Union[str, Path] = './documents',
        questions_file_path: Optional[Union[str, Path]] = None,
        new_challenge_pipeline: bool = False,
        subset_path: Optional[Union[str, Path]] = None,
        parent_document_retrieval: bool = False,  # 是否启用父文档检索
        use_reranking: bool = False,              # 是否启用Jina重排
        reranking_sample_size: int = 30,
        top_n_retrieval: int = 10,
        parallel_requests: int = 10,
        api_provider: str = "dashscope", # openai
        answering_model: str = "qwen-turbo-latest", # gpt-4o-2024-08-06
        full_context: bool = False,
        use_document_routing: bool = False,
        summaries_path: Optional[Union[str, Path]] = None,
        enable_trace: bool = False,                # 是否启用流程追踪
    ):
        # 初始化问题处理器，配置检索、模型、并发等参数
        self.questions = self._load_questions(questions_file_path)
        self.documents_dir = Path(documents_dir)
        self.vector_db_dir = Path(vector_db_dir)
        self.subset_path = Path(subset_path) if subset_path else None

        self.new_challenge_pipeline = new_challenge_pipeline
        self.return_parent_pages = parent_document_retrieval
        self.use_reranking = use_reranking
        self.reranking_sample_size = reranking_sample_size
        self.top_n_retrieval = top_n_retrieval
        self.answering_model = answering_model
        self.parallel_requests = parallel_requests
        self.api_provider = api_provider
        self.openai_processor = APIProcessor(provider=api_provider)
        self.full_context = full_context
        self.use_document_routing = use_document_routing
        self.summaries_path = Path(summaries_path) if summaries_path else None
        self.enable_trace = enable_trace
        self.trace = {}  # 存储完整流程追踪数据

        self.answer_details = []
        self.detail_counter = 0
        self._lock = threading.Lock()

    def _load_questions(self, questions_file_path: Optional[Union[str, Path]]) -> List[Dict[str, str]]:
        # 加载问题文件，返回问题列表
        if questions_file_path is None:
            return []
        with open(questions_file_path, 'r', encoding='utf-8') as file:
            return json.load(file)

    def _format_retrieval_results(self, retrieval_results) -> str:
        """将检索结果格式化为RAG上下文字符串"""
        if not retrieval_results:
            return ""
        
        context_parts = []
        for result in retrieval_results:
            page_number = result['page']
            text = result['text']
            context_parts.append(f'Text retrieved from page {page_number}: \n"""\n{text}\n"""')
            
        return "\n\n---\n\n".join(context_parts)

    def _extract_references(self, pages_list: list, company_name: str) -> list:
        # 根据公司名和页码列表，提取引用信息
        if self.subset_path is None:
            raise ValueError("subset_path is required for new challenge pipeline when processing references.")
        # 优先尝试 utf-8，失败则尝试 gbk
        try:
            self.companies_df = pd.read_csv(self.subset_path, encoding='utf-8')
        except UnicodeDecodeError:
            print('警告：subset.csv 不是 utf-8 编码，自动尝试 gbk 编码...')
            self.companies_df = pd.read_csv(self.subset_path, encoding='gbk')

        # Find the company's SHA1 from the subset CSV
        matching_rows = self.companies_df[self.companies_df['company_name'] == company_name]
        if matching_rows.empty:
            company_sha1 = ""
        else:
            company_sha1 = matching_rows.iloc[0]['sha1']

        refs = []
        for page in pages_list:
            refs.append({"pdf_sha1": company_sha1, "page_index": page})
        return refs

    def _validate_page_references(self, claimed_pages: list, retrieval_results: list, min_pages: int = 2, max_pages: int = 8) -> list:
        """
        校验LLM答案中引用的页码是否真实存在于检索结果中。
        返回 dict 列表，每项包含 sha1, file_name, chunk_lines。
        若不足最小页数，则补充检索结果中的top结果。
        """
        if claimed_pages is None:
            claimed_pages = []

        # 用 chunk 起始行号作为标识
        retrieved_line_starts = [r.get("page", 0) for r in retrieval_results]

        # 验证：claimed page 是否存在于检索结果中
        validated_indices = []
        for page in claimed_pages:
            if page in retrieved_line_starts:
                validated_indices.append(retrieved_line_starts.index(page))

        # 构建已验证的引用
        validated = []
        for idx in validated_indices:
            r = retrieval_results[idx]
            validated.append({
                "sha1": r.get("sha1", ""),
                "file_name": r.get("file_name", ""),
                "chunk_lines": r.get("chunk_lines", [0, 0]),
            })

        if len(validated) < len(claimed_pages):
            removed = set(claimed_pages) - {r.get("chunk_lines", [0,0])[0] for r in validated}
            print(f"Warning: Removed {len(removed)} hallucinated page references: {removed}")

        # 不足最小数量，用 top 结果补充
        if len(validated) < min_pages and retrieval_results:
            existing_keys = {(v["sha1"], v["chunk_lines"][0]) for v in validated}
            for r in retrieval_results:
                key = (r.get("sha1", ""), r.get("chunk_lines", [0,0])[0])
                if key not in existing_keys:
                    validated.append({
                        "sha1": r.get("sha1", ""),
                        "file_name": r.get("file_name", ""),
                        "chunk_lines": r.get("chunk_lines", [0, 0]),
                    })
                    existing_keys.add(key)
                    if len(validated) >= min_pages:
                        break

        if len(validated) > max_pages:
            print(f"Trimming references from {len(validated)} to {max_pages}")
            validated = validated[:max_pages]

        return validated

    def get_answer_for_company(self, company_name: str, question: str, schema: str) -> dict:
        # 针对单个公司，检索上下文并调用LLM生成答案
        t0 = time.time()
        if self.use_reranking:
            retriever = HybridRetriever(
                vector_db_dir=self.vector_db_dir,
                documents_dir=self.documents_dir
            )
        else:
            retriever = VectorRetriever(
                vector_db_dir=self.vector_db_dir,
                documents_dir=self.documents_dir
            )
        t1 = time.time()
        print(f"[计时] [get_answer_for_company] 检索器初始化耗时: {t1-t0:.2f} 秒")
        if self.full_context:
            retrieval_results = retriever.retrieve_all(company_name)
        else:
            t2 = time.time()
            retrieval_results = retriever.retrieve_by_company_name(
                company_name=company_name,
                query=question,
                reranking_sample_size=self.reranking_sample_size,
                top_n=self.top_n_retrieval,
                return_parent_pages=self.return_parent_pages
            )
            t3 = time.time()
            print(f"[计时] [get_answer_for_company] 检索耗时: {t3-t2:.2f} 秒")
        if not retrieval_results:
            raise ValueError("No relevant context found")
        t4 = time.time()
        rag_context = self._format_retrieval_results(retrieval_results)
        t5 = time.time()
        print(f"[计时] [get_answer_for_company] 构建rag_context耗时: {t5-t4:.2f} 秒")
        answer_dict = self.openai_processor.get_answer_from_rag_context(
            question=question,
            rag_context=rag_context,
            schema=schema,
            model=self.answering_model
        )
        t6 = time.time()
        print(f"[计时] [get_answer_for_company] LLM调用耗时: {t6-t5:.2f} 秒")
        self.response_data = self.openai_processor.response_data
        if self.new_challenge_pipeline:
            pages = answer_dict.get("relevant_pages", [])
            validated_refs = self._validate_page_references(pages, retrieval_results)
            answer_dict["relevant_pages"] = [r["chunk_lines"] for r in validated_refs]
            answer_dict["references"] = [
                {
                    "pdf_sha1": r["sha1"],
                    "file_name": r["file_name"],
                    "chunk_lines": r["chunk_lines"],
                }
                for r in validated_refs
            ]
        print(f"[计时] [get_answer_for_company] 总耗时: {t6-t0:.2f} 秒")
        return answer_dict

    def get_answer_with_routing(
        self, question: str, schema: str, company_name: str = None
    ) -> dict:
        """Use document routing to select relevant documents, then search all of them."""
        from src.document_router import DocumentRouter
        from src.prompts import AnswerWithRAGContextStringPrompt, AnswerWithRAGContextNumberPrompt

        self.trace = {
            "question": question,
            "schema": schema,
            "steps": [],
            "final_prompt": {},
            "llm_response": None,
        }

        router = DocumentRouter(
            summaries_path=self.summaries_path,
            api_provider=self.api_provider,
            model=self.answering_model,
        )

        result = router.route_question(question, max_documents=3)
        if isinstance(result, tuple):
            selected_sha1s, router_reasoning = result
        else:
            selected_sha1s, router_reasoning = result, ""

        # Trace: 路由结果
        if self.enable_trace:
            routed_docs = []
            for sha1 in selected_sha1s:
                meta = router.get_document_meta(sha1)
                routed_docs.append({
                    "sha1": sha1[:12],
                    "name": meta.get("file_name", "")[:60],
                    "type": meta.get("document_type", ""),
                    "institution": meta.get("issuing_institution", ""),
                })
            self.trace["steps"].append({
                "step": "1. 文档路由",
                "detail": f"从 {len(router.sha1_list)} 篇文档中选出 {len(selected_sha1s)} 篇最相关",
                "router_reasoning": router_reasoning[:500] if router_reasoning else "",
                "routed_documents": routed_docs,
                "all_documents_count": len(router.sha1_list),
            })

        if not selected_sha1s:
            if company_name:
                return self.get_answer_for_company(
                    company_name=company_name, question=question, schema=schema
                )
            raise ValueError("No relevant documents found for the question.")

        if self.use_reranking:
            retriever = HybridRetriever(
                vector_db_dir=self.vector_db_dir,
                documents_dir=self.documents_dir,
            )
            retrieval_results = retriever.retrieve_by_sha1_list(
                sha1_list=selected_sha1s,
                query=question,
                reranking_sample_size=self.reranking_sample_size,
                top_n=self.top_n_retrieval,
                return_parent_pages=self.return_parent_pages,
            )
        else:
            retriever = VectorRetriever(
                vector_db_dir=self.vector_db_dir,
                documents_dir=self.documents_dir,
            )
            retrieval_results = retriever.retrieve_by_sha1_list(
                sha1_list=selected_sha1s,
                query=question,
                top_n=self.top_n_retrieval,
                return_parent_pages=self.return_parent_pages,
            )

        if not retrieval_results:
            raise ValueError("No relevant context found in selected documents")

        # Trace: 检索+重排结果
        if self.enable_trace:
            ranked_preview = []
            for r in retrieval_results[:5]:
                lines = r.get("chunk_lines", [0, 0])
                ranked_preview.append({
                    "source": f"{r.get('file_name', '?')[:50]} 行{lines[0]}-{lines[1]}",
                    "score": r.get("combined_score", 0),
                    "text_preview": r.get("text", "")[:150] + "...",
                })
            self.trace["steps"].append({
                "step": "2. 向量检索 + Jina 重排",
                "detail": f"从选中文档中检索到 {len(retrieval_results)} 个相关片段",
                "top_results": ranked_preview,
            })

        rag_context = self._format_retrieval_results(retrieval_results)

        # Trace: 最终提示词
        if self.enable_trace:
            system_prompt, _, user_prompt_template = self.openai_processor._build_rag_context_prompts(schema)
            final_user_prompt = user_prompt_template.format(context=rag_context, question=question)
            self.trace["steps"].append({
                "step": "3. 拼接最终提示词",
                "detail": f"system_prompt ({len(system_prompt)} 字符) + user_prompt ({len(final_user_prompt)} 字符)",
            })
            self.trace["final_prompt"] = {
                "system_prompt": system_prompt,
                "user_prompt": final_user_prompt,
                "model": self.answering_model,
            }
            self.trace["retrieval_context"] = rag_context

        answer_dict = self.openai_processor.get_answer_from_rag_context(
            question=question,
            rag_context=rag_context,
            schema=schema,
            model=self.answering_model,
        )
        self.response_data = self.openai_processor.response_data

        # Trace: LLM 响应
        if self.enable_trace:
            self.trace["llm_response"] = {
                "model": self.answering_model,
                "usage": self.response_data if isinstance(self.response_data, dict) else {},
            }

        if self.new_challenge_pipeline:
            pages = answer_dict.get("relevant_pages", [])
            validated_refs = self._validate_page_references(pages, retrieval_results)
            # 提取行号列表（向后兼容 relevant_pages）
            answer_dict["relevant_pages"] = [r["chunk_lines"] for r in validated_refs]
            # 构建引用：sha1 + 文件名 + 行号
            answer_dict["references"] = [
                {
                    "pdf_sha1": r["sha1"],
                    "file_name": r["file_name"],
                    "chunk_lines": r["chunk_lines"],
                }
                for r in validated_refs
            ]

        return answer_dict

    def _extract_references_multi_doc(
        self, pages_list: list, sha1_list: list
    ) -> list:
        """Build references across multiple documents for a given page set."""
        refs = []
        for sha1 in sha1_list:
            for page in pages_list:
                refs.append({"pdf_sha1": sha1, "page_index": page})
        return refs

    def _extract_companies_from_subset(self, question_text: str) -> list[str]:
        """从问题文本中提取公司名，匹配subset文件中的公司"""
        if not hasattr(self, 'companies_df'):
            if self.subset_path is None:
                raise ValueError("subset_path must be provided to use subset extraction")
            self.companies_df = pd.read_csv(self.subset_path, encoding='utf-8')
        
        found_companies = []
        company_names = sorted(self.companies_df['company_name'].unique(), key=len, reverse=True)
        
        for company in company_names:
            if not isinstance(company, str) or len(company.strip()) == 0:
                continue
            if company in question_text:
                found_companies.append(company)
                question_text = question_text.replace(company, '')
        
        return found_companies

    def process_question(self, question: str, schema: str):
        # 处理单个问题，支持多公司比较
        if self.new_challenge_pipeline:
            extracted_companies = self._extract_companies_from_subset(question)
        else:
            extracted_companies = re.findall(r'"([^"]*)"', question)

        if len(extracted_companies) == 0:
            raise ValueError("No company name found in the question.")

        if self.use_document_routing:
            company_name = extracted_companies[0] if extracted_companies else None
            if len(extracted_companies) == 1:
                return self.get_answer_with_routing(
                    question=question, schema=schema, company_name=company_name
                )
            else:
                return self.process_comparative_question(
                    question, extracted_companies, schema
                )

        if len(extracted_companies) == 1:
            company_name = extracted_companies[0]
            answer_dict = self.get_answer_for_company(company_name=company_name, question=question, schema=schema)
            return answer_dict
        else:
            return self.process_comparative_question(question, extracted_companies, schema)
    
    def _create_answer_detail_ref(self, answer_dict: dict, question_index: int) -> str:
        """创建答案详情的引用ID，并存储详细内容"""
        ref_id = f"#/answer_details/{question_index}"
        with self._lock:
            self.answer_details[question_index] = {
                "step_by_step_analysis": answer_dict['step_by_step_analysis'],
                "reasoning_summary": answer_dict['reasoning_summary'],
                "relevant_pages": answer_dict['relevant_pages'],
                "response_data": self.response_data,
                "self": ref_id
            }
        return ref_id

    def _calculate_statistics(self, processed_questions: List[dict], print_stats: bool = False) -> dict:
        """统计处理结果，包括总数、错误数、N/A数、成功数"""
        total_questions = len(processed_questions)
        error_count = sum(1 for q in processed_questions if "error" in q)
        na_count = sum(1 for q in processed_questions if (q.get("value") if "value" in q else q.get("answer")) == "N/A")
        success_count = total_questions - error_count - na_count
        if print_stats:
            print(f"\nFinal Processing Statistics:")
            print(f"Total questions: {total_questions}")
            print(f"Errors: {error_count} ({(error_count/total_questions)*100:.1f}%)")
            print(f"N/A answers: {na_count} ({(na_count/total_questions)*100:.1f}%)")
            print(f"Successfully answered: {success_count} ({(success_count/total_questions)*100:.1f}%)\n")
        
        return {
            "total_questions": total_questions,
            "error_count": error_count,
            "na_count": na_count,
            "success_count": success_count
        }

    def process_questions_list(self, questions_list: List[dict], output_path: str = None, submission_file: bool = False, pipeline_details: str = "") -> dict:
        # 批量处理问题列表，支持并行与断点保存，返回处理结果和统计信息
        total_questions = len(questions_list)
        # 给每个问题加索引，便于后续答案详情定位
        questions_with_index = [{**q, "_question_index": i} for i, q in enumerate(questions_list)]
        self.answer_details = [None] * total_questions  # 预分配答案详情列表
        processed_questions = []
        parallel_threads = self.parallel_requests

        if parallel_threads <= 1:
            # 单线程顺序处理
            for question_data in tqdm(questions_with_index, desc="Processing questions"):
                processed_question = self._process_single_question(question_data)
                processed_questions.append(processed_question)
                if output_path:
                    self._save_progress(processed_questions, output_path, submission_file=submission_file, pipeline_details=pipeline_details)
        else:
            # 多线程并行处理
            with tqdm(total=total_questions, desc="Processing questions") as pbar:
                for i in range(0, total_questions, parallel_threads):
                    batch = questions_with_index[i : i + parallel_threads]
                    with concurrent.futures.ThreadPoolExecutor(max_workers=parallel_threads) as executor:
                        # executor.map 保证结果顺序与输入一致
                        batch_results = list(executor.map(self._process_single_question, batch))
                    processed_questions.extend(batch_results)
                    
                    if output_path:
                        self._save_progress(processed_questions, output_path, submission_file=submission_file, pipeline_details=pipeline_details)
                    pbar.update(len(batch_results))
        
        statistics = self._calculate_statistics(processed_questions, print_stats = True)
        
        return {
            "questions": processed_questions,
            "answer_details": self.answer_details,
            "statistics": statistics
        }

    def _process_single_question(self, question_data: dict) -> dict:
        question_index = question_data.get("_question_index", 0)
        
        if self.new_challenge_pipeline:
            question_text = question_data.get("text")
            schema = question_data.get("kind")
        else:
            question_text = question_data.get("question")
            schema = question_data.get("schema")
        try:
            answer_dict = self.process_question(question_text, schema)
            
            if "error" in answer_dict:
                detail_ref = self._create_answer_detail_ref({
                    "step_by_step_analysis": None,
                    "reasoning_summary": None,
                    "relevant_pages": None
                }, question_index)
                if self.new_challenge_pipeline:
                    return {
                        "question_text": question_text,
                        "kind": schema,
                        "value": None,
                        "references": [],
                        "error": answer_dict["error"],
                        "answer_details": {"$ref": detail_ref}
                    }
                else:
                    return {
                        "question": question_text,
                        "schema": schema,
                        "answer": None,
                        "error": answer_dict["error"],
                        "answer_details": {"$ref": detail_ref},
                    }
            detail_ref = self._create_answer_detail_ref(answer_dict, question_index)
            if self.new_challenge_pipeline:
                return {
                    "question_text": question_text,
                    "kind": schema,
                    "value": answer_dict.get("final_answer"),
                    "references": answer_dict.get("references", []),
                    "answer_details": {"$ref": detail_ref}
                }
            else:
                return {
                    "question": question_text,
                    "schema": schema,
                    "answer": answer_dict.get("final_answer"),
                    "answer_details": {"$ref": detail_ref},
                }
        except Exception as err:
            return self._handle_processing_error(question_text, schema, err, question_index)

    def _handle_processing_error(self, question_text: str, schema: str, err: Exception, question_index: int) -> dict:
        """
        处理问题处理过程中的异常。
        记录错误详情并返回包含错误信息的字典。
        """
        import traceback
        error_message = str(err)
        tb = traceback.format_exc()
        error_ref = f"#/answer_details/{question_index}"
        error_detail = {
            "error_traceback": tb,
            "self": error_ref
        }
        
        with self._lock:
            self.answer_details[question_index] = error_detail
        
        print(f"Error encountered processing question: {question_text}")
        print(f"Error type: {type(err).__name__}")
        print(f"Error message: {error_message}")
        print(f"Full traceback:\n{tb}\n")
        
        if self.new_challenge_pipeline:
            return {
                "question_text": question_text,
                "kind": schema,
                "value": None,
                "references": [],
                "error": f"{type(err).__name__}: {error_message}",
                "answer_details": {"$ref": error_ref}
            }
        else:
            return {
                "question": question_text,
                "schema": schema,
                "answer": None,
                "error": f"{type(err).__name__}: {error_message}",
                "answer_details": {"$ref": error_ref},
            }

    def _post_process_submission_answers(self, processed_questions: List[dict]) -> List[dict]:
        """
        提交格式后处理：
        1. 页码从1-based转为0-based
        2. N/A答案清空引用
        3. 格式化为比赛提交schema
        4. 包含step_by_step_analysis
        """
        submission_answers = []
        
        for q in processed_questions:
            question_text = q.get("question_text") or q.get("question")
            kind = q.get("kind") or q.get("schema")
            value = "N/A" if "error" in q else (q.get("value") if "value" in q else q.get("answer"))
            references = q.get("references", [])
            
            answer_details_ref = q.get("answer_details", {}).get("$ref", "")
            step_by_step_analysis = None
            if answer_details_ref and answer_details_ref.startswith("#/answer_details/"):
                try:
                    index = int(answer_details_ref.split("/")[-1])
                    if 0 <= index < len(self.answer_details) and self.answer_details[index]:
                        step_by_step_analysis = self.answer_details[index].get("step_by_step_analysis")
                except (ValueError, IndexError):
                    pass
            
            # Clear references if value is N/A
            if value == "N/A":
                references = []
            else:
                # Convert page indices from one-based to zero-based (competition requires 0-based page indices, but for debugging it is easier to use 1-based)
                references = [
                    {
                        "pdf_sha1": ref["pdf_sha1"],
                        "page_index": ref["page_index"] - 1
                    }
                    for ref in references
                ]
            
            submission_answer = {
                "question_text": question_text,
                "kind": kind,
                "value": value,
                "references": references,
            }
            
            if step_by_step_analysis:
                submission_answer["reasoning_process"] = step_by_step_analysis
            
            submission_answers.append(submission_answer)
        
        return submission_answers

    def _save_progress(self, processed_questions: List[dict], output_path: Optional[str], submission_file: bool = False, pipeline_details: str = ""):
        if output_path:
            statistics = self._calculate_statistics(processed_questions)
            
            # Prepare debug content
            result = {
                "questions": processed_questions,
                "answer_details": self.answer_details,
                "statistics": statistics
            }
            output_file = Path(output_path)
            debug_file = output_file.with_name(output_file.stem + "_debug" + output_file.suffix)
            with open(debug_file, 'w', encoding='utf-8') as file:
                json.dump(result, file, ensure_ascii=False, indent=2)
            
            if submission_file:
                # Post-process answers for submission
                submission_answers = self._post_process_submission_answers(processed_questions)
                submission = {
                    "answers": submission_answers,
                    "details": pipeline_details
                }
                with open(output_file, 'w', encoding='utf-8') as file:
                    json.dump(submission, file, ensure_ascii=False, indent=2)

    def process_all_questions(self, output_path: str = 'questions_with_answers.json', submission_file: bool = False, pipeline_details: str = ""):
        result = self.process_questions_list(
            self.questions,
            output_path,
            submission_file=submission_file,
            pipeline_details=pipeline_details
        )
        return result

    def process_comparative_question(self, question: str, companies: List[str], schema: str) -> dict:
        """
        处理多公司比较类问题：
        1. 先将比较问题重写为单公司问题
        2. 并行处理每个公司
        3. 汇总结果并生成最终比较答案
        """
        # Step 1: Rephrase the comparative question
        rephrased_questions = self.openai_processor.get_rephrased_questions(
            original_question=question,
            companies=companies
        )
        
        individual_answers = {}
        aggregated_references = []
        
        # Step 2: Process each individual question in parallel
        def process_company_question(company: str) -> tuple[str, dict]:
            """Helper function to process one company's question and return (company, answer)"""
            sub_question = rephrased_questions.get(company)
            if not sub_question:
                raise ValueError(f"Could not generate sub-question for company: {company}")
            
            answer_dict = self.get_answer_for_company(
                company_name=company, 
                question=sub_question, 
                schema="number"
            )
            return company, answer_dict

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_company = {
                executor.submit(process_company_question, company): company 
                for company in companies
            }
            
            for future in concurrent.futures.as_completed(future_to_company):
                try:
                    company, answer_dict = future.result()
                    individual_answers[company] = answer_dict
                    
                    company_references = answer_dict.get("references", [])
                    aggregated_references.extend(company_references)
                except Exception as e:
                    company = future_to_company[future]
                    print(f"Error processing company {company}: {str(e)}")
                    raise
        
        # Remove duplicate references
        unique_refs = {}
        for ref in aggregated_references:
            key = (ref.get("pdf_sha1"), ref.get("page_index"))
            unique_refs[key] = ref
        aggregated_references = list(unique_refs.values())
        
        # Step 3: Get the comparative answer using all individual answers
        comparative_answer = self.openai_processor.get_answer_from_rag_context(
            question=question,
            rag_context=individual_answers,
            schema="comparative",
            model=self.answering_model
        )
        self.response_data = self.openai_processor.response_data
        
        comparative_answer["references"] = aggregated_references
        return comparative_answer

    def process_single_question(self, question: str, kind: str = "string"):
        """
        单条问题推理，返回结构化答案。
        kind: 支持 'string'、'number'、'boolean'、'names' 等
        """
        t0 = time.time()
        print("[计时] [单问] 开始公司名抽取 ...")
        # 公司名抽取
        if self.new_challenge_pipeline:
            extracted_companies = self._extract_companies_from_subset(question)
        else:
            extracted_companies = re.findall(r'"([^"]*)"', question)
        t1 = time.time()
        print(f"[计时] [单问] 公司名抽取耗时: {t1-t0:.2f} 秒")
        if len(extracted_companies) == 0:
            raise ValueError("No company name found in the question.")

        if self.use_document_routing and len(extracted_companies) == 1:
            company_name = extracted_companies[0]
            print("[计时] [单问] 开始文档路由+检索+LLM推理 ...")
            t2 = time.time()
            answer_dict = self.get_answer_with_routing(
                question=question, schema=kind, company_name=company_name
            )
            t3 = time.time()
            print(f"[计时] [单问] 路由+检索+LLM推理耗时: {t3-t2:.2f} 秒")
            print(f"[计时] [单问] 总耗时: {t3-t0:.2f} 秒")
            return answer_dict

        if len(extracted_companies) == 1:
            company_name = extracted_companies[0]
            print("[计时] [单问] 开始检索与LLM推理 ...")
            t2 = time.time()
            answer_dict = self.get_answer_for_company(company_name=company_name, question=question, schema=kind)
            t3 = time.time()
            print(f"[计时] [单问] 检索+LLM推理耗时: {t3-t2:.2f} 秒")
            print(f"[计时] [单问] 总耗时: {t3-t0:.2f} 秒")
            return answer_dict
        else:
            print("[计时] [单问] 开始多公司比较 ...")
            t2 = time.time()
            answer_dict = self.process_comparative_question(question, extracted_companies, kind)
            t3 = time.time()
            print(f"[计时] [单问] 多公司比较耗时: {t3-t2:.2f} 秒")
            print(f"[计时] [单问] 总耗时: {t3-t0:.2f} 秒")
            return answer_dict
    