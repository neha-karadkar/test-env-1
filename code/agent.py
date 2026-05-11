import asyncio as _asyncio

import time as _time
from observability.observability_wrapper import (
    trace_agent, trace_step, trace_step_sync, trace_model_call, trace_tool_call,
)
from config import settings as _obs_settings

import logging as _obs_startup_log
from contextlib import asynccontextmanager
from observability.instrumentation import initialize_tracer

_obs_startup_logger = _obs_startup_log.getLogger(__name__)

from modules.guardrails.content_safety_decorator import with_content_safety

GUARDRAILS_CONFIG = {
    'content_safety_enabled': True,
    'runtime_enabled': True,
    'content_safety_severity_threshold': 3,
    'check_toxicity': True,
    'check_jailbreak': True,
    'check_pii_input': False,
    'check_credentials_output': True,
    'check_output': True,
    'check_toxic_code_output': True,
    'sanitize_pii': False
}

import logging
import json
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field, field_validator, ValidationError
from pathlib import Path

import openai
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.models import VectorizedQuery

from config import Config

# =========================
# CONSTANTS
# =========================

SYSTEM_PROMPT = (
    "You are a professional planetary knowledge assistant. Your role is to answer user questions about planets and astronomy using only the information retrieved from the provided knowledge base documents (Earth.pdf, Jupiter.pdf). Follow these instructions:\n\n"
    "- Carefully read the user's question.\n\n"
    "- Use only the retrieved content from the knowledge base to construct your answer.\n\n"
    "- If the answer is not found in the retrieved content, respond with a polite fallback message indicating that the information is not available.\n\n"
    "- Format your answer in clear, concise, and professional language.\n\n"
    "- Do not speculate or provide information not supported by the source documents.\n\n"
    "- Always cite relevant facts from the retrieved content when possible."
)
OUTPUT_FORMAT = "Provide a direct, well-structured answer in text format. If information is not found, use the fallback response."
FALLBACK_RESPONSE = "I'm sorry, but I could not find the information you requested in the available knowledge base documents."
SELECTED_DOCUMENT_TITLES = ["Earth.pdf", "Jupiter.pdf"]

VALIDATION_CONFIG_PATH = Config.VALIDATION_CONFIG_PATH or str(Path(__file__).parent / "validation_config.json")

# =========================
# INPUT/OUTPUT MODELS
# =========================

class QueryRequest(BaseModel):
    query: str = Field(..., description="The user's question about planets or astronomy.")

    @field_validator('query')
    @classmethod
    @with_content_safety(config=GUARDRAILS_CONFIG)
    def validate_query(cls, v):
        if not v or not v.strip():
            raise ValueError("Query must not be empty.")
        if len(v) > 50000:
            raise ValueError("Query is too long (max 50,000 characters).")
        return v.strip()

class QueryResponse(BaseModel):
    success: bool = Field(..., description="Whether the query was processed successfully.")
    answer: Optional[str] = Field(None, description="The answer to the user's question.")
    error: Optional[str] = Field(None, description="Error message if the query failed.")
    tool_calls_made: Optional[List[str]] = Field(None, description="List of tool calls made (always empty for this agent).")

# =========================
# UTILITY: LLM OUTPUT SANITIZER
# =========================

import re as _re

_FENCE_RE = _re.compile(r"```(?:\w+)?\s*\n(.*?)```", _re.DOTALL)
_LONE_FENCE_START_RE = _re.compile(r"^```\w*$")
_WRAPPER_RE = _re.compile(
    r"^(?:"
    r"Here(?:'s| is)(?: the)? (?:the |your |a )?(?:code|solution|implementation|result|explanation|answer)[^:]*:\s*"
    r"|Sure[!,.]?\s*"
    r"|Certainly[!,.]?\s*"
    r"|Below is [^:]*:\s*"
    r")",
    _re.IGNORECASE,
)
_SIGNOFF_RE = _re.compile(
    r"^(?:Let me know|Feel free|Hope this|This code|Note:|Happy coding|If you)",
    _re.IGNORECASE,
)
_BLANK_COLLAPSE_RE = _re.compile(r"\n{3,}")

def _strip_fences(text: str, content_type: str) -> str:
    """Extract content from Markdown code fences."""
    fence_matches = _FENCE_RE.findall(text)
    if fence_matches:
        if content_type == "code":
            return "\n\n".join(block.strip() for block in fence_matches)
        for match in fence_matches:
            fenced_block = _FENCE_RE.search(text)
            if fenced_block:
                text = text[:fenced_block.start()] + match.strip() + text[fenced_block.end():]
        return text
    lines = text.splitlines()
    if lines and _LONE_FENCE_START_RE.match(lines[0].strip()):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()

def _strip_trailing_signoffs(text: str) -> str:
    """Remove conversational sign-off lines from the end of code output."""
    lines = text.splitlines()
    while lines and _SIGNOFF_RE.match(lines[-1].strip()):
        lines.pop()
    return "\n".join(lines).rstrip()

@with_content_safety(config=GUARDRAILS_CONFIG)
def sanitize_llm_output(raw: str, content_type: str = "code") -> str:
    """
    Generic post-processor that cleans common LLM output artefacts.
    Args:
        raw: Raw text returned by the LLM.
        content_type: 'code' | 'text' | 'markdown'.
    Returns:
        Cleaned string ready for validation, formatting, or direct return.
    """
    if not raw:
        return ""
    text = _strip_fences(raw.strip(), content_type)
    text = _WRAPPER_RE.sub("", text, count=1).strip()
    if content_type == "code":
        text = _strip_trailing_signoffs(text)
    return _BLANK_COLLAPSE_RE.sub("\n\n", text).strip()

# =========================
# COMPONENTS
# =========================

class InputProcessor:
    """Receives and validates user queries; applies communication templates."""

    @with_content_safety(config=GUARDRAILS_CONFIG)
    def process_input(self, query: str) -> str:
        """Validate and sanitize the user query."""
        if not query or not query.strip():
            raise ValueError("Query must not be empty.")
        if len(query) > 50000:
            raise ValueError("Query is too long (max 50,000 characters).")
        return query.strip()

class ChunkRetriever:
    """Queries Azure AI Search for relevant chunks using vector + keyword search; applies OData filter for selected_document_titles."""

    def __init__(self):
        self._search_client = None
        self._openai_client = None

    def _get_search_client(self):
        if self._search_client is None:
            if not Config.AZURE_SEARCH_ENDPOINT or not Config.AZURE_SEARCH_API_KEY or not Config.AZURE_SEARCH_INDEX_NAME:
                raise RuntimeError("Azure Search credentials are not configured.")
            self._search_client = SearchClient(
                endpoint=Config.AZURE_SEARCH_ENDPOINT,
                index_name=Config.AZURE_SEARCH_INDEX_NAME,
                credential=AzureKeyCredential(Config.AZURE_SEARCH_API_KEY),
            )
        return self._search_client

    def _get_openai_client(self):
        if self._openai_client is None:
            if not Config.AZURE_OPENAI_API_KEY or not Config.AZURE_OPENAI_ENDPOINT:
                raise RuntimeError("Azure OpenAI credentials are not configured.")
            self._openai_client = openai.AsyncAzureOpenAI(
                api_key=Config.AZURE_OPENAI_API_KEY,
                api_version="2024-02-01",
                azure_endpoint=Config.AZURE_OPENAI_ENDPOINT,
            )
        return self._openai_client

    @with_content_safety(config=GUARDRAILS_CONFIG)
    async def retrieve_chunks(self, query: str, selected_document_titles: List[str]) -> List[str]:
        """
        Retrieve relevant chunks from Azure AI Search using vector + keyword search.
        Applies OData filter for selected_document_titles.
        Returns a list of chunk strings.
        """
        search_client = self._get_search_client()
        openai_client = self._get_openai_client()

        # Step 1: Embed the query using Azure OpenAI
        _t0 = _time.time()
        try:
            embedding_resp = await openai_client.embeddings.create(
                input=query,
                model=Config.AZURE_OPENAI_EMBEDDING_DEPLOYMENT or "text-embedding-ada-002"
            )
            try:
                trace_tool_call(
                    tool_name="openai_client.embeddings.create",
                    latency_ms=int((_time.time() - _t0) * 1000),
                    args={"input": query, "model": Config.AZURE_OPENAI_EMBEDDING_DEPLOYMENT or "text-embedding-ada-002"},
                    output=str(embedding_resp)[:200],
                    status="success"
                )
            except Exception:
                pass
        except Exception as e:
            try:
                trace_tool_call(
                    tool_name="openai_client.embeddings.create",
                    latency_ms=int((_time.time() - _t0) * 1000),
                    args={"input": query, "model": Config.AZURE_OPENAI_EMBEDDING_DEPLOYMENT or "text-embedding-ada-002"},
                    output=str(e),
                    status="error",
                    error=e
                )
            except Exception:
                pass
            raise RuntimeError(f"Failed to generate embedding: {e}")

        if not embedding_resp or not embedding_resp.data or not embedding_resp.data[0].embedding:
            return []

        vector_query = VectorizedQuery(
            vector=embedding_resp.data[0].embedding,
            k_nearest_neighbors=5,
            fields="vector"
        )

        # Step 2: Build OData filter for selected_document_titles
        search_kwargs = {
            "search_text": query,
            "vector_queries": [vector_query],
            "top": 5,
            "select": ["chunk", "title"],
        }
        if selected_document_titles:
            odata_parts = [f"title eq '{t}'" for t in selected_document_titles]
            search_kwargs["filter"] = " or ".join(odata_parts)

        # Step 3: Search
        _t1 = _time.time()
        try:
            results = search_client.search(**search_kwargs)
            context_chunks = [r["chunk"] for r in results if r.get("chunk")]
            try:
                trace_tool_call(
                    tool_name="search_client.search",
                    latency_ms=int((_time.time() - _t1) * 1000),
                    args=search_kwargs,
                    output=str(context_chunks)[:200],
                    status="success"
                )
            except Exception:
                pass
            return context_chunks
        except Exception as e:
            try:
                trace_tool_call(
                    tool_name="search_client.search",
                    latency_ms=int((_time.time() - _t1) * 1000),
                    args=search_kwargs,
                    output=str(e),
                    status="error",
                    error=e
                )
            except Exception:
                pass
            return []

class LLMService:
    """Calls Azure OpenAI GPT-4.1 with enhanced system prompt, user query, and retrieved chunks as context."""

    def __init__(self):
        self._client = None

    def _get_client(self):
        if self._client is None:
            if not Config.AZURE_OPENAI_API_KEY or not Config.AZURE_OPENAI_ENDPOINT:
                raise RuntimeError("Azure OpenAI credentials are not configured.")
            self._client = openai.AsyncAzureOpenAI(
                api_key=Config.AZURE_OPENAI_API_KEY,
                api_version="2024-02-01",
                azure_endpoint=Config.AZURE_OPENAI_ENDPOINT,
            )
        return self._client

    @with_content_safety(config=GUARDRAILS_CONFIG)
    @trace_agent(agent_name=_obs_settings.AGENT_NAME, project_name=_obs_settings.PROJECT_NAME)
    async def generate_response(self, prompt: str, context: List[str], user_query: str) -> str:
        """
        Calls LLM with system prompt, user query, and retrieved chunks as context.
        Returns the answer as a string.
        """
        client = self._get_client()
        system_message = f"{prompt}\n\nOutput Format: {OUTPUT_FORMAT}"
        context_text = "\n\n".join(context) if context else ""
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": f"{user_query}\n\nContext:\n{context_text}"}
        ]
        _t0 = _time.time()
        try:
            response = await client.chat.completions.create(
                model=Config.LLM_MODEL or "gpt-4.1",
                messages=messages,
                **Config.get_llm_kwargs()
            )
            content = response.choices[0].message.content if response.choices and response.choices[0].message else ""
            try:
                trace_model_call(
                    provider="azure",
                    model_name=Config.LLM_MODEL or "gpt-4.1",
                    prompt_tokens=getattr(getattr(response, "usage", None), "prompt_tokens", 0) or 0,
                    completion_tokens=getattr(getattr(response, "usage", None), "completion_tokens", 0) or 0,
                    latency_ms=int((_time.time() - _t0) * 1000),
                    response_summary=content[:200] if content else "",
                )
            except Exception:
                pass
            return sanitize_llm_output(content, content_type="text")
        except Exception as e:
            try:
                trace_model_call(
                    provider="azure",
                    model_name=Config.LLM_MODEL or "gpt-4.1",
                    prompt_tokens=0,
                    completion_tokens=0,
                    latency_ms=int((_time.time() - _t0) * 1000),
                    response_summary=str(e),
                    status="error",
                    error=e
                )
            except Exception:
                pass
            return FALLBACK_RESPONSE

class AuditLogger:
    """Logs all agent actions, errors, and responses for compliance and monitoring."""

    def __init__(self):
        self.logger = logging.getLogger("agent.audit")
        self.logger.setLevel(logging.INFO)

    def log_event(self, event_type: str, details: Dict[str, Any]):
        try:
            self.logger.info(f"{event_type}: {json.dumps(details, default=str)}")
        except Exception as e:
            self.logger.warning(f"Failed to log event: {e}")

class ToolRegistry:
    """Stub for extensibility; no tools for this agent."""

    def register_tool(self, tool):
        pass

    async def execute_tool_call(self, tool_name: str, params: Dict[str, Any]):
        return None

# =========================
# MAIN AGENT
# =========================

class Agent:
    """Orchestrates the flow: receives query, retrieves chunks, calls LLM, handles response/fallback, manages error handling and audit logging."""

    def __init__(self):
        self.input_processor = InputProcessor()
        self.chunk_retriever = ChunkRetriever()
        self.llm_service = LLMService()
        self.audit_logger = AuditLogger()
        self.tool_registry = ToolRegistry()

    @with_content_safety(config=GUARDRAILS_CONFIG)
    async def answer_query(self, user_query: str) -> Dict[str, Any]:
        """
        Orchestrates the full query-answer flow: input processing, chunk retrieval, LLM call, response formatting, error handling.
        Returns a dict matching QueryResponse.
        """
        async with trace_step(
            "process_input", step_type="parse",
            decision_summary="Validate and sanitize user query",
            output_fn=lambda r: f"query={r}"
        ) as step:
            try:
                query = self.input_processor.process_input(user_query)
                step.capture({"query": query})
            except Exception as e:
                self.audit_logger.log_event("input_validation_error", {"error": str(e)})
                return {
                    "success": False,
                    "answer": None,
                    "error": f"Input validation error: {str(e)}",
                    "tool_calls_made": []
                }

        async with trace_step(
            "retrieve_chunks", step_type="process",
            decision_summary="Retrieve relevant chunks from Azure AI Search",
            output_fn=lambda r: f"chunks={len(r)}"
        ) as step:
            try:
                chunks = await self.chunk_retriever.retrieve_chunks(query, SELECTED_DOCUMENT_TITLES)
                step.capture({"chunks": len(chunks)})
            except Exception as e:
                self.audit_logger.log_event("retrieval_error", {"error": str(e)})
                return {
                    "success": False,
                    "answer": None,
                    "error": f"Retrieval error: {str(e)}",
                    "tool_calls_made": []
                }

        if not chunks:
            self.audit_logger.log_event("kb_not_found", {"query": query})
            return {
                "success": True,
                "answer": FALLBACK_RESPONSE,
                "error": None,
                "tool_calls_made": []
            }

        async with trace_step(
            "generate_response", step_type="llm_call",
            decision_summary="Generate answer using LLM with retrieved chunks",
            output_fn=lambda r: f"answer={r[:80] if r else ''}"
        ) as step:
            try:
                answer = await self.llm_service.generate_response(SYSTEM_PROMPT, chunks, query)
                answer = sanitize_llm_output(answer, content_type="text")
                step.capture({"answer": answer})
            except Exception as e:
                self.audit_logger.log_event("llm_error", {"error": str(e)})
                return {
                    "success": False,
                    "answer": FALLBACK_RESPONSE,
                    "error": f"LLM error: {str(e)}",
                    "tool_calls_made": []
                }

        self.audit_logger.log_event("answer_generated", {"query": query, "answer": answer})
        return {
            "success": True,
            "answer": answer,
            "error": None,
            "tool_calls_made": []
        }

# =========================
# OBSERVABILITY LIFESPAN
# =========================

@asynccontextmanager
async def _obs_lifespan(application):
    """Initialise observability on startup, clean up on shutdown."""
    try:
        _obs_startup_logger.info('')
        _obs_startup_logger.info('========== Agent Configuration Summary ==========')
        _obs_startup_logger.info(f'Environment: {getattr(Config, "ENVIRONMENT", "N/A")}')
        _obs_startup_logger.info(f'Agent: {getattr(Config, "AGENT_NAME", "N/A")}')
        _obs_startup_logger.info(f'Project: {getattr(Config, "PROJECT_NAME", "N/A")}')
        _obs_startup_logger.info(f'LLM Provider: {getattr(Config, "MODEL_PROVIDER", "N/A")}')
        _obs_startup_logger.info(f'LLM Model: {getattr(Config, "LLM_MODEL", "N/A")}')
        _cs_endpoint = getattr(Config, 'AZURE_CONTENT_SAFETY_ENDPOINT', None)
        _cs_key = getattr(Config, 'AZURE_CONTENT_SAFETY_KEY', None)
        if _cs_endpoint and _cs_key:
            _obs_startup_logger.info('Content Safety: Enabled (Azure Content Safety)')
            _obs_startup_logger.info(f'Content Safety Endpoint: {_cs_endpoint}')
        else:
            _obs_startup_logger.info('Content Safety: Not Configured')
        _obs_startup_logger.info('Observability Database: Azure SQL')
        _obs_startup_logger.info(f'Database Server: {getattr(Config, "OBS_AZURE_SQL_SERVER", "N/A")}')
        _obs_startup_logger.info(f'Database Name: {getattr(Config, "OBS_AZURE_SQL_DATABASE", "N/A")}')
        _obs_startup_logger.info('===============================================')
        _obs_startup_logger.info('')
    except Exception as _e:
        _obs_startup_logger.warning('Config summary failed: %s', _e)

    _obs_startup_logger.info('')
    _obs_startup_logger.info('========== Content Safety & Guardrails ==========')
    if GUARDRAILS_CONFIG.get('content_safety_enabled'):
        _obs_startup_logger.info('Content Safety: Enabled')
        _obs_startup_logger.info(f'  - Severity Threshold: {GUARDRAILS_CONFIG.get("content_safety_severity_threshold", "N/A")}')
        _obs_startup_logger.info(f'  - Check Toxicity: {GUARDRAILS_CONFIG.get("check_toxicity", False)}')
        _obs_startup_logger.info(f'  - Check Jailbreak: {GUARDRAILS_CONFIG.get("check_jailbreak", False)}')
        _obs_startup_logger.info(f'  - Check PII Input: {GUARDRAILS_CONFIG.get("check_pii_input", False)}')
        _obs_startup_logger.info(f'  - Check Credentials Output: {GUARDRAILS_CONFIG.get("check_credentials_output", False)}')
    else:
        _obs_startup_logger.info('Content Safety: Disabled')
    _obs_startup_logger.info('===============================================')
    _obs_startup_logger.info('')

    _obs_startup_logger.info('========== Initializing Agent Services ==========')
    # 1. Observability DB schema (imports are inside function — only needed at startup)
    try:
        from observability.database.engine import create_obs_database_engine
        from observability.database.base import ObsBase
        import observability.database.models  # noqa: F401
        _obs_engine = create_obs_database_engine()
        ObsBase.metadata.create_all(bind=_obs_engine, checkfirst=True)
        _obs_startup_logger.info('✓ Observability database connected')
    except Exception as _e:
        _obs_startup_logger.warning('✗ Observability database connection failed (metrics will not be saved)')
    # 2. OpenTelemetry tracer (initialize_tracer is pre-injected at top level)
    try:
        _t = initialize_tracer()
        if _t is not None:
            _obs_startup_logger.info('✓ Telemetry monitoring enabled')
        else:
            _obs_startup_logger.warning('✗ Telemetry monitoring disabled')
    except Exception as _e:
        _obs_startup_logger.warning('✗ Telemetry monitoring failed to initialize')
    _obs_startup_logger.info('=================================================')
    _obs_startup_logger.info('')
    yield

# =========================
# FASTAPI APP
# =========================

app = FastAPI(lifespan=_obs_lifespan,

    title="Planetary Knowledge Assistant",
    description="A professional planetary knowledge assistant that answers user questions about planets and astronomy using only the information retrieved from the provided knowledge base documents (Earth.pdf, Jupiter.pdf).",
    version=Config.SERVICE_VERSION if hasattr(Config, "SERVICE_VERSION") else "1.0.0",
    # SYNTAX-FIX: lifespan=_obs_lifespan
)

# =========================
# ERROR HANDLING
# =========================

@app.exception_handler(RequestValidationError)
@with_content_safety(config=GUARDRAILS_CONFIG)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "success": False,
            "answer": None,
            "error": f"Malformed request: {exc.errors()}",
            "tool_calls_made": []
        }
    )

@app.exception_handler(ValidationError)
@with_content_safety(config=GUARDRAILS_CONFIG)
async def pydantic_validation_exception_handler(request: Request, exc: ValidationError):
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "success": False,
            "answer": None,
            "error": f"Malformed request: {exc.errors()}",
            "tool_calls_made": []
        }
    )

@app.exception_handler(json.decoder.JSONDecodeError)
@with_content_safety(config=GUARDRAILS_CONFIG)
async def json_decode_exception_handler(request: Request, exc: json.decoder.JSONDecodeError):
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={
            "success": False,
            "answer": None,
            "error": f"Malformed JSON: {str(exc)}. Ensure your request body is valid JSON (check for missing quotes, commas, or brackets).",
            "tool_calls_made": []
        }
    )

@app.exception_handler(Exception)
@with_content_safety(config=GUARDRAILS_CONFIG)
async def generic_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "success": False,
            "answer": None,
            "error": f"Internal server error: {str(exc)}",
            "tool_calls_made": []
        }
    )

# =========================
# ENDPOINTS
# =========================

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok"}

_agent_instance = Agent()

@app.post("/query", response_model=QueryResponse)
@with_content_safety(config=GUARDRAILS_CONFIG)
async def query_endpoint(req: QueryRequest):
    """
    Main endpoint for querying planetary knowledge.
    """
    result = await _agent_instance.answer_query(user_query=req.query)
    return result

# =========================
# MAIN ENTRYPOINT
# =========================

async def _run_agent():
    """Entrypoint: runs the agent with observability (trace collection only)."""
    import uvicorn

    # Unified logging config — routes uvicorn, agent, and observability through
    # the same handler so all telemetry appears in a single consistent stream.
    _LOG_CONFIG = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "()": "uvicorn.logging.DefaultFormatter",
                "fmt": "%(levelprefix)s %(name)s: %(message)s",
                "use_colors": None,
            },
            "access": {
                "()": "uvicorn.logging.AccessFormatter",
                "fmt": '%(levelprefix)s %(client_addr)s - "%(request_line)s" %(status_code)s',
            },
        },
        "handlers": {
            "default": {
                "formatter": "default",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stderr",
            },
            "access": {
                "formatter": "access",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stdout",
            },
        },
        "loggers": {
            "uvicorn":        {"handlers": ["default"], "level": "INFO", "propagate": False},
            "uvicorn.error":  {"level": "INFO"},
            "uvicorn.access": {"handlers": ["access"], "level": "INFO", "propagate": False},
            "agent":          {"handlers": ["default"], "level": "INFO", "propagate": False},
            "__main__":       {"handlers": ["default"], "level": "INFO", "propagate": False},
            "observability": {"handlers": ["default"], "level": "INFO", "propagate": False},
            "config": {"handlers": ["default"], "level": "INFO", "propagate": False},
            "azure":   {"handlers": ["default"], "level": "WARNING", "propagate": False},
            "urllib3": {"handlers": ["default"], "level": "WARNING", "propagate": False},
        },
    }

    config = uvicorn.Config(
        "agent:app",
        host="0.0.0.0",
        port=8080,
        reload=False,
        log_level="info",
        log_config=_LOG_CONFIG,
    )
    server = uvicorn.Server(config)
    await server.serve()


if __name__ == "__main__":
    _asyncio.run(_run_agent())