"""LLM abstraction for answering and judging in LongMemEval."""

from __future__ import annotations

import json
import os
import re
import time
from typing import Any
from rich.console import Console

from .models import EvidenceBundle, VerifiedAnswer

console = Console()


def _parse_json_object(raw: str) -> dict[str, Any] | None:
    match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        return None


def _string_tuple(value: Any, limit: int) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(str(item).strip() for item in value[:limit] if str(item).strip())


class BaseLLM:
    """Base LLM interface."""

    def generate(self, prompt: str, max_tokens: int = 4096, temperature: float = 0.0) -> str:
        raise NotImplementedError

    def judge_bool(self, prompt: str) -> tuple[bool, str]:
        """Returns (is_correct, explanation)."""
        raise NotImplementedError

    def extract_evidence(
        self,
        question: str,
        context: str,
        question_date: str | None = None,
    ) -> EvidenceBundle:
        """Stage 1: Extract up to 12 facts and classify support status."""
        from .prompts import build_extract_evidence_prompt
        prompt = build_extract_evidence_prompt(question, context, question_date)
        raw = self.generate(prompt, max_tokens=2048, temperature=0.0).strip()
        parsed = _parse_json_object(raw)
        if parsed is None:
            raw = self.generate(
                f"{prompt}\n\nYour previous output was not valid JSON. Return one valid JSON object with keys: status, facts, requirements.",
                max_tokens=2048,
                temperature=0.0,
            ).strip()
            parsed = _parse_json_object(raw)

        if parsed is None:
            return EvidenceBundle("unsupported", (), ("Answer the question accurately.",))

        status = str(parsed.get("status", "unsupported")).lower()
        if status not in {"direct", "inferable", "unsupported"}:
            status = "direct" if parsed.get("facts") else "unsupported"
        facts = _string_tuple(parsed.get("facts"), limit=12)
        requirements = _string_tuple(parsed.get("requirements"), limit=8)
        return EvidenceBundle(status, facts, requirements)

    def answer_from_evidence(
        self,
        question: str,
        evidence: EvidenceBundle,
        question_date: str | None = None,
        max_tokens: int = 1024,
    ) -> str:
        """Stage 2: Generate candidate answer from extracted facts."""
        from .prompts import build_evidence_answer_prompt
        prompt = build_evidence_answer_prompt(question, evidence.as_context(), question_date)
        return self.generate(prompt, max_tokens=max_tokens, temperature=0.0).strip()

    def infer_answer(
        self,
        question: str,
        evidence: EvidenceBundle,
        question_date: str | None = None,
        max_tokens: int = 1024,
    ) -> str:
        """Stage 3: Conditional narrow inference or temporal calculation."""
        from .prompts import build_infer_answer_prompt
        prompt = build_infer_answer_prompt(question, evidence.as_context(), question_date)
        return self.generate(prompt, max_tokens=max_tokens, temperature=0.0).strip()

    def verify_answer(
        self,
        question: str,
        evidence: EvidenceBundle,
        candidates: tuple[str, ...],
        question_date: str | None = None,
    ) -> VerifiedAnswer:
        """Stage 4: Verify candidate answer, repair counts/dates, or hold back if unsupported."""
        from .prompts import build_verify_answer_prompt
        prompt = build_verify_answer_prompt(question, evidence.as_context(), candidates, question_date)
        raw = self.generate(prompt, max_tokens=1024, temperature=0.0).strip()
        parsed = _parse_json_object(raw)
        if parsed is None or not str(parsed.get("answer", "")).strip():
            raw = self.generate(
                f"{prompt}\n\nYour previous output was not valid JSON. Return JSON only: {{\"answer\":\"final answer\",\"reason\":\"brief reason\"}}",
                max_tokens=1024,
                temperature=0.0,
            ).strip()
            parsed = _parse_json_object(raw)

        if parsed is None or not str(parsed.get("answer", "")).strip():
            fallback = candidates[0] if candidates else "You did not mention this information in our chats."
            return VerifiedAnswer(fallback, "Verifier fallback.")

        return VerifiedAnswer(
            str(parsed.get("answer", "")).strip(),
            str(parsed.get("reason", "Verified against extracted facts.")).strip(),
        )


class GeminiLLM(BaseLLM):
    """Google Gemini LLM using google-genai."""

    def __init__(self, model_name: str = "gemini-3.8-flash"):
        from google import genai
        from google.genai import types

        self.model_name = model_name
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is required")
        self.client = genai.Client(api_key=api_key)
        self.types = types

    def generate(self, prompt: str, max_tokens: int = 4096, temperature: float = 0.0) -> str:
        for attempt in range(5):
            try:
                resp = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=self.types.GenerateContentConfig(
                        temperature=temperature,
                        max_output_tokens=max_tokens,
                        automatic_function_calling=self.types.AutomaticFunctionCallingConfig(disable=True),
                    ),
                )
                return resp.text or ""
            except Exception as exc:
                if attempt == 4:
                    raise
                wait = 2 * (attempt + 1)
                time.sleep(wait)
        return ""

    def judge_bool(self, prompt: str) -> tuple[bool, str]:
        # Generate answer with explanation
        resp_text = self.generate(prompt, max_tokens=256, temperature=0.0).strip()
        lower = resp_text.lower()
        # Check standard criteria
        is_yes = False
        if "correct: true" in lower or '"correct": true' in lower or "answer yes" in lower:
            is_yes = True
        elif lower.startswith("yes") or "\nyes" in lower or "is correct: yes" in lower:
            is_yes = True
        elif "correct=true" in lower:
            is_yes = True
        elif "yes" in lower and "no" not in lower:
            is_yes = True

        return is_yes, resp_text


class OpenAILLM(BaseLLM):
    """OpenAI API LLM (e.g. gpt-4o, gpt-4o-mini)."""

    def __init__(self, model_name: str = "gpt-4o"):
        from openai import OpenAI
        self.model_name = model_name
        self.client = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
            organization=os.environ.get("OPENAI_ORGANIZATION"),
        )

    def generate(self, prompt: str, max_tokens: int = 4096, temperature: float = 0.0) -> str:
        resp = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content or ""

    def judge_bool(self, prompt: str) -> tuple[bool, str]:
        resp = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=64,
        )
        content = resp.choices[0].message.content or ""
        label = "yes" in content.lower()
        return label, content


class GrokLLM(OpenAILLM):
    """xAI Grok via the OpenAI-compatible Chat Completions API."""

    def __init__(self, model_name: str = "grok-4.6"):
        from openai import OpenAI

        self.model_name = model_name
        api_key = os.environ.get("GROK_API_KEY") or os.environ.get("XAI_API_KEY")
        if not api_key:
            raise ValueError("GROK_API_KEY or XAI_API_KEY environment variable is required")
        self.client = OpenAI(api_key=api_key, base_url="https://api.x.ai/v1")

    def judge_bool(self, prompt: str) -> tuple[bool, str]:
        resp_text = self.generate(prompt, max_tokens=64, temperature=0.0).strip()
        lower = resp_text.lower()
        is_yes = False
        if lower.startswith("yes") or "\nyes" in lower:
            is_yes = True
        elif "yes" in lower and "no" not in lower:
            is_yes = True
        return is_yes, resp_text


def get_llm(provider: str = "gemini", model: str | None = None) -> BaseLLM:
    prov = provider.lower()
    if prov == "gemini":
        return GeminiLLM(model_name=model or os.environ.get("READER_MODEL", "gemini-3.8-flash"))
    elif prov == "openai":
        return OpenAILLM(model_name=model or os.environ.get("OPENAI_MODEL", "gpt-4o"))
    elif prov in ("grok", "xai"):
        return GrokLLM(model_name=model or os.environ.get("GROK_MODEL", "grok-4.6"))
    else:
        raise ValueError(f"Unknown LLM provider '{provider}'. Choose 'gemini', 'openai', or 'grok'.")
