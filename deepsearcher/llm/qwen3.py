from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict
from abc import ABC, abstractmethod
import ast
import re
from abc import ABC
from typing import Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class ChatResponse:
    """Container holding model output.

    Attributes
    ----------
    content: str
        The final assistant reply that should be shown to the user.
    thinking: str, optional
        The intermediate "<think> … </think>" reasoning emitted by Qwen.
    raw_ids: List[int], optional
        The raw token IDs produced by the model (after the prompt).
    """

    content: str
    thinking: str | None = None
    raw_ids: List[int] | None = None


class BaseLLM(ABC):
    """Abstract base class for chat-style language models."""

    @abstractmethod
    def chat(self, messages: List[Dict[str, str]]) -> ChatResponse:  # noqa: D401,ANN401
        """Return the model's response to a list of chat messages."""
        raise NotImplementedError


    @staticmethod
    def literal_eval(response_content: str):
        """
        Parse a string response into a Python object using ast.literal_eval.

        This method attempts to extract and parse JSON or Python literals from the response content,
        handling various formats like code blocks and special tags.

        Args:
            response_content: The string content to parse.

        Returns:
            The parsed Python object.

        Raises:
            ValueError: If the response content cannot be parsed.
        """
        response_content = response_content.strip()

        thinking_process = None



        # remove content between <think> and </think>, especial for DeepSeek reasoning model
        if "<think>" in response_content and "</think>" in response_content:
            match = re.search(r'<think>(.*?)</think>', response_content, re.DOTALL)
            thinking_process = match.group(1).strip()
            end_of_think = response_content.find("</think>") + len("</think>")
            response_content = response_content[end_of_think:]

        try:
            if response_content.startswith("```") and response_content.endswith("```"):
                if response_content.startswith("```python"):
                    response_content = response_content[9:-3]
                elif response_content.startswith("```json"):
                    response_content = response_content[7:-3]
                elif response_content.startswith("```str"):
                    response_content = response_content[6:-3]
                elif response_content.startswith("```\n"):
                    response_content = response_content[4:-3]
                else:
                    raise ValueError("Invalid code block format")
            result = ast.literal_eval(response_content.strip())
        except Exception:
            matches = re.findall(r"(\[.*?\]|\{.*?\})", response_content, re.DOTALL)

            if len(matches) != 1:
                raise ValueError(
                    f"Invalid JSON/List format for response content:\n{response_content}"
                )

            json_part = matches[0]
            return ast.literal_eval(json_part)

        return result, thinking_process

class QwenLLM(BaseLLM):
    """A minimal chat wrapper around *Qwen/Qwen3-8B* (or any other Qwen model).

    The class supports Qwen's optional "thinking mode" by splitting the generated
    sequence on the ``</think>`` special token (ID = ``151668``).
    """

    # Token ID that marks the end of the thinking block for Qwen-style models
    _THINK_END_TOKEN_ID = 151668

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-14B",
        *,
        torch_dtype: str | torch.dtype = "auto",
        device_map: str | Dict | None = "auto",
        max_new_tokens: int = 32768,  # allow *very* long completions by default
        thinking: bool = True,
        do_sample: bool = True,
        **generate_kwargs,
    ) -> None:
        """Load the model and tokenizer once at construction time.

        Parameters
        ----------
        model_name : str, optional
            The Hugging Face model repo or local path.
        torch_dtype : str | torch.dtype, optional
            Passed to :pyfunc:`transformers.AutoModelForCausalLM.from_pretrained`.
        device_map : str | dict | None, optional
            Device placement strategy ("auto", explicit dict, etc.).
        max_new_tokens : int, optional
            Maximum number of *new* tokens to generate.  Defaults to ``32 768`` so
            you rarely need to worry about truncation.
        thinking : bool, optional
            If *True*, enable Qwen's special "thinking" mode so that the model
            emits internal reasoning wrapped in ``<think>...</think>``.  The
            reasoning is parsed out of the final reply but is still returned in
            the :class:`ChatResponse` object.
        do_sample : bool, optional
            Whether to sample (vs. greedy decoding).  All other sampling /
            decoding kwargs can be forwarded via ``**generate_kwargs``.
        generate_kwargs : dict, optional
            Extra keyword arguments forwarded to ``model.generate``.
        """

        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch_dtype, device_map=device_map
        )
        self.max_new_tokens = max_new_tokens
        self.enable_thinking = thinking
        self.do_sample = do_sample
        self.generate_kwargs = generate_kwargs

    # ---------------------------------------------------------------------
    # Private helpers
    # ---------------------------------------------------------------------
    def _build_inputs(self, messages: List[Dict[str, str]]):
        """Convert *messages* to model-ready tensors using Qwen chat template."""

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.enable_thinking,
        )
        return self.tokenizer([text], return_tensors="pt").to(self.model.device)

    def _split_thinking(self, output_ids: List[int]):
        """Return (thinking_str, content_str) from raw *output_ids*."""

        try:
            idx = len(output_ids) - output_ids[::-1].index(self._THINK_END_TOKEN_ID)
        except ValueError:  # no thinking tokens present
            idx = 0

        thinking_ids = output_ids[:idx]
        content_ids = output_ids[idx:]

        thinking = self.tokenizer.decode(
            thinking_ids, skip_special_tokens=True
        ).strip()
        content = self.tokenizer.decode(
            content_ids, skip_special_tokens=True
        ).strip()
        return thinking, content

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def chat(self, messages: List[Dict[str, str]]) -> ChatResponse:  # noqa: D401
        """Generate a reply given *messages* using the wrapped model."""

        model_inputs = self._build_inputs(messages)
        generated = self.model.generate(
            **model_inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=self.do_sample,
            eos_token_id=self.tokenizer.eos_token_id,
            **self.generate_kwargs,
        )

        # Extract only the newly produced tokens (after the prompt)
        output_ids = generated[0][len(model_inputs.input_ids[0]) :].tolist()
        thinking, content = self._split_thinking(output_ids)

        return ChatResponse(content=content, thinking=thinking, raw_ids=output_ids)


# ---------------------------------------------------------------------------
# Example usage (will run only when this file is executed directly)
# ---------------------------------------------------------------------------
# if __name__ == "__main__":  # pragma: no cover
#     qwen = QwenLLM()  # override default if desired
#     reply = qwen.chat([
#         {"role": "system", "content": "You are a helpful assistant."},
#         {"role": "user", "content": "Give me a brief introduction, what is RAG?"}
#     ])
#     print("Thinking:\n", reply.thinking)
#     print("Answer:\n", reply.content)
