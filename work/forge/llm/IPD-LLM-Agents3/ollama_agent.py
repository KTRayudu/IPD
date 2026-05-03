"""
Ollama Agent wrapper for 3-Agent Episodic IPD experiments
Unchanged from IPD-LLM-Agents2 — the agent class is fully agent-count agnostic.
Copied here so IPD-LLM-Agents3 is self-contained.
"""

import requests
from typing import Optional
import time


class OllamaAgent:
    """An agent that uses Ollama LLM for decision-making in IPD"""

    def __init__(
        self,
        agent_id: str,
        model: str,
        host: str = "iron",
        port: int = 11434,
        temperature: float = 0.7,
        system_prompt: str = "",
        decision_token_limit: int = 256,
        reflection_token_limit: int = 1024,
        http_timeout: int = 60,
        force_decision_retries: int = 2
    ):
        self.agent_id = agent_id
        self.model = model
        self.base_url = f"http://{host}:{port}"
        self.temperature = temperature
        self.system_prompt = system_prompt

        self.decision_token_limit = decision_token_limit
        self.reflection_token_limit = reflection_token_limit
        self.http_timeout = http_timeout
        self.force_decision_retries = force_decision_retries

        self.conversation = []
        if system_prompt:
            self.conversation.append({
                "role": "system",
                "content": system_prompt
            })

    def generate(
        self,
        prompt: str,
        max_retries: int = 3,
        num_predict: int = None,
        is_reflection: bool = False
    ) -> Optional[str]:
        if num_predict is None:
            num_predict = self.reflection_token_limit if is_reflection else self.decision_token_limit

        self.conversation.append({"role": "user", "content": prompt})

        url = f"{self.base_url}/api/chat"
        payload = {
            "model": self.model,
            "messages": self.conversation,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": num_predict
            }
        }

        for attempt in range(max_retries):
            try:
                response = requests.post(url, json=payload, timeout=self.http_timeout)
                response.raise_for_status()
                result = response.json()
                assistant_message = result['message']['content']
                self.conversation.append({
                    "role": "assistant",
                    "content": assistant_message
                })
                return assistant_message

            except requests.exceptions.RequestException as e:
                print(f"   {self.agent_id} API error (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2)
                else:
                    return None

        return None

    def generate_with_forced_decision(
        self,
        prompt: str,
        extract_decision_fn
    ) -> tuple[Optional[str], Optional[str]]:
        response = self.generate(prompt, num_predict=self.decision_token_limit)

        if response is None:
            return None, None

        decision = extract_decision_fn(response)
        if decision is not None:
            return decision, response

        for retry in range(self.force_decision_retries):
            print(f"   {self.agent_id} gave ambiguous response, forcing decision "
                  f"(attempt {retry + 1}/{self.force_decision_retries})")

            force_prompt = """Your previous response did not clearly specify COOPERATE or DEFECT.

You MUST choose exactly one action. This is a fundamental requirement of the game.

Respond with ONLY your reasoning (2-3 sentences) followed by exactly one word on its own line:
COOPERATE
or
DEFECT

What is your decision?"""

            response = self.generate(force_prompt, num_predict=self.decision_token_limit)
            if response is None:
                continue

            decision = extract_decision_fn(response)
            if decision is not None:
                return decision, f"[FORCED DECISION AFTER {retry + 1} RETRIES]\n{response}"

        return None, response

    def reset_conversation(self, keep_system_prompt: bool = True):
        if keep_system_prompt and self.system_prompt:
            self.conversation = [{"role": "system", "content": self.system_prompt}]
        else:
            self.conversation = []

    def add_reflection_to_context(self, reflection_text: str):
        self.conversation.append({"role": "user", "content": reflection_text})

    def get_conversation_length(self) -> int:
        return len(self.conversation)

    def __repr__(self) -> str:
        return f"OllamaAgent(id={self.agent_id}, model={self.model}, conv_length={len(self.conversation)})"
