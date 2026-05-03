"""
Ollama Agent wrapper for 3-Agent Episodic IPD experiments
Unchanged from IPD-LLM-Agents2 — the agent class is fully agent-count agnostic.
Copied here so IPD-LLM-Agents3 is self-contained.
"""

import requests
from typing import Optional
import time


class OllamaAgent:
    """LLM agent backed by the Ollama HTTP chat API for IPD game play.

    Maintains a running conversation history so each round builds on prior context.
    Supports forced-decision retries and separate token budgets for decisions vs reflections.
    """

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
        """Set up the agent with its model, host, and generation settings.

        Required: agent_id (unique label), model (Ollama model tag), host (server hostname).
        Optional: temperature, decision_token_limit, reflection_token_limit, force_decision_retries.
        Seeds conversation history with system_prompt if provided.
        """
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
        """Send a prompt to the Ollama chat API and return the response text.

        Appends the prompt as a user turn and the reply as an assistant turn to conversation history.
        Uses reflection_token_limit when is_reflection=True, otherwise decision_token_limit.
        Retries up to max_retries times on HTTP errors; returns None if all attempts fail.
        """
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
                print(f"  ⚠️  {self.agent_id} API error (attempt {attempt + 1}/{max_retries}): {e}")
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
        """Get a definite COOPERATE/DEFECT decision, re-prompting if the first response is ambiguous.

        Calls generate() once; if extract_decision_fn returns None, sends a follow-up forcing prompt.
        Retries up to force_decision_retries times before giving up.
        Returns (decision, response_text); decision is None if all retries fail.
        """
        response = self.generate(prompt, num_predict=self.decision_token_limit)

        if response is None:
            return None, None

        decision = extract_decision_fn(response)
        if decision is not None:
            return decision, response

        for retry in range(self.force_decision_retries):
            print(f"  ⚠️  {self.agent_id} gave ambiguous response, forcing decision "
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
        """Clear the conversation history, optionally retaining the system prompt.

        When keep_system_prompt=True (default), re-seeds history with the original system_prompt.
        Call between episodes to prevent context overflow while preserving game framing.
        """
        if keep_system_prompt and self.system_prompt:
            self.conversation = [{"role": "system", "content": self.system_prompt}]
        else:
            self.conversation = []

    def add_reflection_to_context(self, reflection_text: str):
        """Inject a prior-episode reflection into the conversation as a user turn.

        Used after reset_conversation to carry strategic memory into the next episode.
        The reflection is appended verbatim so the LLM sees it before the first round prompt.
        """
        self.conversation.append({"role": "user", "content": reflection_text})

    def get_conversation_length(self) -> int:
        """Return the current number of messages in conversation history (system + user + assistant turns)."""
        return len(self.conversation)

    def __repr__(self) -> str:
        """Return a concise string representation showing agent id, model, and conversation length."""
        return f"OllamaAgent(id={self.agent_id}, model={self.model}, conv_length={len(self.conversation)})"
