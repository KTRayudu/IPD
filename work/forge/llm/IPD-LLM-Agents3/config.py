"""
Configuration for 3-Agent Episodic IPD LLM Agents
Extends IPD-LLM-Agents2 EpisodeConfig to support a third agent.

Key change from 2-agent version:
  - Added model_2 / host_2 for the third agent.
  - Uses true N-player IPD payoff (TABLE I), not bilateral sum.
  - Each agent's payoff depends on its action and the number of OTHER agents
    who cooperate (k_C), using formula:
      COOPERATE: R=3 if k_C==N-1 else min(k_C, R)
      DEFECT:    T=5 if k_C==N-1 else min(P+k_C, T)
  - Max possible payoff per round = T = 5 (lone defector, all others cooperate).
  - Max cooperative payoff per round = R = 3 (all cooperate).
"""

from dataclasses import dataclass
from typing import Literal


@dataclass
class EpisodeConfig:
    """Configuration for 3-agent episodic IPD simulation"""

    # Episode structure
    num_episodes: int = 5
    rounds_per_episode: int = 20

    # Context management
    reset_conversation_between_episodes: bool = True
    history_window_size: int = 10

    # Agent parameters — three agents
    temperature: float = 0.7
    model_0: str = "llama3:8b-instruct-q5_K_M"
    host_0: str = "iron"
    model_1: str = "llama3:8b-instruct-q5_K_M"
    host_1: str = "iron"
    model_2: str = "llama3:8b-instruct-q5_K_M"   # NEW: third agent
    host_2: str = "iron"                           # NEW: third agent host

    # LLM generation parameters
    decision_token_limit: int = 256
    reflection_token_limit: int = 1024
    http_timeout: int = 60
    force_decision_retries: int = 2

    # Reflection parameters
    reflection_prompt_type: Literal["minimal", "standard", "detailed", "custom"] = "standard"
    include_statistics: bool = True
    show_other_agent_score: bool = True
    reflection_template_file: str = ""  # path to .txt template; only used when reflection_prompt_type="custom"

    # Payoff matrix (standard IPD values)
    temptation: int = 5   # T
    reward: int = 3       # R
    punishment: int = 1   # P
    sucker: int = 0       # S

    # Output
    verbose: bool = True

    @property
    def total_rounds(self) -> int:
        """Total number of rounds in the simulation"""
        return self.num_episodes * self.rounds_per_episode

    def payoff_for_agent(self, action: str, k_C: int, N: int) -> int:
        """Compute per-agent payoff using the true N-player IPD formula (TABLE I).

        action is 'COOPERATE' or 'DEFECT'; k_C is the count of OTHER agents cooperating; N is total agents.
        COOPERATE yields min(k_C, R) or R when all others cooperate; DEFECT yields min(P+k_C, T) or T when lone defector.
        Returns an integer payoff in range 0–5 (N=3 standard case).
        """
        if action in ('C', 'COOPERATE'):
            return self.reward if k_C == N - 1 else min(k_C, self.reward)
        else:
            return self.temptation if k_C == N - 1 else min(self.punishment + k_C, self.temptation)

    @property
    def payoff_matrix_state(self) -> dict:
        """Return current payoff values as a dict — for inclusion in JSON output."""
        return {
            'temptation': self.temptation,
            'reward':     self.reward,
            'punishment': self.punishment,
            'sucker':     self.sucker
        }

    def validate(self) -> bool:
        """Validate that configuration satisfies IPD constraints"""
        if not (self.temptation > self.reward > self.punishment > self.sucker):
            raise ValueError("Payoff matrix must satisfy T > R > P > S")
        if not (2 * self.reward > self.temptation + self.sucker):
            raise ValueError("Payoff matrix must satisfy 2R > T + S")
        return True


# Predefined configurations

BASELINE_CONFIG = EpisodeConfig(
    num_episodes=5,
    rounds_per_episode=20,
    reset_conversation_between_episodes=True,
    temperature=0.7,
    reflection_prompt_type="standard",
    include_statistics=True,
)

SHORT_LEARNING_CONFIG = EpisodeConfig(
    num_episodes=10,
    rounds_per_episode=10,
    reset_conversation_between_episodes=True,
    temperature=0.7,
    reflection_prompt_type="minimal",
    include_statistics=True,
)

LONG_CONTEXT_CONFIG = EpisodeConfig(
    num_episodes=3,
    rounds_per_episode=50,
    reset_conversation_between_episodes=False,
    temperature=0.7,
    reflection_prompt_type="detailed",
    include_statistics=True,
)

HIGH_EXPLORATION_CONFIG = EpisodeConfig(
    num_episodes=5,
    rounds_per_episode=20,
    reset_conversation_between_episodes=True,
    temperature=1.0,
    reflection_prompt_type="standard",
    include_statistics=True,
)
