#!/usr/bin/env python3
"""
3-Agent Episodic IPD with LLM Agents
Extends IPD-LLM-Agents2 from 2 agents to 3 agents.

Payoff model (true N-player IPD, matching TABLE I):
  Each agent makes ONE decision per round (COOPERATE or DEFECT).
  Each agent's payoff depends on its action AND k_C = count of OTHER agents cooperating:
    COOPERATE: R=3 if k_C==N-1 (all others cooperate), else min(k_C, R)
    DEFECT:    T=5 if k_C==N-1 (lone defector),        else min(P+k_C, T)
  TABLE I N=3 values: (C,C,C)=3,3,3  (D,C,C)=5,1,1  (D,D,C)=2,2,0  (D,D,D)=1,1,1

Changes from episodic_ipd_game.py (2-agent):
  - __init__:        accepts agent_2; total_scores tracks 3 agents.
  - play_round:      gets decisions from 3 agents; computes true N-player payoffs
                     using k_C counting; updates 3 histories.
  - play_episode:    tracks cooperations for 3 agents; gets 3 reflections;
                     resets/adds reflection for 3 agents.
  - play_game:       prints summary for 3 agents; builds results dict with
                     agent_2 fields, host_2, model_2.
  - _get_agent_decision_with_retry:
                     passes opp1_score and opp2_score to format_round_prompt.
  - _get_reflection: passes opp1_score and opp2_score.
  - _print_summary:  prints 3-agent per-episode table.
  - main():          adds --model-2 / --host-2 args; creates agent_2.
"""

import json
import time
import socket
import getpass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

from ollama_agent import OllamaAgent
from prompts import (
    load_system_prompt,
    load_reflection_template,
    DEFAULT_SYSTEM_PROMPT,
    format_round_prompt,
    format_episode_reflection_prompt,
    extract_decision
)
from config import EpisodeConfig


class EpisodicIPDGame:
    """Manages a 3-agent episodic IPD game"""

    def __init__(
        self,
        agent_0: OllamaAgent,
        agent_1: OllamaAgent,
        agent_2: OllamaAgent,          # NEW: third agent
        config: EpisodeConfig,
        system_prompt_text: str = "",
        reflection_template_text: str = ""
    ):
        self.agent_0 = agent_0
        self.agent_1 = agent_1
        self.agent_2 = agent_2         # NEW
        self.config = config
        self.system_prompt_text = system_prompt_text
        self.reflection_template_text = reflection_template_text

        config.validate()

        # Track cumulative scores for all 3 agents
        self.total_scores = {0: 0, 1: 0, 2: 0}
        self.all_episodes = []

    # ------------------------------------------------------------------
    # Core game loop
    # ------------------------------------------------------------------
    def play_round(
        self,
        round_num: int,
        episode_num: int,
        episode_history_0: List[Dict],
        episode_history_1: List[Dict],
        episode_history_2: List[Dict],
        episode_scores: Dict[int, int]
    ) -> Tuple[str, str, str, Dict]:
        """
        Play one round with 3 agents.

        Returns:
            (action_0, action_1, action_2, round_data)
        """
        if self.config.verbose:
            print(f"  Round {round_num + 1}/{self.config.rounds_per_episode}", end=" ", flush=True)

        # --- Collect decisions from all three agents ---
        action_0, reasoning_0 = self._get_agent_decision_with_retry(
            self.agent_0, round_num, episode_num,
            episode_history_0,
            episode_scores[0], episode_scores[1], episode_scores[2],
            agent_idx=0
        )
        action_1, reasoning_1 = self._get_agent_decision_with_retry(
            self.agent_1, round_num, episode_num,
            episode_history_1,
            episode_scores[1], episode_scores[0], episode_scores[2],
            agent_idx=1
        )
        action_2, reasoning_2 = self._get_agent_decision_with_retry(
            self.agent_2, round_num, episode_num,
            episode_history_2,
            episode_scores[2], episode_scores[0], episode_scores[1],
            agent_idx=2
        )

        # --- True N-player IPD payoffs (TABLE I formula) ---
        N = 3
        actions_map = {0: action_0, 1: action_1, 2: action_2}
        total_coop = sum(1 for a in actions_map.values() if a == 'COOPERATE')

        payoffs_map = {}
        for i in range(N):
            # k_C = number of OTHER agents who cooperated this round
            if actions_map[i] == 'COOPERATE':
                k_C = total_coop - 1   # subtract self from cooperator count
            else:
                k_C = total_coop       # self is D; all cooperators are others
            payoffs_map[i] = self.config.payoff_for_agent(actions_map[i], k_C, N)

        payoff_0 = payoffs_map[0]
        payoff_1 = payoffs_map[1]
        payoff_2 = payoffs_map[2]

        # --- Update episode and total scores ---
        episode_scores[0] += payoff_0
        episode_scores[1] += payoff_1
        episode_scores[2] += payoff_2

        self.total_scores[0] += payoff_0
        self.total_scores[1] += payoff_1
        self.total_scores[2] += payoff_2

        # --- Update per-agent histories ---
        # Each agent's history records what IT saw: its own action, both opponents' actions,
        # and its total payoff.  Opponent labels are opp1/opp2 from each agent's perspective.
        episode_history_0.append({
            'my_action':   action_0,
            'opp1_action': action_1,
            'opp2_action': action_2,
            'my_payoff':   payoff_0,
            'opp1_payoff': payoff_1,
            'opp2_payoff': payoff_2
        })
        episode_history_1.append({
            'my_action':   action_1,
            'opp1_action': action_0,
            'opp2_action': action_2,
            'my_payoff':   payoff_1,
            'opp1_payoff': payoff_0,
            'opp2_payoff': payoff_2
        })
        episode_history_2.append({
            'my_action':   action_2,
            'opp1_action': action_0,
            'opp2_action': action_1,
            'my_payoff':   payoff_2,
            'opp1_payoff': payoff_0,
            'opp2_payoff': payoff_1
        })

        # --- Record round data for JSON output ---
        round_data = {
            'round':                  round_num + 1,
            'agent_0_action':         action_0,
            'agent_1_action':         action_1,
            'agent_2_action':         action_2,
            'agent_0_reasoning':      reasoning_0,
            'agent_1_reasoning':      reasoning_1,
            'agent_2_reasoning':      reasoning_2,
            # Round payoffs (true N-player TABLE I formula)
            'agent_0_payoff':         payoff_0,
            'agent_1_payoff':         payoff_1,
            'agent_2_payoff':         payoff_2,
            # Cumulative episode scores after this round
            'agent_0_episode_score':  episode_scores[0],
            'agent_1_episode_score':  episode_scores[1],
            'agent_2_episode_score':  episode_scores[2]
        }

        if self.config.verbose:
            actions = f"{action_0[0]}{action_1[0]}{action_2[0]}"
            print(f"→ {actions} ({payoff_0},{payoff_1},{payoff_2})", flush=True)

        return action_0, action_1, action_2, round_data

    def play_episode(self, episode_num: int) -> Dict:
        """Play one complete episode with all 3 agents."""
        print(f"\n{'='*80}", flush=True)
        print(f"PERIOD {episode_num + 1}/{self.config.num_episodes}", flush=True)
        print(f"{'='*80}", flush=True)

        episode_history_0: List[Dict] = []
        episode_history_1: List[Dict] = []
        episode_history_2: List[Dict] = []
        episode_scores = {0: 0, 1: 0, 2: 0}
        round_details = []

        for round_num in range(self.config.rounds_per_episode):
            action_0, action_1, action_2, round_data = self.play_round(
                round_num, episode_num,
                episode_history_0, episode_history_1, episode_history_2,
                episode_scores
            )
            round_details.append(round_data)

        # Episode statistics
        coop_0 = sum(1 for r in episode_history_0 if r['my_action'] == 'COOPERATE')
        coop_1 = sum(1 for r in episode_history_1 if r['my_action'] == 'COOPERATE')
        coop_2 = sum(1 for r in episode_history_2 if r['my_action'] == 'COOPERATE')

        print(f"\nPeriod {episode_num + 1} complete:", flush=True)
        print(f"  Agent 0: {episode_scores[0]} pts ({coop_0}/{self.config.rounds_per_episode} coop)", flush=True)
        print(f"  Agent 1: {episode_scores[1]} pts ({coop_1}/{self.config.rounds_per_episode} coop)", flush=True)
        print(f"  Agent 2: {episode_scores[2]} pts ({coop_2}/{self.config.rounds_per_episode} coop)", flush=True)

        # Reflections
        print("\nGetting reflections...", flush=True)
        reflection_0 = self._get_reflection(
            self.agent_0, episode_num, episode_history_0,
            episode_scores[0], episode_scores[1], episode_scores[2]
        )
        reflection_1 = self._get_reflection(
            self.agent_1, episode_num, episode_history_1,
            episode_scores[1], episode_scores[0], episode_scores[2]
        )
        reflection_2 = self._get_reflection(
            self.agent_2, episode_num, episode_history_2,
            episode_scores[2], episode_scores[0], episode_scores[1]
        )

        # Context management
        if self.config.reset_conversation_between_episodes:
            for agent, reflection, idx in [
                (self.agent_0, reflection_0, 0),
                (self.agent_1, reflection_1, 1),
                (self.agent_2, reflection_2, 2),
            ]:
                agent.reset_conversation(keep_system_prompt=True)
                context = f"PREVIOUS PERIOD {episode_num + 1} REFLECTION:\n{reflection}\n"
                agent.add_reflection_to_context(context)

        n = self.config.rounds_per_episode
        episode_data = {
            'episode': episode_num + 1,
            'rounds':  round_details,
            'agent_0': {
                'episode_score':     episode_scores[0],
                'cooperations':      coop_0,
                'cooperation_rate':  coop_0 / n,
                'reflection':        reflection_0
            },
            'agent_1': {
                'episode_score':     episode_scores[1],
                'cooperations':      coop_1,
                'cooperation_rate':  coop_1 / n,
                'reflection':        reflection_1
            },
            'agent_2': {
                'episode_score':     episode_scores[2],
                'cooperations':      coop_2,
                'cooperation_rate':  coop_2 / n,
                'reflection':        reflection_2
            }
        }
        return episode_data

    def play_game(self) -> Dict:
        """Play the full multi-episode game and return results dict."""
        print(f"\n{'='*80}", flush=True)
        print("3-AGENT EPISODIC IPD SIMULATION", flush=True)
        print(f"{'='*80}", flush=True)
        print(f"Episodes:           {self.config.num_episodes}", flush=True)
        print(f"Rounds per episode: {self.config.rounds_per_episode}", flush=True)
        print(f"History window:     {self.config.history_window_size} rounds", flush=True)
        print(f"Total rounds:       {self.config.total_rounds}", flush=True)
        print(f"Agent 0: {self.agent_0.model} @ {self.config.host_0}", flush=True)
        print(f"Agent 1: {self.agent_1.model} @ {self.config.host_1}", flush=True)
        print(f"Agent 2: {self.agent_2.model} @ {self.config.host_2}", flush=True)
        print(f"Temperature:        {self.config.temperature}", flush=True)
        print(f"Reset between eps:  {self.config.reset_conversation_between_episodes}", flush=True)
        print(f"{'='*80}", flush=True)

        start_time = time.time()

        for episode_num in range(self.config.num_episodes):
            episode_data = self.play_episode(episode_num)
            self.all_episodes.append(episode_data)

        elapsed = time.time() - start_time

        total_coop_0 = sum(ep['agent_0']['cooperations'] for ep in self.all_episodes)
        total_coop_1 = sum(ep['agent_1']['cooperations'] for ep in self.all_episodes)
        total_coop_2 = sum(ep['agent_2']['cooperations'] for ep in self.all_episodes)
        total_r = self.config.total_rounds

        results = {
            'timestamp':   datetime.now().isoformat(),
            'hostname':    socket.gethostname(),
            'username':    getpass.getuser(),
            'host_0':      self.config.host_0,
            'host_1':      self.config.host_1,
            'host_2':      self.config.host_2,       # NEW
            'prompts': {
                'system_prompt':      self.system_prompt_text,
                'reflection_template': self.reflection_template_text
            },
            'config': {
                'num_episodes':             self.config.num_episodes,
                'rounds_per_episode':       self.config.rounds_per_episode,
                'total_rounds':             self.config.total_rounds,
                'history_window_size':      self.config.history_window_size,
                'temperature':              self.config.temperature,
                'reset_between_episodes':   self.config.reset_conversation_between_episodes,
                'reflection_type':          self.config.reflection_prompt_type,
                'model_0':                  self.config.model_0,
                'model_1':                  self.config.model_1,
                'model_2':                  self.config.model_2,   # NEW
                'decision_token_limit':     self.config.decision_token_limit,
                'reflection_token_limit':   self.config.reflection_token_limit,
                'http_timeout':             self.config.http_timeout,
                'force_decision_retries':   self.config.force_decision_retries
            },
            'elapsed_seconds': elapsed,
            'agent_0': {
                'model':                    self.agent_0.model,
                'total_score':              self.total_scores[0],
                'total_cooperations':       total_coop_0,
                'overall_cooperation_rate': total_coop_0 / total_r
            },
            'agent_1': {
                'model':                    self.agent_1.model,
                'total_score':              self.total_scores[1],
                'total_cooperations':       total_coop_1,
                'overall_cooperation_rate': total_coop_1 / total_r
            },
            'agent_2': {                             # NEW
                'model':                    self.agent_2.model,
                'total_score':              self.total_scores[2],
                'total_cooperations':       total_coop_2,
                'overall_cooperation_rate': total_coop_2 / total_r
            },
            'episodes': self.all_episodes
        }

        self._print_summary(results)
        return results

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------
    def _get_agent_decision_with_retry(
        self,
        agent: OllamaAgent,
        round_num: int,
        episode_num: int,
        history: List[Dict],
        my_score: int,
        opp1_score: int,
        opp2_score: int,
        agent_idx: int
    ) -> Tuple[str, str]:
        """Get a definite decision from an agent with retry logic."""
        prompt = format_round_prompt(
            round_num, episode_num, history,
            my_score, opp1_score, opp2_score,
            self.config.history_window_size
        )

        decision, response = agent.generate_with_forced_decision(prompt, extract_decision)

        if decision is None:
            print(f"  ⚠️  CRITICAL: {agent.agent_id} failed to decide after all retries", flush=True)
            print(f"      Defaulting to DEFECT", flush=True)
            if response:
                print(f"      Last response: {response[:200]}...", flush=True)
            return 'DEFECT', response or "Failed to respond after retries"

        return decision, response

    def _get_reflection(
        self,
        agent: OllamaAgent,
        episode_num: int,
        history: List[Dict],
        my_score: int,
        opp1_score: int,
        opp2_score: int
    ) -> str:
        """Get post-episode reflection from agent."""
        # Pass template_file only when reflection_prompt_type is "custom";
        # otherwise format_episode_reflection_prompt uses built-in templates.
        template_file = (
            self.config.reflection_template_file
            if self.config.reflection_prompt_type == "custom"
            else None
        )
        prompt = format_episode_reflection_prompt(
            episode_num, history,
            my_score, opp1_score, opp2_score,
            self.config.rounds_per_episode,
            self.config.reflection_prompt_type,
            self.config.include_statistics,
            template_file=template_file
        )
        reflection = agent.generate(prompt, is_reflection=True)
        return reflection if reflection is not None else "Agent failed to provide reflection"

    def _print_summary(self, results: Dict):
        """Print final game summary."""
        print(f"\n{'='*80}", flush=True)
        print("FINAL SUMMARY (3 AGENTS)", flush=True)
        print(f"{'='*80}", flush=True)
        print(f"Total episodes: {results['config']['num_episodes']}", flush=True)
        print(f"Total rounds:   {results['config']['total_rounds']}", flush=True)
        print(f"Time elapsed:   {results['elapsed_seconds']:.1f}s", flush=True)
        print(flush=True)
        print("OVERALL RESULTS:", flush=True)
        for idx in range(3):
            a = results[f'agent_{idx}']
            print(f"  Agent {idx}: {a['total_score']} pts "
                  f"({a['overall_cooperation_rate']*100:.1f}% coop)", flush=True)
        print(flush=True)
        print("BY EPISODE:", flush=True)
        for ep in results['episodes']:
            line = f"  Period {ep['episode']}:  "
            for idx in range(3):
                a = ep[f'agent_{idx}']
                line += (f"Agent {idx}: {a['episode_score']} pts "
                         f"({a['cooperation_rate']*100:.0f}% coop)  ")
            print(line, flush=True)
        print(f"{'='*80}\n", flush=True)


# ======================================================================
# CLI entry point
# ======================================================================
def main():
    import argparse

    parser = argparse.ArgumentParser(description="3-Agent Episodic IPD with LLM Agents")
    parser.add_argument("--episodes",       type=int,   default=5)
    parser.add_argument("--rounds",         type=int,   default=20)
    parser.add_argument("--history-window", type=int,   default=10)
    parser.add_argument("--temperature",    type=float, default=0.7)

    # Agent 0
    parser.add_argument("--model-0", type=str, default="llama3:8b-instruct-q5_K_M")
    parser.add_argument("--host-0",  type=str, default="tungsten")
    # Agent 1
    parser.add_argument("--model-1", type=str, default="llama3:8b-instruct-q5_K_M")
    parser.add_argument("--host-1",  type=str, default="tungsten")
    # Agent 2 — NEW
    parser.add_argument("--model-2", type=str, default="llama3:8b-instruct-q5_K_M")
    parser.add_argument("--host-2",  type=str, default="tungsten")

    parser.add_argument("--no-reset",        action="store_true")
    parser.add_argument("--reflection-type", type=str, default="standard",
                        choices=["minimal", "standard", "detailed"])
    parser.add_argument("--system-prompt",   type=str, default="system_prompt.txt")
    parser.add_argument("--reflection-template", type=str,
                        default="reflection_prompt_template.txt")
    parser.add_argument("--output",          type=str, default=None)
    parser.add_argument("--quiet",           action="store_true")
    parser.add_argument("--decision-tokens", type=int, default=256)
    parser.add_argument("--reflection-tokens", type=int, default=1024)
    parser.add_argument("--http-timeout",    type=int, default=60)
    parser.add_argument("--force-retries",   type=int, default=2)
    parser.add_argument("--comment",         type=str, default=None)

    args = parser.parse_args()

    # Load prompts
    try:
        system_prompt = load_system_prompt(args.system_prompt)
        print(f"Loaded system prompt from: {args.system_prompt}", flush=True)
    except FileNotFoundError as e:
        print(f"Warning: {e}", flush=True)
        print("Using default system prompt", flush=True)
        system_prompt = DEFAULT_SYSTEM_PROMPT

    try:
        reflection_template = load_reflection_template(args.reflection_template)
        print(f"Loaded reflection template from: {args.reflection_template}", flush=True)
    except FileNotFoundError:
        reflection_template = ""

    # Configuration
    config = EpisodeConfig(
        num_episodes=args.episodes,
        rounds_per_episode=args.rounds,
        history_window_size=args.history_window,
        temperature=args.temperature,
        model_0=args.model_0,
        host_0=args.host_0,
        model_1=args.model_1,
        host_1=args.host_1,
        model_2=args.model_2,
        host_2=args.host_2,
        reset_conversation_between_episodes=not args.no_reset,
        reflection_prompt_type=args.reflection_type,
        verbose=not args.quiet,
        decision_token_limit=args.decision_tokens,
        reflection_token_limit=args.reflection_tokens,
        http_timeout=args.http_timeout,
        force_decision_retries=args.force_retries
    )

    # Create agents
    print("Initializing 3 agents...", flush=True)
    shared = dict(
        temperature=config.temperature,
        system_prompt=system_prompt,
        decision_token_limit=config.decision_token_limit,
        reflection_token_limit=config.reflection_token_limit,
        http_timeout=config.http_timeout,
        force_decision_retries=config.force_decision_retries
    )

    agent_0 = OllamaAgent(agent_id="agent_0", model=config.model_0, host=config.host_0, **shared)
    agent_1 = OllamaAgent(agent_id="agent_1", model=config.model_1, host=config.host_1, **shared)
    agent_2 = OllamaAgent(agent_id="agent_2", model=config.model_2, host=config.host_2, **shared)

    # Play game
    game = EpisodicIPDGame(
        agent_0, agent_1, agent_2, config,
        system_prompt_text=system_prompt,
        reflection_template_text=reflection_template
    )
    results = game.play_game()

    if args.comment:
        results = {'comment': args.comment, **results}

    # Save results
    if args.output:
        output_path = Path(args.output)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path(__file__).parent / "results" / f"3agent_game_{timestamp}.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to: {output_path}", flush=True)


if __name__ == "__main__":
    main()
