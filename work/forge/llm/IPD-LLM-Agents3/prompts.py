"""
Prompt templates for 3-Agent Episodic IPD LLM Agents

Key changes from IPD-LLM-Agents2 (2-agent version):
  - DEFAULT_SYSTEM_PROMPT updated: mentions TWO opponents instead of one.
    Payoff explanations clarify that each round's payoff is the SUM of two
    bilateral interactions (max cooperative = 6 per round, max exploitative = 10).
  - format_round_prompt: history now shows both opponents' actions per round.
  - format_episode_reflection_prompt: shows both opponents' cooperation counts
    and the agent's per-round payoffs from two interactions.
  - reflection_prompt_template.txt variables extended with opp2_* fields.
"""

from typing import List, Dict
from pathlib import Path


def load_system_prompt(prompt_file: str = "system_prompt.txt") -> str:
    prompt_path = Path(prompt_file)
    if not prompt_path.exists():
        raise FileNotFoundError(f"System prompt file not found: {prompt_file}")
    with open(prompt_path, 'r') as f:
        return f.read()


def load_reflection_template(template_file: str = "reflection_prompt_template.txt") -> str:
    template_path = Path(template_file)
    if not template_path.exists():
        raise FileNotFoundError(f"Reflection template file not found: {template_file}")
    with open(template_path, 'r') as f:
        return f.read()


# ---------------------------------------------------------------------------
# Default system prompt — updated for 3 agents
# ---------------------------------------------------------------------------
DEFAULT_SYSTEM_PROMPT = """You are participating in a repeated exchange with TWO other participants across multiple periods.

Each round, you MUST choose exactly one action: COOPERATE or DEFECT
Both other participants also each choose independently.
All three players must commit to a definite action — hedging or ambiguity is not permitted.

Your payoff each round depends on your action and how many of the other players cooperate:

Payoff table (k = number of OTHER players who cooperate):
  You COOPERATE:  k=0 → 0 pts  |  k=1 → 1 pt  |  k=2 → 3 pts (mutual cooperation)
  You DEFECT:     k=0 → 1 pt   |  k=1 → 2 pts  |  k=2 → 5 pts (lone defector)

Example: If you COOPERATE and Other1 COOPERATES but Other2 DEFECTS:
  k=1 → Your round payoff = 1 point

YOUR GOAL: Accumulate as many points as possible across all periods.

You will interact for multiple periods. After each period, you can reflect on
what happened and adjust your approach. Learn from experience to improve your
outcomes over time.

RESPONSE FORMAT - THIS IS MANDATORY:
First, provide your reasoning in 2-3 complete sentences:
- What patterns do you observe in the recent interactions with both opponents?
- Why are you choosing to cooperate or defect?
- What outcome do you expect?

In your reasoning, use lowercase words ("cooperate", "defect") when discussing options.

Then, you MUST end your response with EXACTLY one of these words on its own line:
COOPERATE
or
DEFECT

CRITICAL REQUIREMENTS:
- You MUST make a definite choice — no hedging, no "maybe", no conditional statements
- The final line must contain ONLY the word COOPERATE or ONLY the word DEFECT
- Nothing else on that line — no punctuation, no additional words
- This is a game-theoretic requirement: all players must commit to definite actions

CORRECT EXAMPLE:
"Both opponents cooperated last round and I earned 3 points. This pattern
of mutual cooperation is sustainable. I will continue cooperating.

COOPERATE"

INCORRECT EXAMPLES:
"I might cooperate" — NO: must be definite
"COOPERATE or DEFECT" — NO: must choose one
"I'll COOPERATE this time" — NO: decision must be on its own line
"COOPERATE." — NO: no punctuation on decision line
"My choice is COOPERATE" — NO: decision word must be alone on the line

DO NOT just write a single word like "cooperate" or "defect" as your reasoning.
Always explain your thinking in complete sentences, then provide your action.
"""


# ---------------------------------------------------------------------------
# Round prompt — now shows two opponents' actions
# ---------------------------------------------------------------------------
def format_round_prompt(
    round_num: int,
    episode_num: int,
    history: List[Dict],
    my_score: int,
    opp1_score: int,
    opp2_score: int,
    window_size: int = 10
) -> str:
    """
    Format prompt for a single round within an episode (3-agent version).

    history entries contain keys:
        my_action, opp1_action, opp2_action, my_payoff, opp1_payoff, opp2_payoff
    """
    if round_num == 0:
        return f"""PERIOD {episode_num + 1}, ROUND 1:
This is the first round of this period. You are playing against TWO opponents simultaneously.

What is your choice?"""

    prompt = f"PERIOD {episode_num + 1}, ROUND {round_num + 1}:\n\n"
    prompt += f"Your total points:     {my_score}\n"
    prompt += f"Other1's total points: {opp1_score}\n"
    prompt += f"Other2's total points: {opp2_score}\n\n"
    prompt += "Recent interactions (your action vs. BOTH opponents each round):\n"

    recent_history = history[-window_size:] if len(history) > window_size else history
    start_round = len(history) - len(recent_history) + 1

    for i, rd in enumerate(recent_history, start=start_round):
        my  = rd['my_action'].lower()
        op1 = rd['opp1_action'].lower()
        op2 = rd['opp2_action'].lower()
        prompt += (
            f"  Round {i}: You {my}d | Other1 {op1}d | Other2 {op2}d "
            f"(You: +{rd['my_payoff']})\n"
        )

    if len(history) > window_size:
        prompt += f"\n(Showing last {window_size} rounds of {len(history)} total)\n"

    prompt += "\nWhat is your choice?"
    return prompt


# ---------------------------------------------------------------------------
# Reflection prompt — now covers both opponents
# ---------------------------------------------------------------------------
def format_episode_reflection_prompt(
    episode_num: int,
    history: List[Dict],
    my_score: int,
    opp1_score: int,
    opp2_score: int,
    rounds_in_episode: int,
    reflection_type: str = "standard",
    include_statistics: bool = True,
    template_file: str = None
) -> str:
    """
    Format reflection prompt at end of episode (3-agent version).

    history entries contain keys:
        my_action, opp1_action, opp2_action, my_payoff, opp1_payoff, opp2_payoff
    """
    my_coops  = sum(1 for r in history if r['my_action']   == 'COOPERATE')
    opp1_coops = sum(1 for r in history if r['opp1_action'] == 'COOPERATE')
    opp2_coops = sum(1 for r in history if r['opp2_action'] == 'COOPERATE')
    my_avg = my_score / len(history) if history else 0

    if reflection_type == "minimal":
        return f"""PERIOD {episode_num + 1} COMPLETE

Your points:   {my_score}
Other1 points: {opp1_score}
Other2 points: {opp2_score}
"""

    if reflection_type == "custom" and template_file:
        try:
            template = load_reflection_template(template_file)
            round_history = ""
            for i, rd in enumerate(history, start=1):
                round_history += (
                    f"Round {i}: You {rd['my_action']}, Other1 {rd['opp1_action']}, "
                    f"Other2 {rd['opp2_action']} (You: +{rd['my_payoff']})\n"
                )
            prompt = template.format(
                episode_num=episode_num + 1,
                rounds_in_episode=rounds_in_episode,
                my_score=my_score,
                opp1_score=opp1_score,
                opp2_score=opp2_score,
                my_avg=f"{my_avg:.2f}",
                my_cooperations=my_coops,
                my_defections=len(history) - my_coops,
                opp1_cooperations=opp1_coops,
                opp1_defections=len(history) - opp1_coops,
                opp2_cooperations=opp2_coops,
                opp2_defections=len(history) - opp2_coops,
                round_history=round_history.rstrip()
            )
            return prompt
        except FileNotFoundError:
            print(f"Warning: Template file {template_file} not found, falling back to standard")
            reflection_type = "standard"

    if reflection_type == "standard":
        prompt = f"""PERIOD {episode_num + 1} COMPLETE (Rounds 1-{rounds_in_episode})

Your points this period:   {my_score}
Other1's points this period: {opp1_score}
Other2's points this period: {opp2_score}
"""
        if include_statistics:
            prompt += f"""
Your average: {my_avg:.2f} points per round
Your choices:   {my_coops} cooperate, {len(history) - my_coops} defect
Other1's choices: {opp1_coops} cooperate, {len(history) - opp1_coops} defect
Other2's choices: {opp2_coops} cooperate, {len(history) - opp2_coops} defect
"""
        prompt += "\nWhat happened this period:\n"
        for i, rd in enumerate(history, start=1):
            prompt += (
                f"Round {i}: You {rd['my_action']}, Other1 {rd['opp1_action']}, "
                f"Other2 {rd['opp2_action']} (You: +{rd['my_payoff']})\n"
            )
        prompt += "\nAs you continue to the next period, what are you thinking?\n"
        return prompt

    else:  # detailed
        prompt = f"""PERIOD {episode_num + 1} COMPLETE (Rounds 1-{rounds_in_episode})

OUTCOMES:
Your points this period:     {my_score}
Other1's points this period: {opp1_score}
Other2's points this period: {opp2_score}

PERFORMANCE:
Your average: {my_avg:.2f} points per round
Theoretical range: 0 to 5 points per round
Your choices:   {my_coops} cooperate ({my_coops/len(history)*100:.1f}%), {len(history)-my_coops} defect
Other1's choices: {opp1_coops} cooperate ({opp1_coops/len(history)*100:.1f}%), {len(history)-opp1_coops} defect
Other2's choices: {opp2_coops} cooperate ({opp2_coops/len(history)*100:.1f}%), {len(history)-opp2_coops} defect

WHAT HAPPENED:
"""
        for i, rd in enumerate(history, start=1):
            prompt += (
                f"Round {i}: You {rd['my_action']}, Other1 {rd['opp1_action']}, "
                f"Other2 {rd['opp2_action']} (You: +{rd['my_payoff']})\n"
            )
        prompt += "\nReflect on this period and consider your approach for the next period.\n"
        return prompt


# ---------------------------------------------------------------------------
# Decision extractor — unchanged from IPD-LLM-Agents2
# ---------------------------------------------------------------------------
def extract_decision(response: str) -> str:
    """
    Extract COOPERATE or DEFECT from LLM response.
    Returns 'COOPERATE', 'DEFECT', or None if ambiguous.
    """
    if not response:
        return None

    lines = [line.strip() for line in response.strip().split('\n') if line.strip()]
    if not lines:
        return None

    last_line = lines[-1].strip().upper()

    if last_line == 'COOPERATE':
        return 'COOPERATE'
    if last_line == 'DEFECT':
        return 'DEFECT'

    last_line_cleaned = last_line.rstrip('.!,;:')
    if last_line_cleaned == 'COOPERATE':
        return 'COOPERATE'
    if last_line_cleaned == 'DEFECT':
        return 'DEFECT'

    if len(last_line.split()) <= 3:
        has_coop = 'COOPERATE' in last_line
        has_def  = 'DEFECT' in last_line
        if has_coop and not has_def:
            return 'COOPERATE'
        if has_def and not has_coop:
            return 'DEFECT'

    if last_line.endswith('COOPERATE') and 'DEFECT' not in last_line:
        if len(last_line.split()) <= 5:
            return 'COOPERATE'
    if last_line.endswith('DEFECT') and 'COOPERATE' not in last_line:
        if len(last_line.split()) <= 5:
            return 'DEFECT'

    return None
