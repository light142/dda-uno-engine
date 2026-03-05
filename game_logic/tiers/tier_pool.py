"""Tier model pool: loads and caches agents by tier name.

Handles DQN tiers (loaded from disk), RandomAgent, rule-v1,
and heuristic bots (noob/casual/pro). The model directory is
configurable so both the simulator and API can share this class.
"""

import os
import glob
from typing import Optional

from engine.config.game import NUM_ACTIONS
from engine.game_logic.tiers.tier_config import (
    AGENT_CHOICES, AGENT_ALIASES, DQN_TIERS, resolve_agent_name,
)

# Default model directory: simulator/models/ relative to project root
_DEFAULT_MODEL_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
    "simulator", "models",
)


def resolve_model_path(tier_name: str, model_dir: str = None) -> Optional[str]:
    """Find the best available model file for a DQN tier.

    Priority: {tier}_agent.pt > latest checkpoint_*.pt
    Returns None if no model file found.

    Args:
        tier_name: Name of the tier (e.g., "adversarial").
        model_dir: Base directory containing tier subdirectories.
            Defaults to simulator/models/.
    """
    if model_dir is None:
        model_dir = _DEFAULT_MODEL_DIR

    base_dir = os.path.join(model_dir, tier_name)
    if not os.path.isdir(base_dir):
        return None

    # Try preferred filename first
    preferred = os.path.join(base_dir, f"{tier_name}_agent.pt")
    if os.path.exists(preferred):
        return preferred

    # Fallback: latest checkpoint
    pattern = os.path.join(base_dir, "checkpoint_*.pt")
    files = glob.glob(pattern)
    if not files:
        return None

    best_path, best_ep = None, 0
    for f in files:
        name = os.path.basename(f)
        try:
            ep = int(name.replace("checkpoint_", "").replace(".pt", ""))
            if ep > best_ep:
                best_ep = ep
                best_path = f
        except ValueError:
            continue

    return best_path


class TierModelPool:
    """Loads agent models once at startup, provides cached agents on demand.

    Handles DQN tiers (loaded from disk), RandomAgent, rule-v1,
    and heuristic bots (noob/casual/pro).
    """

    def __init__(self, tiers_to_load: list = None, model_dir: str = None):
        """Load the requested tiers.

        Args:
            tiers_to_load: List of tier/agent names to load. If None,
                loads all AGENT_CHOICES.
            model_dir: Directory containing model subdirectories.
                Defaults to simulator/models/.
        """
        self._agents = {}
        self._model_dir = model_dir or _DEFAULT_MODEL_DIR

        if tiers_to_load is None:
            tiers_to_load = list(AGENT_CHOICES)

        for name in set(tiers_to_load):
            resolved = resolve_agent_name(name)
            self._load(resolved)

    def _load(self, name: str):
        """Load a single agent by name."""
        if name in self._agents:
            return

        if name in ("random", "random-vd"):
            from rlcard.agents import RandomAgent
            self._agents[name] = RandomAgent(num_actions=NUM_ACTIONS)

        elif name == "rule-v1":
            from rlcard.models import load as load_model
            self._agents[name] = load_model('uno-rule-v1').agents[0]

        elif name in ("noob", "casual", "pro"):
            from engine.game_logic.bots import get_bot
            self._agents[name] = get_bot(name)

        elif name in DQN_TIERS:
            path = resolve_model_path(name, self._model_dir)
            if path is None:
                from rlcard.agents import RandomAgent
                print(f"  WARNING: No model found for '{name}', using random fallback")
                self._agents[name] = RandomAgent(num_actions=NUM_ACTIONS)
            else:
                from engine.game_logic.agents import RLAgent
                print(f"  Loading {name}: {os.path.basename(path)}")
                rl = RLAgent(model_path=path)
                self._agents[name] = rl.agent  # The underlying DQNAgent

        else:
            raise ValueError(
                f"Unknown agent '{name}'. Choices: {AGENT_CHOICES}"
            )

    def get(self, name: str):
        """Get the cached agent for a tier/agent name."""
        resolved = resolve_agent_name(name)
        if resolved not in self._agents:
            raise KeyError(f"Agent '{resolved}' not loaded. Loaded: {list(self._agents.keys())}")
        return self._agents[resolved]
