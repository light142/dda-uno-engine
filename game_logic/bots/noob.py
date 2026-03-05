"""NoobBot: simulates a beginner player.

Strategy: picks a random legal action with slight preference for
matching colors. Doesn't strategize about action cards or wilds.
Represents a player who knows the rules but doesn't think ahead.
"""

import random
import numpy as np

from engine.game_logic.agents.base import BaseAgent


class NoobBot(BaseAgent):
    """Beginner-level fixed bot for simulation."""

    def __init__(self):
        super().__init__()
        self.use_raw = True  # Use raw state for readable card info

    def step(self, state: dict) -> str:
        """Pick a mostly random legal action with slight color bias."""
        return self._choose(state)

    def eval_step(self, state: dict) -> tuple:
        """Same as step — no distinction for fixed bots."""
        action = self._choose(state)
        return action, {}

    def _choose(self, state: dict) -> str:
        raw_obs = state['raw_obs']
        legal_actions = state['raw_legal_actions']

        if not legal_actions:
            return 'draw'

        # If only one option, take it
        if len(legal_actions) == 1:
            return legal_actions[0]

        # Slight preference for cards matching the current color
        target = raw_obs.get('target', '')
        target_color = target.split('-')[0] if '-' in target else ''

        # Weight matching-color cards slightly higher
        weights = []
        for action in legal_actions:
            if action == 'draw':
                weights.append(0.3)  # Noobs sometimes draw unnecessarily
            elif target_color and action.startswith(target_color):
                weights.append(2.0)  # Slight preference for matching color
            else:
                weights.append(1.0)

        # Normalize and pick
        total = sum(weights)
        weights = [w / total for w in weights]
        return random.choices(legal_actions, weights=weights, k=1)[0]
