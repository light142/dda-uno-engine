"""CasualBot: simulates an average player.

Strategy: plays highest-value cards first to minimize points held.
Saves wild cards for when stuck. Prefers action cards (skip, reverse,
draw 2) over number cards. Represents a player who understands basic
strategy but doesn't track opponents' cards.
"""

import random

from engine.game_logic.agents.base import BaseAgent


# Card value scores (higher = play first to reduce hand value)
CARD_PRIORITY = {
    'wild_draw_4': 50,  # High value, but save if possible
    'wild': 40,
    'draw_2': 30,
    'skip': 25,
    'reverse': 25,
    '9': 9, '8': 8, '7': 7, '6': 6, '5': 5,
    '4': 4, '3': 3, '2': 2, '1': 1, '0': 0,
}


class CasualBot(BaseAgent):
    """Average-level fixed bot for simulation."""

    def __init__(self):
        super().__init__()
        self.use_raw = True

    def step(self, state: dict) -> str:
        return self._choose(state)

    def eval_step(self, state: dict) -> tuple:
        action = self._choose(state)
        return action, {}

    def _choose(self, state: dict) -> str:
        raw_obs = state['raw_obs']
        legal_actions = state['raw_legal_actions']

        if not legal_actions:
            return 'draw'

        if len(legal_actions) == 1:
            return legal_actions[0]

        # Separate cards from draw action
        playable = [a for a in legal_actions if a != 'draw']
        if not playable:
            return 'draw'

        # Count cards in hand to decide whether to save wilds
        hand = raw_obs.get('hand', [])
        hand_size = len(hand)

        # If hand is small (2 or fewer cards), play anything including wilds
        if hand_size <= 2:
            return self._pick_highest_value(playable)

        # Try to play non-wild cards first
        non_wild = [a for a in playable if 'wild' not in a]
        if non_wild:
            return self._pick_highest_value(non_wild)

        # Only wilds left — play them
        return self._pick_highest_value(playable)

    def _pick_highest_value(self, actions: list) -> str:
        """Pick the action with the highest card value score."""

        def card_score(action: str) -> int:
            # Extract trait from "color-trait" format
            parts = action.split('-', 1)
            trait = parts[1] if len(parts) > 1 else parts[0]
            return CARD_PRIORITY.get(trait, 0)

        # Sort by score descending, with slight randomness for ties
        actions_scored = [(a, card_score(a)) for a in actions]
        max_score = max(s for _, s in actions_scored)
        top_actions = [a for a, s in actions_scored if s == max_score]

        return random.choice(top_actions)
