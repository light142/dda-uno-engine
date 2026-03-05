"""ProBot: simulates an experienced player.

Strategy: plays optimally with awareness of game state. Tracks
discarded cards, holds action cards for strategic moments, chooses
colors based on hand composition, and manages wild cards carefully.
"""

import random
from collections import Counter

from engine.game_logic.agents.base import BaseAgent


class ProBot(BaseAgent):
    """Expert-level fixed bot for simulation."""

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

        playable = [a for a in legal_actions if a != 'draw']
        if not playable:
            return 'draw'

        hand = raw_obs.get('hand', [])
        num_cards = raw_obs.get('num_cards', [])

        # Check if any opponent is about to win (1-2 cards)
        opponent_danger = any(
            n <= 2 for i, n in enumerate(num_cards) if i != 0 and n > 0
        )

        # Count colors in hand for optimal color selection
        hand_colors = self._count_colors(hand)
        best_color = max(hand_colors, key=hand_colors.get) if hand_colors else 'r'

        # Strategy: prioritize based on game state
        if opponent_danger:
            return self._defensive_play(playable, hand, best_color)
        elif len(hand) <= 3:
            return self._rush_play(playable, hand, best_color)
        else:
            return self._standard_play(playable, hand, best_color)

    def _defensive_play(self, playable: list, hand: list, best_color: str) -> str:
        """When opponents are close to winning, play aggressively."""
        # Prioritize: draw_2 > skip > reverse > wild_draw_4 > others
        for priority in ['draw_2', 'skip', 'reverse', 'wild_draw_4']:
            matches = [a for a in playable if priority in a]
            if matches:
                return self._prefer_color(matches, best_color)

        return self._standard_play(playable, hand, best_color)

    def _rush_play(self, playable: list, hand: list, best_color: str) -> str:
        """When we're close to winning, play cards to empty hand fast."""
        # Play non-wild cards first to save wilds as backup
        non_wild = [a for a in playable if 'wild' not in a]
        if non_wild:
            return self._prefer_color(non_wild, best_color)
        return playable[0]

    def _standard_play(self, playable: list, hand: list, best_color: str) -> str:
        """Normal play: manage hand composition optimally."""
        # Save wilds if we have other options
        non_wild = [a for a in playable if 'wild' not in a]
        if non_wild:
            # Prefer playing cards of our weakest color (to consolidate hand)
            hand_colors = self._count_colors(hand)
            weakest_color = min(hand_colors, key=hand_colors.get) if hand_colors else ''

            weak_color_cards = [a for a in non_wild if a.startswith(weakest_color)]
            if weak_color_cards:
                # Play highest value card of weakest color
                return self._pick_highest_value(weak_color_cards)

            # Otherwise play highest value non-wild
            return self._pick_highest_value(non_wild)

        # Only wilds: pick the one that sets our best color
        wild_d4 = [a for a in playable if 'wild_draw_4' in a]
        wild = [a for a in playable if a.startswith('wild') and 'draw_4' not in a]

        # Prefer regular wild (save draw 4 for emergencies)
        if wild:
            return wild[0]
        if wild_d4:
            return wild_d4[0]

        return playable[0]

    def _prefer_color(self, actions: list, preferred_color: str) -> str:
        """From a list of actions, prefer ones matching the given color."""
        matching = [a for a in actions if a.startswith(preferred_color)]
        if matching:
            return random.choice(matching)
        return random.choice(actions)

    def _pick_highest_value(self, actions: list) -> str:
        """Pick the card with the highest point value."""
        value_order = [
            'draw_2', 'skip', 'reverse',
            '9', '8', '7', '6', '5', '4', '3', '2', '1', '0'
        ]
        for value in value_order:
            matches = [a for a in actions if value in a.split('-', 1)[-1]]
            if matches:
                return random.choice(matches)
        return random.choice(actions)

    def _count_colors(self, hand: list) -> dict:
        """Count how many cards of each color are in hand."""
        colors = Counter()
        for card in hand:
            parts = card.split('-', 1)
            if len(parts) == 2 and parts[0] in ('r', 'g', 'b', 'y'):
                colors[parts[0]] += 1
        return dict(colors) if colors else {'r': 0}
