"""Adaptive tier controller: deterministic base + variation injection.

Two-layer design:
  Layer 1 (primary): Deterministic error-band mapping guarantees convergence
      to the target win rate. error = current_win_rate - target_win_rate is
      mapped to a tier via fixed bands.
  Layer 2 (secondary): With probability that scales with headroom (how close
      to target), override the base tier with a random "variation" tier so
      the player experiences all 6 tiers over time. The deterministic
      controller auto-corrects on subsequent games.
"""

import random

from engine.game_logic.tiers.tier_config import (
    VARIATION_WEIGHTS,
    MAX_VARIATION_RATE,
    VARIATION_THRESHOLD,
)


class AdaptiveTierController:
    """Selects which agent tier to use based on player win rate vs target.

    All 3 bot seats use the same tier for a given game (except
    hyper_adversarial which uses a special seat override handled by callers).

    Error bands (Layer 1, deterministic):
        error < -0.015 ->  hyper_altruistic   (far below target)
        error < -0.010 ->  altruistic         (moderately below)
        error < -0.005 ->  random             (slightly below)
        error <  0.005 ->  selfish            (near target)
        error <  0.010 ->  hard (adv/hadv)    (moderately above)
        error >= 0.010 ->  hard (adv/hadv)    (far above target)

    Variation injection (Layer 2):
        variation_rate = MAX_VARIATION_RATE * max(0, 1 - |error| / THRESHOLD)
        At target: up to 25% of games are variation.
        Far from target: 0% variation, pure deterministic correction.
    """

    DEFAULT_BANDS = [
        (-0.015, "hyper_altruistic"),
        (-0.010, "altruistic"),
        (-0.005, "random"),
        ( 0.005, "selfish"),
        ( 0.010, "hard"),       # resolved to adv or hadv randomly
    ]
    DEFAULT_FALLBACK = "hard"   # resolved to adv or hadv randomly
    DEFAULT_NO_HISTORY = "selfish"

    # Tiers that are interchangeable (near-identical win rates)
    INTERCHANGEABLE = {
        "hard": ("adversarial", "hyper_adversarial"),
    }

    def __init__(self, target_win_rate: float = 0.25):
        self.target_win_rate = target_win_rate
        self._var_tiers, self._var_cum_weights = self._build_variation_table()

    def select_tier(self, current_win_rate: float, games_played: int = 0) -> str:
        """Choose the agent tier for the next game.

        Args:
            current_win_rate: Player's accumulated win rate (0.0 to 1.0).
            games_played: Total games played. If 0, returns neutral baseline.

        Returns:
            Tier name string.
        """
        tier, _ = self.select_tier_detailed(current_win_rate, games_played)
        return tier

    def select_tier_detailed(
        self, current_win_rate: float, games_played: int = 0
    ) -> tuple[str, bool]:
        """Choose tier and report whether it was a variation pick.

        Returns:
            (tier_name, is_variation) tuple.
        """
        if games_played == 0:
            return self.DEFAULT_NO_HISTORY, False

        error = current_win_rate - self.target_win_rate

        # Layer 2: check if this game should be a variation game
        var_rate = self._variation_rate(error)
        if var_rate > 0 and random.random() < var_rate:
            return self._variation_tier(), True

        # Layer 1: deterministic base selection
        return self._base_tier(error), False

    def _base_tier(self, error: float) -> str:
        """Deterministic tier from error bands."""
        for upper_bound, tier in self.DEFAULT_BANDS:
            if error < upper_bound:
                return self._resolve(tier)
        return self._resolve(self.DEFAULT_FALLBACK)

    def _resolve(self, tier: str) -> str:
        """Resolve interchangeable tier groups to a concrete tier."""
        group = self.INTERCHANGEABLE.get(tier)
        if group:
            return random.choice(group)
        return tier

    def _variation_rate(self, error: float) -> float:
        """Compute variation probability from headroom.

        Linear ramp: max at error=0, zero at |error| >= VARIATION_THRESHOLD.
        """
        abs_error = abs(error)
        if abs_error >= VARIATION_THRESHOLD:
            return 0.0
        return MAX_VARIATION_RATE * (1.0 - abs_error / VARIATION_THRESHOLD)

    def _variation_tier(self) -> str:
        """Pick a random tier from the weighted variation distribution."""
        return random.choices(
            self._var_tiers, cum_weights=self._var_cum_weights, k=1
        )[0]

    @staticmethod
    def _build_variation_table() -> tuple[list[str], list[float]]:
        """Pre-compute cumulative weights for random.choices."""
        tiers = list(VARIATION_WEIGHTS.keys())
        weights = [VARIATION_WEIGHTS[t] for t in tiers]
        # Build cumulative weights
        cum = []
        total = 0.0
        for w in weights:
            total += w
            cum.append(total)
        return tiers, cum
