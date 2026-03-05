"""Win rate controller that adjusts bot strength to hit target win rates.

Uses proportional control: if the player is winning too much, increase
bot strength (make bots harder). If losing too much, decrease strength
(make bots easier).
"""

from engine.config.controller import (
    TARGET_WIN_RATE, ADJUSTMENT_STEP,
    STRENGTH_MIN, STRENGTH_MAX, INITIAL_STRENGTH,
)


class WinRateController:
    """Proportional controller for player win rate.

    After each game, call adjust() with the game result. The controller
    computes the error between actual and target win rate, then adjusts
    the bot strength accordingly.
    """

    def __init__(
        self,
        target_win_rate: float = TARGET_WIN_RATE,
        adjustment_step: float = ADJUSTMENT_STEP,
        strength_min: float = STRENGTH_MIN,
        strength_max: float = STRENGTH_MAX,
        initial_strength: float = INITIAL_STRENGTH,
    ):
        """Initialize the controller.

        Args:
            target_win_rate: Desired win rate for seat 0 (0.0 to 1.0).
            adjustment_step: Proportional gain — how much to adjust per game.
            strength_min: Minimum bot strength (0.0 = full support).
            strength_max: Maximum bot strength (1.0 = full attack).
            initial_strength: Starting bot strength when no history exists.
        """
        self.target_win_rate = target_win_rate
        self.adjustment_step = adjustment_step
        self.strength_min = strength_min
        self.strength_max = strength_max
        self.initial_strength = initial_strength

    def adjust(self, current_win_rate: float, current_strength: float) -> float:
        """Compute new bot strength based on current win rate.

        Args:
            current_win_rate: Player's actual win rate so far (0.0 to 1.0).
            current_strength: Current bot strength value.

        Returns:
            New bot strength value, clamped to [strength_min, strength_max].
        """
        # Error: positive means player is winning too much
        error = current_win_rate - self.target_win_rate

        # Proportional adjustment: increase strength if player winning too much
        new_strength = current_strength + self.adjustment_step * error

        # Clamp to valid range
        return max(self.strength_min, min(self.strength_max, new_strength))
