"""Adaptive agent that blends strong and weak RL agents via a strength knob."""

import random

from engine.game_logic.agents.base import BaseAgent
from engine.game_logic.agents.rl_agent import RLAgent
from engine.config.controller import INITIAL_STRENGTH


class AdaptiveAgent(BaseAgent):
    """Agent that blends a strong and weak RL agent based on a strength parameter.

    The strength knob (0.0 to 1.0) controls the mix:
        - strength = 1.0: always uses the strong agent (plays to win / hinder player)
        - strength = 0.0: always uses the weak agent (plays to help player win)
        - strength = 0.5: 50/50 blend

    The WinRateController adjusts strength after each game to converge
    the player's win rate toward the target.
    """

    def __init__(
        self,
        strong_model_path: str,
        weak_model_path: str,
        strength: float = INITIAL_STRENGTH,
        device: str = None,
    ):
        """Initialize the adaptive agent.

        Args:
            strong_model_path: Path to the strong agent's trained weights.
            weak_model_path: Path to the weak agent's trained weights.
            strength: Initial blending strength (0.0 = weak, 1.0 = strong).
            device: PyTorch device for model inference.
        """
        super().__init__()
        self.strong_agent = RLAgent(model_path=strong_model_path, device=device)
        self.weak_agent = RLAgent(model_path=weak_model_path, device=device)
        self.strength = strength

    @property
    def strength(self) -> float:
        return self._strength

    @strength.setter
    def strength(self, value: float) -> None:
        self._strength = max(0.0, min(1.0, value))

    def _pick_agent(self) -> RLAgent:
        """Randomly pick strong or weak agent based on current strength."""
        if random.random() < self._strength:
            return self.strong_agent
        return self.weak_agent

    def step(self, state: dict) -> int:
        """Choose action by blending strong/weak agents."""
        return self._pick_agent().step(state)

    def eval_step(self, state: dict) -> tuple:
        """Choose action by blending strong/weak agents (eval mode)."""
        return self._pick_agent().eval_step(state)
