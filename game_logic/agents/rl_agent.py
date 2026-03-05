"""DQN-based RL agent wrapping RLCard's DQNAgent with custom reward support."""

import os
import numpy as np
import torch
from rlcard.agents import DQNAgent

from engine.game_logic.agents.base import BaseAgent
from engine.config.game import NUM_ACTIONS, STATE_SHAPE


class RLAgent(BaseAgent):
    """RL agent that wraps RLCard's DQNAgent.

    Used for both strong agents (trained to win) and weak agents
    (trained to help seat 0 win). The difference is in the reward
    function used during training, not in the agent architecture.

    When loading a pre-trained model (model_path), no hyperparameters are needed.
    When creating a fresh agent for training, pass hyperparameters as kwargs.
    """

    def __init__(self, model_path: str = None, device: str = None, **kwargs):
        """Initialize the RL agent.

        Args:
            model_path: Path to load pre-trained weights from. If None,
                creates a fresh agent for training.
            device: PyTorch device ('cpu', 'cuda'). Auto-detected if None.
            **kwargs: Training hyperparameters (only used when creating fresh agent):
                learning_rate, batch_size, replay_memory_size, replay_memory_init_size,
                update_target_every, discount_factor, epsilon_start, epsilon_end,
                epsilon_decay_steps, train_every, save_path, save_every
        """
        super().__init__()

        if model_path and os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
            self._fix_checkpoint_epsilon(checkpoint)
            self._agent = DQNAgent.from_checkpoint(checkpoint)
        else:
            self._agent = DQNAgent(
                num_actions=NUM_ACTIONS,
                state_shape=STATE_SHAPE,
                mlp_layers=kwargs.get('mlp_layers', [256, 256]),
                replay_memory_size=kwargs.get('replay_memory_size', 100_000),
                replay_memory_init_size=kwargs.get('replay_memory_init_size', 1_000),
                update_target_estimator_every=kwargs.get('update_target_every', 500),
                discount_factor=kwargs.get('discount_factor', 0.99),
                epsilon_start=kwargs.get('epsilon_start', 1.0),
                epsilon_end=kwargs.get('epsilon_end', 0.1),
                epsilon_decay_steps=kwargs.get('epsilon_decay_steps', 1_000_000),
                batch_size=kwargs.get('batch_size', 32),
                train_every=kwargs.get('train_every', 8),
                learning_rate=kwargs.get('learning_rate', 0.0001),
                device=device,
                save_path=kwargs.get('save_path'),
                save_every=kwargs.get('save_every'),
            )

        self.use_raw = self._agent.use_raw

    @staticmethod
    def _fix_checkpoint_epsilon(checkpoint):
        """Fix RLCard's epsilon save bug.

        RLCard's checkpoint_attributes() uses epsilons.min()/max() which
        swaps start and end for a decreasing schedule. This corrects the
        checkpoint in-place so from_checkpoint rebuilds the right schedule.
        """
        eps_start = checkpoint.get('epsilon_start')
        eps_end = checkpoint.get('epsilon_end')
        if eps_start is not None and eps_end is not None and eps_start < eps_end:
            checkpoint['epsilon_start'] = eps_end
            checkpoint['epsilon_end'] = eps_start

    def step(self, state: dict) -> int:
        """Choose action with epsilon-greedy exploration (training mode)."""
        return self._agent.step(state)

    def eval_step(self, state: dict) -> tuple:
        """Choose action greedily (evaluation mode)."""
        return self._agent.eval_step(state)

    def feed(self, transition: list) -> None:
        """Feed a training transition to update the Q-network."""
        self._agent.feed(transition)

    def save(self, path: str, filename: str = "agent.pt") -> None:
        """Save model weights to disk.

        Args:
            path: Directory to save the checkpoint.
            filename: Checkpoint filename.
        """
        os.makedirs(path, exist_ok=True)
        self._agent.save_checkpoint(path, filename)

    def load(self, filepath: str, device: str = None) -> None:
        """Load model weights from disk.

        Args:
            filepath: Full path to the .pt checkpoint file.
            device: PyTorch device to load onto.
        """
        checkpoint = torch.load(filepath, map_location=device)
        self._fix_checkpoint_epsilon(checkpoint)
        self._agent = DQNAgent.from_checkpoint(checkpoint)
        self.use_raw = self._agent.use_raw

    @property
    def agent(self) -> DQNAgent:
        """Access the underlying RLCard DQNAgent (for advanced use)."""
        return self._agent
