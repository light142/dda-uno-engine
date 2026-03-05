"""Base agent interface shared by all agents and bots."""

from abc import ABC, abstractmethod


class BaseAgent(ABC):
    """Abstract base class for all UNO agents and bots.

    Both RL-trained agents (in game_logic/agents/) and fixed-strategy bots
    (in simulation/bots/) implement this interface, making them interchangeable
    at any seat.
    """

    def __init__(self):
        # RLCard compatibility: determines whether step/eval_step receive
        # raw state (hand strings, card names) or extracted state (numpy obs).
        # True = raw state dict, False = extracted numpy observation.
        self.use_raw = False

    @abstractmethod
    def step(self, state: dict) -> int:
        """Choose an action during training.

        Args:
            state: Game state dict from RLCard containing:
                - 'obs': numpy array of shape STATE_SHAPE
                - 'legal_actions': OrderedDict of {action_id: None}
                - 'raw_obs': dict with hand, target, legal_actions as strings
                - 'raw_legal_actions': list of action strings

        Returns:
            Action ID (int, 0-60) representing the chosen card or draw.
        """
        pass

    @abstractmethod
    def eval_step(self, state: dict) -> tuple:
        """Choose an action during evaluation (no exploration).

        Args:
            state: Same as step().

        Returns:
            Tuple of (action_id, info_dict) where info_dict contains
            additional information like probabilities or Q-values.
        """
        pass

    def feed(self, transition: list) -> None:
        """Feed a training transition to the agent (optional).

        Only RL agents need to implement this. Fixed bots can ignore it.

        Args:
            transition: [state, action, reward, next_state, done]
        """
        pass
