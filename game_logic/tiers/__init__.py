"""Tier system: config, model pool, and adaptive controller."""

from engine.game_logic.tiers.tier_config import (
    TIER_ORDER,
    TIER_NAMES,
    AGENT_CHOICES,
    AGENT_ALIASES,
    TARGET_SEAT_TIERS,
    FIXED_TARGET,
    DQN_TIERS,
    MIXABLE_TIERS,
    VOLUNTARY_DRAW_POLICY,
    VARIATION_WEIGHTS,
    MAX_VARIATION_RATE,
    VARIATION_THRESHOLD,
    TIER_SEAT_OVERRIDE,
    resolve_agent_name,
)
from engine.game_logic.tiers.tier_pool import TierModelPool, resolve_model_path
from engine.game_logic.tiers.tier_controller import AdaptiveTierController
