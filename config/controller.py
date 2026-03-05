# Win rate controller settings

# Target win rate for the player at seat 0
TARGET_WIN_RATE = 0.25

# Proportional adjustment step: how much to adjust bot strength per game
# Higher = faster convergence but more oscillation
ADJUSTMENT_STEP = 0.05

# Bot strength bounds (0.0 = full support/weak, 1.0 = full attack/strong)
STRENGTH_MIN = 0.0
STRENGTH_MAX = 1.0

# Initial bot strength when no player history exists
INITIAL_STRENGTH = 0.5
