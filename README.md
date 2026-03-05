# DDA UNO Engine — Pure Core Package

Multi-agent AI system for UNO where 3 AI bots play against 1 human player.
The bots dynamically adjust their play strength so the player's win rate
converges to a configurable target (e.g., 50%).

## Architecture

The engine is a **pure core package** — it contains only game logic, agents,
bots, and configuration. It has **no** training or simulation code.

```
ada-uno/
├── engine/        ← This package: pure core (game logic, agents, bots)
├── api/           ← FastAPI service (wraps engine for live play)
├── simulator/     ← Offline training & simulation (wraps engine)
└── app/           ← Phaser.js frontend (connects to api)
```

```
┌──────────────┐  ┌──────────────┐
│  simulator/  │  │     api/     │
│  (train/sim) │  │   (serve)    │
└──────┬───────┘  └──────┬───────┘
       │                 │
       ▼                 ▼
┌─────────────────────────────────────────────────┐
│              engine/ (this package)              │
│  game_logic/  config/  models/  data/           │
└─────────────────────────────────────────────────┘
```

### Package Structure

```
engine/
├── __init__.py               # Re-exports: UnoGame, TierModelPool, AdaptiveTierController, etc.
├── controller.py             # Re-exports WinRateController for convenience (legacy)
├── requirements.txt
├── README.md
├── config/
│   ├── __init__.py           # Re-exports all settings
│   ├── game.py               # NUM_PLAYERS, SEED, STATE_SHAPE, NUM_ACTIONS, etc.
│   └── controller.py         # TARGET_WIN_RATE, ADJUSTMENT_STEP, INITIAL_STRENGTH
├── game_logic/
│   ├── __init__.py           # Re-exports core classes
│   ├── game.py               # UnoGame: RLCard env wrapper (4-player UNO)
│   ├── controller.py         # WinRateController: proportional control (legacy)
│   ├── store.py              # PlayerStore: SQLite player history backend
│   ├── agents/
│   │   ├── __init__.py       # Re-exports BaseAgent, RLAgent, AdaptiveAgent
│   │   ├── base.py           # BaseAgent ABC (shared interface)
│   │   ├── rl_agent.py       # RLAgent: DQN wrapper, train/save/load
│   │   └── adaptive.py       # AdaptiveAgent: blends strong + weak (legacy)
│   ├── tiers/                # Tier-based adaptive difficulty system
│   │   ├── __init__.py       # Re-exports TierModelPool, AdaptiveTierController, constants
│   │   ├── tier_config.py    # TIER_ORDER, TIER_NAMES, VD policy, agent choices
│   │   ├── tier_pool.py      # TierModelPool: loads/caches agents by tier name
│   │   └── tier_controller.py # AdaptiveTierController: maps win rate to tier
│   └── bots/
│       ├── __init__.py       # get_bot("noob"|"casual"|"pro") factory
│       ├── noob.py           # NoobBot: random + color preference
│       ├── casual.py         # CasualBot: high-value first, save wilds
│       └── pro.py            # ProBot: tracks state, plays optimally
├── models/                   # Trained .pt weight files (gitignored)
└── data/                     # Runtime data: simulation results, SQLite DB (gitignored)
```

## Setup

```bash
cd engine
python -m venv venv
source venv/bin/activate        # Linux/Mac
# or: venv\Scripts\activate     # Windows
pip install -r requirements.txt
```

## Usage

The engine is imported by `api/` and `simulator/`:

```python
from engine import UnoGame, TierModelPool, AdaptiveTierController
from engine.game_logic.tiers.tier_config import TIER_ORDER, TIER_NAMES, VOLUNTARY_DRAW_POLICY
from engine.config.game import NUM_PLAYERS, STATE_SHAPE
from engine.game_logic.bots import get_bot
```

## How It Works

### Tier-Based Adaptive Difficulty

The engine provides a 6-tier system for controlling bot difficulty. Each tier is a distinct agent with a different reward structure and play style:

| Tier | Behavior | Difficulty for Seat 0 |
|------|----------|-----------------------|
| hyper_adversarial | Cooperative bots targeting seat 0 | Hardest |
| adversarial | Team play against seat 0 | Hard |
| selfish | Each bot plays to win individually | Neutral |
| random | Random legal actions | Easy |
| altruistic | Trained to help seat 0 win | Easier |
| hyper_altruistic | Strongly trained to help seat 0 win | Easiest |

### The Adaptive Tier Controller

The `AdaptiveTierController` selects which tier to use based on the player's win rate vs a configurable target:

```
error = current_win_rate - target_win_rate

error < -0.20  ->  hyper_altruistic   (way below target)
error < -0.10  ->  altruistic         (below target)
error < -0.05  ->  random             (slightly below)
error <  0.10  ->  selfish            (near target, baseline)
error <  0.20  ->  adversarial        (above target)
error >= 0.20  ->  hyper_adversarial  (way above target)
```

All 3 bot seats use the same tier per game. No games played yet defaults to "selfish".

### TierModelPool

Loads and caches DQN agents by tier name. Accepts a configurable `model_dir` so the API and simulator can point to different model directories. Falls back to `RandomAgent` when model files are missing.

### Legacy System

The `WinRateController` (proportional float) and `AdaptiveAgent` (coin-flip blending) are kept for backward compatibility but are no longer used by the API or simulator.

### Fixed Bots

Used as seat-0 stand-ins during simulation, or as fallbacks:

| Bot | Strategy |
|-----|----------|
| NoobBot | Random card selection with color preference |
| CasualBot | Plays high-value cards first, saves wilds |
| ProBot | Tracks game state, plays optimally |

## Config Reference

| Setting | File | Default | Description |
|---------|------|---------|-------------|
| `NUM_PLAYERS` | config/game.py | 4 | Players per game |
| `SEED` | config/game.py | None | Random seed |
| `STATE_SHAPE` | config/game.py | [12,4,15] | Enriched observation: 12 planes (hand, target, seat, counts, direction, discard, etc.) |
| `NUM_ACTIONS` | config/game.py | 61 | RLCard action space size |
| `TARGET_WIN_RATE` | config/controller.py | 0.25 | Target win rate |
| `ADJUSTMENT_STEP` | config/controller.py | 0.05 | Controller gain |
| `INITIAL_STRENGTH` | config/controller.py | 0.5 | Starting bot strength |

## Related Packages

- **[api/](../api/README.md)** — FastAPI service wrapping engine for live HTTP play
- **[simulator/](../simulator/)** — Offline training and simulation (produces model weights)
- **[app/](../app/)** — Phaser.js frontend connecting to the API
