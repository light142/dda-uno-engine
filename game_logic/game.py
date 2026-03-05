"""UNO game engine wrapping RLCard's UNO environment.

Provides a clean interface for running UNO games with configurable agents
at each seat. Used by training, simulation, and the API layer.
"""

import numpy as np
import rlcard
from rlcard.games.uno.game import UnoGame as RLCardUnoGame
from rlcard.games.uno.round import UnoRound
from rlcard.games.uno.card import UnoCard
from rlcard.games.uno.utils import encode_hand, encode_target, WILD, WILD_DRAW_4
from rlcard.utils import reorganize

from engine.config.game import NUM_PLAYERS, PLAYER_SEAT, NUM_ACTIONS, STATE_SHAPE, SEED


# ---------------------------------------------------------------------------
# RLCard bug fix: auto-played wild_draw_4 from a draw has no penalty.
#
# In RLCard's _perform_draw_action, drawn wild cards (including wild_draw_4)
# are auto-played but only set the target — _preform_non_number_action is
# never called, so the next player does NOT receive 4 penalty cards.
#
# This patch makes wild_draw_4 call _preform_non_number_action just like
# it does when played from hand, so the penalty is correctly applied.
# ---------------------------------------------------------------------------

def _patched_perform_draw_action(self, players):
    if not self.dealer.deck:
        self.replace_deck()

    if not self.dealer.deck:
        # No cards anywhere — end the game (fewest cards wins)
        self.is_over = True
        self.winner = [min(range(self.num_players),
                          key=lambda i: len(players[i].hand))]
        return

    card = self.dealer.deck.pop()

    if card.type == 'wild':
        card.color = self.np_random.choice(UnoCard.info['color'])
        self.played_cards.append(card)
        if card.trait == 'wild_draw_4':
            # Must call _preform_non_number_action to deal 4 cards
            # and skip the penalized player (original code skips this).
            self._preform_non_number_action(players, card)
        else:
            self.target = card
            self.current_player = (
                self.current_player + self.direction
            ) % self.num_players

    elif card.color == self.target.color:
        if card.type == 'number':
            self.target = card
            self.played_cards.append(card)
            self.current_player = (
                self.current_player + self.direction
            ) % self.num_players
        else:
            self.played_cards.append(card)
            self._preform_non_number_action(players, card)

    else:
        players[self.current_player].hand.append(card)
        self.current_player = (
            self.current_player + self.direction
        ) % self.num_players


UnoRound._perform_draw_action = _patched_perform_draw_action


# ---------------------------------------------------------------------------
# RLCard bug fix: deck can run empty during +2/+4 penalty dealing.
#
# When _preform_non_number_action deals penalty cards (2 for draw_2,
# 4 for wild_draw_4), it calls dealer.deal_cards() which pops from the
# deck without checking if enough cards remain. In long games the deck
# empties and pop() raises IndexError.
#
# Fix: temporarily replace dealer.deal_cards with a safe version that
# reshuffles the discard pile back into the deck before each pop.
# ---------------------------------------------------------------------------

_original_non_number_action = UnoRound._preform_non_number_action


def _patched_non_number_action(self, players, card):
    round_ref = self

    def _safe_deal_cards(player, num):
        for _ in range(num):
            if not round_ref.dealer.deck:
                round_ref.replace_deck()
            if not round_ref.dealer.deck:
                break  # No cards left anywhere
            player.hand.append(round_ref.dealer.deck.pop())

    # Temporarily shadow the class method with our safe version
    self.dealer.deal_cards = _safe_deal_cards
    try:
        _original_non_number_action(self, players, card)
    finally:
        del self.dealer.deal_cards  # Remove instance override, restore class method

UnoRound._preform_non_number_action = _patched_non_number_action


# ---------------------------------------------------------------------------
# Rule change: wild_draw_4 is always playable (no color restriction).
#
# In official UNO (and RLCard default), wild_draw_4 can only be played when
# the player has NO cards matching the target color.  This patch removes that
# restriction so wild_draw_4 is always a legal option — like regular wilds.
# This gives bots (and the human) more strategic freedom.
# ---------------------------------------------------------------------------

_original_get_legal_actions = UnoRound.get_legal_actions


def _patched_get_legal_actions(self, players, player_id):
    wild_flag = 0
    wild_draw_4_flag = 0
    legal_actions = []
    hand = players[player_id].hand
    target = self.target

    if target.type == 'wild':
        for card in hand:
            if card.type == 'wild':
                if card.trait == 'wild_draw_4':
                    if wild_draw_4_flag == 0:
                        wild_draw_4_flag = 1
                elif wild_flag == 0:
                    wild_flag = 1
                    legal_actions.extend(WILD)
            elif card.color == target.color:
                legal_actions.append(card.str)
    else:
        for card in hand:
            if card.type == 'wild':
                if card.trait == 'wild_draw_4':
                    if wild_draw_4_flag == 0:
                        wild_draw_4_flag = 1
                elif wild_flag == 0:
                    wild_flag = 1
                    legal_actions.extend(WILD)
            elif card.color == target.color or card.trait == target.trait:
                legal_actions.append(card.str)

    # Always include wild_draw_4 when held (no color restriction)
    if wild_draw_4_flag:
        legal_actions.extend(WILD_DRAW_4)

    if not legal_actions:
        legal_actions = ['draw']

    return legal_actions


UnoRound.get_legal_actions = _patched_get_legal_actions


# ---------------------------------------------------------------------------
# Enriched state extraction: extends RLCard's default 4-plane observation
# with extra planes so the DQN can learn seat-aware and context-aware play.
#
# Default RLCard obs [4, 4, 15]:
#   Planes 0-2: Agent's hand encoding
#   Plane  3:   Top card (target)
#
# Enriched obs [12, 4, 15]:
#   Planes 0-2: Agent's hand encoding (unchanged)
#   Plane  3:   Top card (unchanged)
#   Plane  4:   Seat identity — one-hot row for this agent's seat
#   Plane  5:   Card counts — each player's hand size (normalized)
#   Plane  6:   Next player one-hot + direction indicator
#   Plane  7:   Discard pile — card counting (which cards have been played)
#   Plane  8:   Last card played per player — reveals color preferences
#   Plane  9:   Draw vulnerability — per-player draw counts by target color
#   Plane 10:   Deck size — how many cards remain in draw pile
#   Plane 11:   Target seat — which seat this agent should help win (all zeros if none)
# ---------------------------------------------------------------------------

def _enriched_extract_state(env, state):
    """Build enriched observation with seat, card count, and direction info."""
    obs = np.zeros(STATE_SHAPE, dtype=int)

    # Planes 0-2: hand, Plane 3: target (same as RLCard default)
    encode_hand(obs[:3], state['hand'])
    encode_target(obs[3], state['target'])

    player_id = state['current_player']
    game_round = env.game.round

    # Plane 4: Seat identity — row `player_id` is filled with 1s
    obs[4, player_id % 4, :] = 1

    # Plane 5: Card counts — row per player, fill columns with normalized count
    # Max hand size ~30 cards; normalize by dividing by 15 and clamping to 1
    num_cards = state.get('num_cards', [7] * NUM_PLAYERS)
    for i, count in enumerate(num_cards):
        normalized = min(count / 15.0, 1.0)
        obs[5, i % 4, :] = int(round(normalized * 14))  # encode as 0-14 across cols

    # Plane 6: Next player + direction
    # Row 0-3: one-hot for next player seat
    direction = game_round.direction  # +1 clockwise, -1 counter-clockwise
    next_player = (player_id + direction) % NUM_PLAYERS
    obs[6, next_player % 4, :] = 1
    # Encode direction: if counter-clockwise, set row for human seat (0) as marker
    if direction == -1:
        obs[6, :, 14] = 1  # last column = direction flag

    # Plane 7: Discard pile — count of each card played, normalized by total copies
    # Rows = colors (r=0, g=1, b=2, y=3), Cols = values (0-9, skip, reverse, +2, wild, +4)
    # Value = (cards_played / total_copies) so 1.0 means all copies are gone.
    # UNO deck: 0 = 1/color, 1-9 = 2/color, skip/rev/+2 = 2/color, wild/+4 = 4 total
    # Lets agent reason: "all red skips played = nobody can skip with red"
    COLOR_MAP = {'r': 0, 'g': 1, 'b': 2, 'y': 3}
    TRAIT_MAP = {
        '0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6,
        '7': 7, '8': 8, '9': 9, 'skip': 10, 'reverse': 11,
        'draw_2': 12, 'wild': 13, 'wild_draw_4': 14,
    }
    # Max copies per card type per color (wilds have 4 total, spread across 4 color rows)
    MAX_COPIES = {
        0: 1,   # 0 cards: 1 per color
        1: 2, 2: 2, 3: 2, 4: 2, 5: 2, 6: 2, 7: 2, 8: 2, 9: 2,  # 1-9: 2 per color
        10: 2, 11: 2, 12: 2,  # skip, reverse, +2: 2 per color
        13: 1, 14: 1,  # wild, +4: 4 total = 1 per color row
    }
    discard_counts = {}
    played_cards = state.get('played_cards', [])
    for card_str in played_cards:
        parts = card_str.split('-')
        if len(parts) == 2:
            color, trait = parts
            c = COLOR_MAP.get(color)
            t = TRAIT_MAP.get(trait)
            if c is not None and t is not None:
                key = (c, t)
                discard_counts[key] = discard_counts.get(key, 0) + 1
    for (c, t), count in discard_counts.items():
        max_c = MAX_COPIES.get(t, 2)
        obs[7, c, t] = min(count, max_c)  # raw count capped at max copies

    # Plane 8: Last card played by each player
    # Row = player seat, Col = card trait (0-14), Value = color index + 1 (1=r,2=g,3=b,4=y)
    # Key insight: wild color choices reveal hand composition
    # e.g. human chose blue on a wild → likely holds blue cards
    last_action_per_player = {}
    for rec in env.action_recorder:
        pid, action_str = rec[0], rec[1]
        if isinstance(action_str, str) and action_str != 'draw':
            last_action_per_player[pid] = action_str
    for pid, card_str in last_action_per_player.items():
        parts = card_str.split('-')
        if len(parts) == 2:
            color, trait = parts
            c = COLOR_MAP.get(color)
            t = TRAIT_MAP.get(trait)
            if c is not None and t is not None:
                obs[8, pid % 4, t] = c + 1  # 1=r, 2=g, 3=b, 4=y

    # Plane 9: Draw vulnerability — per-player draw counts by target color
    # Row = player seat, Cols 0-3 = draws when target was r/g/b/y
    # Even 1 draw on blue means: "this player has no blue, no matching number, no wilds"
    # Resets when the player plays a card (their hand changed)
    # Reconstructed by walking the action history
    COLOR_IDX = {'r': 0, 'g': 1, 'b': 2, 'y': 3}
    draw_per_color = {i: [0, 0, 0, 0] for i in range(NUM_PLAYERS)}
    # Track the current target color through the action history
    # Start with the initial target from the game's first played card
    tracking_color = None
    initial_played = state.get('played_cards', [])
    if initial_played:
        first_parts = initial_played[0].split('-')
        if len(first_parts) == 2:
            tracking_color = first_parts[0]
    for rec in env.action_recorder:
        pid, action_str = rec[0], rec[1]
        if action_str == 'draw':
            if tracking_color and tracking_color in COLOR_IDX:
                draw_per_color[pid][COLOR_IDX[tracking_color]] += 1
        else:
            # Player played a card — reset their draws, update tracking color
            draw_per_color[pid] = [0, 0, 0, 0]
            parts = action_str.split('-')
            if len(parts) == 2:
                tracking_color = parts[0]
    for pid in range(NUM_PLAYERS):
        for ci in range(4):
            obs[9, pid % 4, ci] = min(draw_per_color[pid][ci], 14)

    # Plane 10: Deck size — remaining cards in draw pile
    # UNO deck = 108 cards. Normalize: value = remaining / 8 (capped at 14)
    # Full deck ~79 after dealing → ~10. Empty → 0. Tells agent early/mid/late game.
    deck_remaining = len(game_round.dealer.deck)
    deck_val = min(int(deck_remaining / 8), 14)
    obs[10, :, :] = deck_val

    # Plane 11: Target seat — which seat this agent should help win
    # One-hot row encoding (same pattern as plane 4 seat identity)
    # All zeros when no target is set (adversarial/selfish modes)
    # Supports per-player dict: {player_id: target_seat} or global int
    _ts = getattr(env, '_target_seat', None)
    if _ts is not None:
        target_seat = _ts[player_id] if isinstance(_ts, dict) else _ts
        if target_seat is not None:
            obs[11, target_seat % 4, :] = 1

    # Build full state dict (raw_obs unchanged — API uses this)
    # Note: voluntary draw is handled at env._get_legal_actions level
    legal_action_id = env._get_legal_actions()

    extracted_state = {'obs': obs, 'legal_actions': legal_action_id}
    extracted_state['raw_obs'] = state
    raw_legal = [a for a in state['legal_actions']]
    if 60 in legal_action_id and 'draw' not in raw_legal:
        raw_legal.append('draw')
    extracted_state['raw_legal_actions'] = raw_legal
    extracted_state['action_record'] = env.action_recorder
    return extracted_state


class UnoGame:
    """Wrapper around RLCard's UNO environment.

    Manages game creation, agent assignment, and game execution.
    Supports both full-game mode (training/simulation) and step-by-step
    mode (API layer where a human plays at seat 0).
    """

    def __init__(self, seed: int = SEED):
        """Create a new UNO game environment.

        Args:
            seed: Random seed for reproducibility. None for random.
        """
        config = {}
        if seed is not None:
            config['seed'] = seed

        self.env = rlcard.make('uno', config=config)

        # RLCard's UNO env ignores game_num_players config (only supported
        # for blackjack/holdem). Manually replace the game object to get
        # the correct player count.
        if NUM_PLAYERS != 2:
            self.env.game = RLCardUnoGame(num_players=NUM_PLAYERS)
            self.env.num_players = NUM_PLAYERS

        # Override state extraction with enriched version and update shape
        self.env.state_shape = [STATE_SHAPE for _ in range(NUM_PLAYERS)]
        self.env.action_shape = [None for _ in range(NUM_PLAYERS)]
        self.env._extract_state = lambda state: _enriched_extract_state(
            self.env, state
        )
        self.env._target_seat = None
        self.env._allow_voluntary_draw = True
        self.env._max_voluntary_draws = None  # None = unlimited (deployment)
        self.env._voluntary_draw_counts = {}  # per-player count, reset each game
        self.env._voluntary_draw_offered_to = None  # player_id if we just offered

        # Patch _get_legal_actions to support voluntary draw at the env level.
        # This ensures both state extraction AND action decoding see draw as legal.
        # Respects per-player cap when _max_voluntary_draws is set.
        _original_get_legal = self.env._get_legal_actions
        _env_ref = self.env

        def _get_legal_with_voluntary_draw():
            legal = _original_get_legal()
            _env_ref._voluntary_draw_offered_to = None
            if getattr(_env_ref, '_allow_voluntary_draw', False) and 60 not in legal:
                # Check per-player cap (supports int or dict)
                max_vd = _env_ref._max_voluntary_draws
                if max_vd is not None:
                    pid = _env_ref.game.round.current_player
                    # Dict = per-player caps, int = global cap
                    cap = max_vd[pid] if isinstance(max_vd, dict) else max_vd
                    if _env_ref._voluntary_draw_counts.get(pid, 0) >= cap:
                        return legal  # Capped — don't offer voluntary draw
                legal[60] = None
                _env_ref._voluntary_draw_offered_to = _env_ref.game.round.current_player
            return legal

        self.env._get_legal_actions = _get_legal_with_voluntary_draw

        # Wrap env.step to track when a voluntary draw is actually taken.
        _original_env_step = self.env.step

        def _step_with_draw_tracking(action, raw_action=False):
            is_draw = (action == 'draw') if raw_action else (action == 60)
            offered_to = _env_ref._voluntary_draw_offered_to
            _env_ref._voluntary_draw_offered_to = None
            # Step first — _decode_action internally calls _get_legal_actions,
            # so the count must not be incremented yet or action 60 disappears.
            result = _original_env_step(action, raw_action)
            if is_draw and offered_to is not None:
                _env_ref._voluntary_draw_counts[offered_to] = \
                    _env_ref._voluntary_draw_counts.get(offered_to, 0) + 1
            return result

        self.env.step = _step_with_draw_tracking

        # Wrap env.reset to clear voluntary draw counts each game.
        _original_env_reset = self.env.reset

        def _reset_with_draw_tracking():
            _env_ref._voluntary_draw_counts = {}
            _env_ref._voluntary_draw_offered_to = None
            return _original_env_reset()

        self.env.reset = _reset_with_draw_tracking
        self._agents = None

    def set_target_seat(self, seat=None):
        """Set which seat the support agents should help win.

        Args:
            seat: One of:
                - int (0-3): Global target — all agents see the same plane 11.
                - dict {player_id: target_seat}: Per-player targets — each
                  agent sees its own plane 11. Use None as value for agents
                  that don't need a target (e.g. selfish).
                - None: No target (adversarial/selfish — plane 11 all zeros).
        """
        self.env._target_seat = seat

    def set_allow_voluntary_draw(self, allow: bool = True):
        """Control whether agents can draw even with playable cards.

        Default is True (realistic UNO rules). Set False for altruistic
        and cooperative training to force them to help by playing smart
        cards rather than passing.
        """
        self.env._allow_voluntary_draw = allow

    def set_max_voluntary_draws(self, max_draws=None):
        """Limit voluntary draws per player per game.

        Args:
            max_draws: Cap on voluntary draws per player per game.
                int = same cap for all players.
                dict = per-player caps {seat_id: max_draws}.
                None = unlimited (default, for deployment/API).
        """
        self.env._max_voluntary_draws = max_draws

    def set_agents(self, agents: list) -> None:
        """Assign agents to seats.

        Args:
            agents: List of agents, one per seat. Length must equal NUM_PLAYERS.
                Each agent must implement step() and eval_step() methods
                (BaseAgent interface or RLCard-compatible agent).
        """
        if len(agents) != NUM_PLAYERS:
            raise ValueError(f"Expected {NUM_PLAYERS} agents, got {len(agents)}")
        self._agents = agents
        self.env.set_agents(agents)

    def run_game(self, is_training: bool = False) -> dict:
        """Run a complete game from start to finish.

        All seats are controlled by their assigned agents (no human input).
        Used by training and simulation.

        Args:
            is_training: If True, agents use step() (with exploration).
                If False, agents use eval_step() (greedy).

        Returns:
            dict with:
                - 'winner': seat index of the winner (int)
                - 'payoffs': list of payoffs per seat
                - 'trajectories': raw trajectory data (for training)
        """
        if self._agents is None:
            raise RuntimeError("Agents not set. Call set_agents() first.")

        trajectories, payoffs = self.env.run(is_training=is_training)

        # Determine winner: seat with highest payoff
        winner = max(range(NUM_PLAYERS), key=lambda i: payoffs[i])

        return {
            'winner': winner,
            'payoffs': list(payoffs),
            'trajectories': trajectories,
        }

    def get_training_data(self, trajectories: list, payoffs: list) -> list:
        """Reorganize raw trajectories into per-agent training transitions.

        Args:
            trajectories: Raw trajectory data from run_game().
            payoffs: Payoff list from run_game().

        Returns:
            List of per-agent transition lists. Each transition is
            [state, action, reward, next_state, done].
        """
        return reorganize(trajectories, payoffs)

    def get_training_data_custom_reward(
        self, trajectories: list, payoffs: list, seat: int, reward_fn
    ) -> list:
        """Reorganize trajectories with a custom reward function.

        Used for training weak agents where the reward depends on
        whether seat 0 won, not whether the agent's own seat won.

        Args:
            trajectories: Raw trajectory data from run_game().
            payoffs: Original payoff list from run_game().
            seat: Which seat's transitions to extract.
            reward_fn: Function(payoffs, seat) -> float that computes
                the custom reward for this seat.

        Returns:
            List of transitions for the given seat with modified rewards.
        """
        custom_payoffs = list(payoffs)
        custom_payoffs[seat] = reward_fn(payoffs, seat)
        reorganized = reorganize(trajectories, custom_payoffs)
        return reorganized[seat]

    # --- Step-by-step mode for API layer (Phase 2) ---

    def start_game(self) -> dict:
        """Start a new game and return the initial state for seat 0.

        Returns:
            dict with:
                - 'state': game state for seat 0
                - 'current_player': seat index of who plays first
        """
        state, player_id = self.env.reset()
        return {
            'state': state,
            'current_player': player_id,
        }

    def player_step(self, action: int) -> dict:
        """Apply the human player's action and run all bot turns.

        Applies the player's chosen action, then automatically runs
        all bot turns until it's the player's turn again (or game ends).

        Args:
            action: Action ID chosen by the player (0-60).

        Returns:
            dict with:
                - 'bot_moves': list of {seat, action} for each bot turn
                - 'state': new game state for seat 0
                - 'current_player': seat index of next player
                - 'game_over': bool
                - 'winner': seat index if game_over, else None
                - 'payoffs': payoff list if game_over, else None
        """
        bot_moves = []

        # Apply player's action
        state, player_id = self.env.step(action)

        # Check if game ended after player's move
        if self.env.is_over():
            payoffs = self.env.get_payoffs()
            winner = max(range(NUM_PLAYERS), key=lambda i: payoffs[i])
            return {
                'bot_moves': bot_moves,
                'state': state,
                'current_player': player_id,
                'game_over': True,
                'winner': winner,
                'payoffs': list(payoffs),
            }

        # Run bot turns until it's seat 0's turn again or game ends
        while player_id != PLAYER_SEAT and not self.env.is_over():
            bot_agent = self._agents[player_id]
            bot_action, _ = bot_agent.eval_step(state)
            bot_moves.append({'seat': player_id, 'action': bot_action})
            state, player_id = self.env.step(bot_action)

        game_over = self.env.is_over()
        payoffs = list(self.env.get_payoffs()) if game_over else None
        winner = max(range(NUM_PLAYERS), key=lambda i: payoffs[i]) if game_over else None

        return {
            'bot_moves': bot_moves,
            'state': state,
            'current_player': player_id,
            'game_over': game_over,
            'winner': winner,
            'payoffs': payoffs,
        }
