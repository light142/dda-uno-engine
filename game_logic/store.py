"""Player history store with SQLite backend.

Uses a repository pattern so the backend can be swapped to Postgres
in Phase 2 by replacing the SQLiteStore class.
"""

import os
import json
import sqlite3
from typing import Optional

from engine.config.controller import INITIAL_STRENGTH, TARGET_WIN_RATE


# --- Data model ---

class PlayerStats:
    """Player statistics and bot configuration."""

    def __init__(
        self,
        player_id: str,
        games_played: int = 0,
        wins: int = 0,
        bot_strength: float = INITIAL_STRENGTH,
        target_win_rate: float = TARGET_WIN_RATE,
    ):
        self.player_id = player_id
        self.games_played = games_played
        self.wins = wins
        self.bot_strength = bot_strength
        self.target_win_rate = target_win_rate

    @property
    def win_rate(self) -> float:
        if self.games_played == 0:
            return 0.0
        return self.wins / self.games_played

    def to_dict(self) -> dict:
        return {
            'player_id': self.player_id,
            'games_played': self.games_played,
            'wins': self.wins,
            'win_rate': self.win_rate,
            'bot_strength': self.bot_strength,
            'target_win_rate': self.target_win_rate,
        }

    def __repr__(self) -> str:
        return (
            f"PlayerStats(id={self.player_id}, "
            f"games={self.games_played}, wins={self.wins}, "
            f"wr={self.win_rate:.2%}, strength={self.bot_strength:.3f})"
        )


# --- Repository interface ---

class BasePlayerStore:
    """Abstract interface for player storage backends."""

    def get_player(self, player_id: str) -> Optional[PlayerStats]:
        raise NotImplementedError

    def create_player(self, player_id: str) -> PlayerStats:
        raise NotImplementedError

    def update_player(self, stats: PlayerStats) -> None:
        raise NotImplementedError

    def get_or_create_player(self, player_id: str) -> PlayerStats:
        player = self.get_player(player_id)
        if player is None:
            player = self.create_player(player_id)
        return player

    def record_game(self, player_id: str, won: bool, new_strength: float) -> PlayerStats:
        """Record a game result and update bot strength.

        Args:
            player_id: The player's ID.
            won: Whether the player won this game.
            new_strength: Updated bot strength from the controller.

        Returns:
            Updated PlayerStats.
        """
        stats = self.get_or_create_player(player_id)
        stats.games_played += 1
        if won:
            stats.wins += 1
        stats.bot_strength = new_strength
        self.update_player(stats)
        return stats


# --- SQLite implementation ---

class PlayerStore(BasePlayerStore):
    """SQLite-backed player store.

    In Phase 2, replace this with a PostgresPlayerStore that implements
    the same BasePlayerStore interface.
    """

    def __init__(self, db_path: str = None):
        """Initialize the SQLite store.

        Args:
            db_path: Path to the SQLite database file.
                Defaults to data/players.db.
        """
        if db_path is None:
            data_dir = os.path.join(
                os.path.dirname(os.path.dirname(__file__)), "data"
            )
            os.makedirs(data_dir, exist_ok=True)
            db_path = os.path.join(data_dir, "players.db")

        self.db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS players (
                    player_id TEXT PRIMARY KEY,
                    games_played INTEGER DEFAULT 0,
                    wins INTEGER DEFAULT 0,
                    bot_strength REAL DEFAULT ?,
                    target_win_rate REAL DEFAULT ?
                )
            """, (INITIAL_STRENGTH, TARGET_WIN_RATE))
            conn.commit()

    def get_player(self, player_id: str) -> Optional[PlayerStats]:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT player_id, games_played, wins, bot_strength, target_win_rate "
                "FROM players WHERE player_id = ?",
                (player_id,)
            ).fetchone()

        if row is None:
            return None

        return PlayerStats(
            player_id=row[0],
            games_played=row[1],
            wins=row[2],
            bot_strength=row[3],
            target_win_rate=row[4],
        )

    def create_player(self, player_id: str) -> PlayerStats:
        stats = PlayerStats(player_id=player_id)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO players (player_id, games_played, wins, bot_strength, target_win_rate) "
                "VALUES (?, ?, ?, ?, ?)",
                (stats.player_id, stats.games_played, stats.wins,
                 stats.bot_strength, stats.target_win_rate)
            )
            conn.commit()
        return stats

    def update_player(self, stats: PlayerStats) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE players SET games_played = ?, wins = ?, "
                "bot_strength = ?, target_win_rate = ? "
                "WHERE player_id = ?",
                (stats.games_played, stats.wins, stats.bot_strength,
                 stats.target_win_rate, stats.player_id)
            )
            conn.commit()
