import json
import sqlite3
from typing import Any, Optional, Tuple

class KVStore:
    """
    Base de données clé-valeur simple.
    - Stocke les données dans un fichier SQLite (store.db par défaut)
    - Chaque entrée a :
        k = clé (ex : ID ou hash)
        v = valeur (stockée en JSON)
    """

    def __init__(self, path: str = "store.db") -> None:
        self.conn = sqlite3.connect(path)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS kv (
                k TEXT PRIMARY KEY,
                v TEXT NOT NULL
            )
        """)
        self.conn.commit()

    def put(self, key: str, value: Any) -> None:
        """Ajoute ou remplace la valeur associée à la clé."""
        payload = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        self.conn.execute(
            "INSERT INTO kv (k, v) VALUES (?, ?) "
            "ON CONFLICT(k) DO UPDATE SET v=excluded.v",
            (key, payload),
        )
        self.conn.commit()

    def get(self, key: str) -> Optional[Any]:
        """Récupère la valeur pour une clé, ou None si absente."""
        cur = self.conn.execute("SELECT v FROM kv WHERE k = ?", (key,))
        row = cur.fetchone()
        return json.loads(row[0]) if row else None

    def exists(self, key: str) -> bool:
        """Renvoie True si la clé existe déjà."""
        cur = self.conn.execute("SELECT 1 FROM kv WHERE k = ? LIMIT 1", (key,))
        return cur.fetchone() is not None

    def upsert_with_check(self, key: str, value: Any) -> Tuple[bool, Any]:
        """
        Si la clé existe :
            → renvoie (True, valeur_existante)
        Sinon :
            → stocke value et renvoie (False, value)
        """
        cur = self.conn.execute("SELECT v FROM kv WHERE k = ? LIMIT 1", (key,))
        row = cur.fetchone()
        if row:
            return True, json.loads(row[0])
        self.put(key, value)
        return False, value

    def delete(self, key: str) -> bool:
        """Supprime la clé. Renvoie True si quelque chose a été supprimé."""
        cur = self.conn.execute("DELETE FROM kv WHERE k = ?", (key,))
        self.conn.commit()
        return cur.rowcount > 0

    def keys(self, limit: int = 100, offset: int = 0) -> list[str]:
        """Renvoie une liste des clés présentes (pour debug)."""
        cur = self.conn.execute(
            "SELECT k FROM kv ORDER BY k LIMIT ? OFFSET ?", (limit, offset)
        )
        return [r[0] for r in cur.fetchall()]

    def close(self) -> None:
        """Ferme la connexion proprement."""
        self.conn.close()

