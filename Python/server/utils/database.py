import json
import sqlite3
from typing import Any, Optional, Tuple, List, Dict, Iterable

class KVStore:
    """
    KV store SQLite minimaliste pour données déjà hachées.
    - k = hash (str)
    - v = résultat (JSON texte compact)

    Fournit :
      - get / put / exists / delete / keys / close
      - split_with_data(hash_list) -> (known[list[data]], unknown[list[hash]])
      - check_many(hash_iter) -> {"hits": {hash: data}, "misses": [hash,...]}
      - put_many({hash: data}) -> int
      - get_or_insert(hash, value_if_new) -> (hit: bool, value: Any)
      - get_or_insert_many({hash: value_if_new}) -> {hash: {"status","value"}}
      - split_hashes(hash_iter, unique=False) -> (known_hashes, unknown_hashes)
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

    # --------- Unitaires de base ---------
    def get(self, hash_key: str) -> Optional[Any]:
        cur = self.conn.execute("SELECT v FROM kv WHERE k = ? LIMIT 1", (hash_key,))
        row = cur.fetchone()
        return json.loads(row[0]) if row else None

    def exists(self, hash_key: str) -> bool:
        cur = self.conn.execute("SELECT 1 FROM kv WHERE k = ? LIMIT 1", (hash_key,))
        return cur.fetchone() is not None

    def put(self, hash_key: str, value: Any) -> None:
        payload = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        self.conn.execute(
            "INSERT INTO kv (k, v) VALUES (?, ?) "
            "ON CONFLICT(k) DO UPDATE SET v=excluded.v",
            (hash_key, payload),
        )
        self.conn.commit()

    def delete(self, hash_key: str) -> bool:
        cur = self.conn.execute("DELETE FROM kv WHERE k = ?", (hash_key,))
        self.conn.commit()
        return cur.rowcount > 0

    def keys(self, limit: int = 100, offset: int = 0) -> list[str]:
        cur = self.conn.execute(
            "SELECT k FROM kv ORDER BY k LIMIT ? OFFSET ?", (limit, offset)
        )
        return [r[0] for r in cur.fetchall()]

    def close(self) -> None:
        self.conn.close()

    # --------- Aide : “insérer si inconnu” ---------
    def get_or_insert(self, hash_key: str, value_if_new: Any) -> Tuple[bool, Any]:
        cur = self.conn.execute("SELECT v FROM kv WHERE k = ? LIMIT 1", (hash_key,))
        row = cur.fetchone()
        if row:
            return True, json.loads(row[0])
        self.put(hash_key, value_if_new)
        return False, value_if_new

    # --------- Par lot : checks / inserts ---------
    def check_many(self, hash_keys: Iterable[str]) -> Dict[str, Any]:
        keys = list(dict.fromkeys(hash_keys))
        if not keys:
            return {"hits": {}, "misses": []}

        qmarks = ",".join("?" * len(keys))
        cur = self.conn.execute(f"SELECT k, v FROM kv WHERE k IN ({qmarks})", keys)
        hits = {k: json.loads(v) for (k, v) in cur.fetchall()}
        misses = [k for k in keys if k not in hits]
        return {"hits": hits, "misses": misses}

    def put_many(self, mapping: Dict[str, Any]) -> int:
        if not mapping:
            return 0
        rows = []
        for k, val in mapping.items():
            payload = json.dumps(val, ensure_ascii=False, separators=(",", ":"))
            rows.append((k, payload))
        with self.conn:
            self.conn.executemany(
                "INSERT INTO kv (k, v) VALUES (?, ?) "
                "ON CONFLICT(k) DO UPDATE SET v=excluded.v",
                rows,
            )
        return len(rows)

    def get_or_insert_many(self, candidates: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        result: Dict[str, Dict[str, Any]] = {}
        check = self.check_many(candidates.keys())
        hits, misses = check["hits"], check["misses"]

        for k, v in hits.items():
            result[k] = {"status": "hit", "value": v}

        to_insert = {k: candidates[k] for k in misses}
        self.put_many(to_insert)
        for k in misses:
            result[k] = {"status": "inserted", "value": candidates[k]}

        return result

    # --------- Ce que tu veux exactement : split avec data ---------
    def split_with_data(self, hash_keys: List[str]) -> tuple[List[Any], List[str]]:
        """
        Sépare la liste des hash en deux :
          - known : liste des valeurs déjà stockées (data)
          - unknown : liste des hash non présents
        Préserve l'ordre d'apparition et garde les doublons de l'entrée.
        """
        if not hash_keys:
            return [], []

        distinct = list(dict.fromkeys(hash_keys))
        qmarks = ",".join("?" * len(distinct))
        cur = self.conn.execute(f"SELECT k, v FROM kv WHERE k IN ({qmarks})", distinct)
        existing = {k: json.loads(v) for k, v in cur.fetchall()}

        known: List[Any] = []
        unknown: List[str] = []
        for h in hash_keys:
            if h in existing:
                known.append(existing[h])
            else:
                unknown.append(h)
        return known, unknown

    # --------- Option : split simple (sans data) ---------
    def split_hashes(
        self,
        hash_keys: Iterable[str],
        unique: bool = False,
    ) -> tuple[List[str], List[str]]:
        if unique:
            seen = set()
            keys = []
            for k in hash_keys:
                if k not in seen:
                    seen.add(k)
                    keys.append(k)
        else:
            keys = list(hash_keys)

        if not keys:
            return [], []

        qmarks = ",".join("?" * len(keys))
        cur = self.conn.execute(f"SELECT k FROM kv WHERE k IN ({qmarks})", keys)
        existing = {row[0] for row in cur.fetchall()}

        known = [k for k in keys if k in existing]
        unknown = [k for k in keys if k not in existing]
        return known, unknown
