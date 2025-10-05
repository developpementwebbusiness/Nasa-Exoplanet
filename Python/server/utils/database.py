import json
import sqlite3
from typing import Any, List, Tuple, Dict

class KVStore:
    """
    KV store SQLite minimaliste pour données déjà hachées.
    - k = hash (str)
    - v = résultat (JSON texte compact)

    Fournit uniquement :
      - split_with_data(hash_list) -> (known_values[list[Any]], unknown_hashes[list[str]])
      - insert_new_many({hash: value}) -> int  (insère SANS remplacer)
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

    def close(self) -> None:
        self.conn.close()

    # (1) Entrée: liste de hashs -> Sortie: (valeurs connues, hashs inconnus)
    def split_with_data(self, hash_list: List[str]) -> Tuple[List[Any], List[str]]:
        """
        Retourne:
          - known_values: liste des valeurs (JSON -> Python) déjà en DB,
                          alignées aux positions où le hash était connu.
          - unknown_hashes: liste des hashs absents de la DB (ordre d'entrée conservé).
        """
        if not hash_list:
            return [], []

        # On interroge une seule fois sur les clés distinctes (pour perf),
        # tout en préservant ensuite l'ordre et les doublons d'entrée.
        distinct = list(dict.fromkeys(hash_list))
        qmarks = ",".join("?" * len(distinct))

        cur = self.conn.execute(f"SELECT k, v FROM kv WHERE k IN ({qmarks})", distinct)
        existing = {k: json.loads(v) for k, v in cur.fetchall()}

        known_values: List[Any] = []
        unknown_hashes: List[str] = []
        for h in hash_list:
            if h in existing:
                known_values.append(existing[h])
            else:
                unknown_hashes.append(h)

        return known_values, unknown_hashes

    # (2) Plus tard: insérer en lot ce que l'API renvoie pour les inconnus, SANS remplacer l'existant
    # mapping = {hash: valeur_python_JSON_sérialisable}
    def insert_new_many(self, mapping: Dict[str, Any]) -> int:
        """
        Insère uniquement les entrées nouvelles (ne remplace jamais une valeur existante).
        Retourne le nombre d'items passés (pas le nombre effectivement inséré).
        """
        if not mapping:
            return 0
        rows = []
        for k, val in mapping.items():
            payload = json.dumps(val, ensure_ascii=False, separators=(",", ":"))
            rows.append((k, payload))
        with self.conn:  # transaction
            self.conn.executemany(
                "INSERT OR IGNORE INTO kv (k, v) VALUES (?, ?)",
                rows,
            )
        return len(rows)
