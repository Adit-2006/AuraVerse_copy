
import os, json, sqlite3, re
from pathlib import Path
from typing import Dict, Any, List

# ---------- Helpers ----------
def ensure_dir(p: str) -> None:
    Path(p).mkdir(parents=True, exist_ok=True)

def safe_ident(name: str) -> str:
    # safe SQL identifier: letters, numbers, underscore
    name = re.sub(r"[^A-Za-z0-9_]", "_", name)
    if not name or name[0].isdigit():
        name = f"t_{name}"
    return name.lower()

def map_type(t: str) -> str:
    # JSON type -> SQLite type
    return {
        "number": "REAL",
        "string": "TEXT",
        "bool":   "INTEGER",
        "null":   "TEXT",
        "unknown":"TEXT",
    }.get(t, "TEXT")

# ---------- MEDIA (images/videos) ----------
def save_media_to_category(src_path: str, label: str, base_dir: str = "media_store") -> str:
    """
    Create directory per predicted label and copy/move file into it.
    """
    ensure_dir(base_dir)
    category_dir = os.path.join(base_dir, label)
    ensure_dir(category_dir)
    # Keep filename, avoid collisions
    fname = os.path.basename(src_path)
    out_path = os.path.join(category_dir, fname)
    # If collision, add suffix
    stem, ext = os.path.splitext(fname)
    k = 1
    while os.path.exists(out_path):
        out_path = os.path.join(category_dir, f"{stem}_{k}{ext}")
        k += 1
    # Move (or copy if you prefer)
    import shutil
    shutil.copy2(src_path, out_path)
    return out_path

# ---------- JSON: SQL path (SQLite) ----------
def _create_parent_table(conn: sqlite3.Connection, table: str, schema: Dict[str, Dict[str, Any]]) -> None:
    """
    Create parent table with scalar fields only (no arrays).
    Also add raw_json column for completeness & schema evolution.
    """
    cols = ["id INTEGER PRIMARY KEY AUTOINCREMENT", "raw_json TEXT"]
    for key, meta in schema.items():
        if "[]" in key:
            continue  # skip arrays for parent
        # only keep shallow-ish scalar fields to avoid exploding columns
        if meta["types"]:
            sqltype = map_type(next(iter(meta["types"].keys())))
        else:
            sqltype = "TEXT"
        col = safe_ident(key.replace(".", "_"))
        cols.append(f"{col} {sqltype}")
    ddl = f'CREATE TABLE IF NOT EXISTS {safe_ident(table)} (\n  ' + ",\n  ".join(cols) + "\n);"
    conn.execute(ddl)

def _create_child_table(conn: sqlite3.Connection, parent: str, array_root: str, schema: Dict[str, Dict[str, Any]]) -> None:
    """
    For each array like items[], create child table parent_items with FK.
    Include primitive fields under that array (e.g., items[].price).
    """
    base = array_root.replace("[]", "")
    child = f"{parent}_{base.replace('.', '_')}"  # parent_items
    cols = ["id INTEGER PRIMARY KEY AUTOINCREMENT", f"{safe_ident(parent)}_id INTEGER", "raw_json TEXT"]
    for key, meta in schema.items():
        if not key.startswith(array_root + "."):
            continue
        if "[]" in key:
            continue  # skip nested arrays for simplicity
        col = safe_ident(key.replace(array_root + ".", "").replace(".", "_"))
        if meta["types"]:
            sqltype = map_type(next(iter(meta["types"].keys())))
        else:
            sqltype = "TEXT"
        cols.append(f"{col} {sqltype}")
    ddl = f'CREATE TABLE IF NOT EXISTS {safe_ident(child)} (\n  ' + ",\n  ".join(cols) + "\n);"
    conn.execute(ddl)

def _insert_parent(conn: sqlite3.Connection, table: str, obj: Dict[str, Any], schema: Dict[str, Dict[str, Any]]) -> int:
    """
    Insert one parent row; return row id.
    """
    cols, vals = ["raw_json"], [json.dumps(obj, ensure_ascii=False)]
    for key in schema:
        if "[]" in key:
            continue
        # walk object by keypath
        v = _get_by_keypath(obj, key)
        if isinstance(v, (dict, list)):
            continue
        cols.append(safe_ident(key.replace(".", "_")))
        vals.append(v)
    placeholders = ",".join(["?"] * len(vals))
    sql = f"INSERT INTO {safe_ident(table)} ({','.join(cols)}) VALUES ({placeholders})"
    cur = conn.execute(sql, vals)
    return int(cur.lastrowid)

def _insert_children(conn: sqlite3.Connection, parent: str, parent_id: int, obj: Dict[str, Any], schema: Dict[str, Dict[str, Any]]) -> None:
    """
    Insert rows for each array-of-objects under this parent.
    """
    arrays = sorted({k.split(".")[0] for k in schema if "[]" in k})
    for arr_key in arrays:
        base = arr_key.replace("[]", "")
        child_table = f"{parent}_{base.replace('.', '_')}"
        arr = _get_by_keypath(obj, base)
        if not isinstance(arr, list):
            continue
        for el in arr:
            raw = json.dumps(el, ensure_ascii=False)
            cols = [f"{safe_ident(parent)}_id", "raw_json"]
            vals = [parent_id, raw]
            # primitive subfields
            for k in schema:
                if not k.startswith(arr_key + "."):
                    continue
                sub_key = k.replace(arr_key + ".", "")
                sv = _get_by_keypath(el, sub_key.replace("[]", ""))
                if isinstance(sv, (dict, list)):
                    continue
                cols.append(safe_ident(sub_key.replace(".", "_")))
                vals.append(sv)
            placeholders = ",".join(["?"] * len(vals))
            sql = f"INSERT INTO {safe_ident(child_table)} ({','.join(cols)}) VALUES ({placeholders})"
            conn.execute(sql, vals)

def _get_by_keypath(d: Dict[str, Any], keypath: str):
    cur = d
    for part in keypath.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
        if cur is None:
            return None
    return cur

def save_json_sqlite(entity_name: str, schema: Dict[str, Dict[str, Any]], objects: List[Dict[str, Any]], db_path: str = "store.db") -> None:
    """
    Create/extend SQLite schema and insert the batch.
    - Parent table: scalar fields + raw_json
    - Child tables: one per array-of-objects under parent
    """
    parent = safe_ident(entity_name or "entity")
    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA foreign_keys = ON;")
        _create_parent_table(conn, parent, schema)
        # Create child tables for arrays
        for k in sorted(schema.keys()):
            if k.endswith("[]"):
                _create_child_table(conn, parent, k, schema)
        # Insert rows
        for obj in objects:
            pid = _insert_parent(conn, parent, obj, schema)
            _insert_children(conn, parent, pid, obj, schema)
        conn.commit()

# ---------- JSON: NoSQL path (JSONL files by collection) ----------
def save_json_nosql(entity_name: str, objects: List[Dict[str, Any]], base_dir: str = "nosql_store") -> str:
    """
    Append objects to a JSONL file per entity (simple document-store emulation).
    """
    ensure_dir(base_dir)
    coll = os.path.join(base_dir, f"{safe_ident(entity_name or 'entity')}.jsonl")
    with open(coll, "a", encoding="utf-8") as f:
        for obj in objects:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    return coll
