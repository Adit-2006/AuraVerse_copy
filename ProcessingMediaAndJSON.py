
import os, sys, json, glob
from typing import List, Dict, Any

from Classify_image import ClassifyImages
from Save_Classified_File import save_media_to_category, save_json_sqlite, save_json_nosql
from CatagorisingJSON import categorize_and_model

MEDIA_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tif", ".tiff", ".mp4", ".mov", ".avi", ".mkv", ".heic", ".avif"}
JSON_EXTS  = {".json", ".jsonl", ".ndjson"}

def is_media(path: str) -> bool:
    return os.path.splitext(path.lower())[1] in MEDIA_EXTS

def is_json(path: str) -> bool:
    return os.path.splitext(path.lower())[1] in JSON_EXTS

def load_json_objects(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()
    if not text:
        return []
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return [data]
        if isinstance(data, list):
            return [x for x in data if isinstance(x, dict)]
        return []
    except json.JSONDecodeError:
        objs = []
        with open(path, "r", encoding="utf-8") as f2:
            for line in f2:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if isinstance(obj, dict):
                        objs.append(obj)
                except json.JSONDecodeError:
                    pass
        return objs

def process_media(paths: List[str], out_base: str = "media_store") -> None:
    for p in paths:
        try:
            label = ClassifyImages(p)
            saved = save_media_to_category(p, label, base_dir=out_base)
            print(f"[MEDIA] {p} -> label={label} -> {saved}")
        except Exception as e:
            print(f"[MEDIA][ERR] {p}: {e}")

def process_json(paths: List[str], db_path: str = "store.db", nosql_dir: str = "nosql_store", metadata: Dict[str, Any] = None) -> None:
    """
    Ensures each JSON object has a `metacomments` field if metadata is provided,
    and writes an augmented copy of each input file as <name>.with_meta.json.
    """
    per_file: List[Tuple[str, List[Dict[str, Any]]]] = []
    all_objs: List[Dict[str, Any]] = []

    for p in paths:
        objs = load_json_objects(p)
        if metadata:
            # accept "metacomments", or fallbacks
            meta_comment = metadata.get("metacomments") or metadata.get("comment") or metadata.get("comments") if isinstance(metadata, dict) else str(metadata)
            for obj in objs:
                if meta_comment is not None and "metacomments" not in obj:
                    obj["metacomments"] = meta_comment
                obj["_meta"] = metadata
        per_file.append((p, objs))
        all_objs.extend(objs)

    if not all_objs:
        print("[JSON] No JSON objects found.")
        return

    # NEW: write augmented copies next to sources
    for src, objs in per_file:
        base, ext = os.path.splitext(src)
        out_path = f"{base}.with_meta.json"
        try:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(objs, f, ensure_ascii=False, indent=2)
            print(f"[JSON][WRITE] {out_path} (objects={len(objs)})")
        except Exception as e:
            print(f"[JSON][WRITE][ERR] {src}: {e}")

    # Existing clustering + storage paths
    labels, clusters_info = categorize_and_model(all_objs)

    for cname, details in clusters_info.items():
        idxs = details["indices"]
        storage = details["storage"]
        entities = details["proposed_entities"]
        entity = entities[0] if entities else "Entity"
        schema = details["schema"]
        batch = [all_objs[i] for i in idxs]

        if storage == "SQL":
            save_json_sqlite(entity_name=entity, schema=schema, objects=batch, db_path=db_path)
            print(f"[JSON][SQL] cluster={cname} -> table={entity} rows={len(batch)} db={db_path}")
        else:
            path = save_json_nosql(entity_name=entity, objects=batch, base_dir=nosql_dir)
            print(f"[JSON][NoSQL] cluster={cname} -> collection={os.path.basename(path)} docs={len(batch)} dir={nosql_dir}")


    if metadata:
        if isinstance(metadata, dict):
            meta_comment = metadata.get("metacomments") or metadata.get("comment") or metadata.get("comments")
        else:
            meta_comment = str(metadata)
        for obj in all_objs:
            if meta_comment is not None and "metacomments" not in obj:
                obj["metacomments"] = meta_comment
            obj["_meta"] = metadata

    labels, clusters_info = categorize_and_model(all_objs)

    for cname, details in clusters_info.items():
        idxs = details["indices"]
        storage = details["storage"]
        entities = details["proposed_entities"]
        entity = entities[0] if entities else "Entity"
        schema = details["schema"]
        batch = [all_objs[i] for i in idxs]

        if storage == "SQL":
            save_json_sqlite(entity_name=entity, schema=schema, objects=batch, db_path=db_path)
            print(f"[JSON][SQL] cluster={cname} -> table={entity} rows={len(batch)} db={db_path}")
        else:
            path = save_json_nosql(entity_name=entity, objects=batch, base_dir=nosql_dir)
            print(f"[JSON][NoSQL] cluster={cname} -> collection={os.path.basename(path)} docs={len(batch)} dir={nosql_dir}")

def collect_inputs(input_path: str) -> (List[str], List[str]):
    media, jsons = [], []
    paths = []
    if os.path.isdir(input_path):
        for p in glob.glob(os.path.join(input_path, "**", "*.*"), recursive=True):
            paths.append(p)
    else:
        paths = [input_path]

    for p in paths:
        if is_media(p):
            media.append(p)
        elif is_json(p):
            jsons.append(p)
    return media, jsons

def main():
    """
    Usage:
      python main_with_metacomments.py <path> [--meta '{"metacomments":"note","source":"ui"}']
    """
    if len(sys.argv) < 2:
        print("Usage: python main_with_metacomments.py <path> [--meta '{\"metacomments\":\"x\"}']")
        sys.exit(1)

    input_path = sys.argv[1]
    meta = None
    if "--meta" in sys.argv:
        i = sys.argv.index("--meta")
        if i + 1 < len(sys.argv):
            try:
                meta = json.loads(sys.argv[i + 1])
            except Exception:
                meta = {"metacomments": sys.argv[i + 1]}

    media, jsons = collect_inputs(input_path)

    if media:
        print(f"[INFO] Processing {len(media)} media file(s)...")
        process_media(media)

    if jsons:
        print(f"[INFO] Processing {len(jsons)} JSON file(s)...")
        process_json(jsons, metadata=meta)

    if not media and not jsons:
        print("[INFO] Nothing to process (no supported files found).")

if __name__ == "__main__":
    main()
