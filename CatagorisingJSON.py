import json
from collections import defaultdict, Counter
from typing import Any, Dict, List, Tuple, Set
from sklearn.cluster import DBSCAN
import numpy as np

def type_of(v: Any) -> str:
    if v is None: return "null"
    if isinstance(v, bool): return "bool"
    if isinstance(v, (int, float)): return "number"
    if isinstance(v, str): return "string"
    if isinstance(v, list): return "array"
    if isinstance(v, dict): return "object"
    return "unknown"

def flatten_json(d: Any, prefix: str = "") -> List[Tuple[str, str]]:
    out = []
    t = type_of(d)
    if t in ("string", "number", "bool", "null"):
        out.append((prefix.rstrip("."), t))
        return out
    if t == "object":
        if not prefix:
            base = ""
        else:
            base = prefix + "."
        for k, v in d.items():
            out.extend(flatten_json(v, base + k))
        return out
    if t == "array":
        arr_path = prefix + "[]"
        out.append((arr_path.rstrip("."), "array"))
        if len(d) > 0:
            N = min(3, len(d))
            paths_seen = set()
            for i in range(N):
                for p, tp in flatten_json(d[i], arr_path):
                    if p not in paths_seen:
                        out.append((p, tp))
                        paths_seen.add(p)
        return out
    if t == "unknown":
        out.append((prefix.rstrip("."), "unknown"))
        return out
    return out

def build_signature(d: Dict[str, Any]) -> Tuple[Set[str], Dict[str, Counter]]:
    pairs = flatten_json(d)
    key_set = set(k for k, _ in pairs if k)
    type_counter = defaultdict(Counter)
    for k, t in pairs:
        if k:
            type_counter[k][t] += 1
    return key_set, type_counter

def jaccard_distance(a: Set[str], b: Set[str]) -> float:
    if not a and not b: return 0.0
    inter = len(a & b)
    union = len(a | b)
    return 1.0 - (inter / union if union else 0.0)

def type_mismatch_penalty(a_types: Dict[str, Counter], b_types: Dict[str, Counter]) -> float:
    overlap = set(a_types.keys()) & set(b_types.keys())
    if not overlap: return 0.0
    mismatches = 0
    for k in overlap:
        a_top = a_types[k].most_common(1)[0][0]
        b_top = b_types[k].most_common(1)[0][0]
        if a_top != b_top:
            mismatches += 1
    return mismatches / max(1, len(overlap))

def pairwise_distance(signatures):
    n = len(signatures)
    D = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i+1, n):
            ak, at = signatures[i]
            bk, bt = signatures[j]
            d = jaccard_distance(ak, bk)
            d += 0.3 * type_mismatch_penalty(at, bt)
            D[i, j] = D[j, i] = d
    return D

def cluster_json_objects(objs: List[Dict[str, Any]], eps: float = 0.35, min_samples: int = 2):
    signatures = [build_signature(o) for o in objs]
    D = pairwise_distance(signatures)
    labels = DBSCAN(eps=eps, min_samples=min_samples, metric="precomputed").fit(D).labels_
    clusters = defaultdict(list)
    for idx, lab in enumerate(labels):
        clusters[lab].append(idx)
    return labels, clusters, signatures

def infer_schema(objs: List[Dict[str, Any]], indices: List[int]) -> Dict[str, Dict[str, Any]]:
    field_counts = Counter()
    type_counts = defaultdict(Counter)
    example_values = {}
    n = len(indices)

    for i in indices:
        pairs = flatten_json(objs[i])
        seen_in_record = set()
        for k, t in pairs:
            if not k: continue
            type_counts[k][t] += 1
            if k not in seen_in_record:
                field_counts[k] += 1
                seen_in_record.add(k)
            if k not in example_values:
                v = get_example_at_path(objs[i], k)
                if not isinstance(v, (dict, list)):
                    example_values[k] = v

    schema = {}
    for k in sorted(field_counts.keys()):
        schema[k] = {
            "presence": field_counts[k] / n,
            "types": dict(type_counts[k]),
            "example": example_values.get(k, None),
        }
    return schema

def get_example_at_path(d: Any, keypath: str) -> Any:
    parts = keypath.split(".")
    cur = d
    for p in parts:
        if p.endswith("[]"):
            p = p[:-2]
            cur = (cur.get(p) if isinstance(cur, dict) else None)
            if isinstance(cur, list) and cur:
                cur = cur[0]
            else:
                return None
        else:
            if isinstance(cur, dict):
                cur = cur.get(p)
            else:
                return None
    return cur

def recommend_storage(schema: Dict[str, Dict[str, Any]]) -> str:
    presence = [v["presence"] for v in schema.values()]
    avg_presence = sum(presence)/len(presence) if presence else 1.0

    array_fields = [k for k in schema if "[]" in k]
    max_depth = max((k.count(".") for k in schema), default=0)

    type_drift = sum(1 for v in schema.values() if len(v["types"]) > 1) / max(1, len(schema))

    if array_fields or max_depth >= 3 or type_drift > 0.15 or avg_presence < 0.7:
        return "NoSQL"
    return "SQL"

def categorize_and_model(json_objects: List[Dict[str, Any]]):
    labels, clusters, _ = cluster_json_objects(json_objects)
    result = {}

    for lab, idxs in clusters.items():
        if lab == -1:
            for i in idxs:
                schema = infer_schema(json_objects, [i])
                storage = recommend_storage(schema)
                result[f"cluster_single_{i}"] = {
                    "indices": [i],
                    "schema": schema,
                    "storage": storage,
                    "proposed_entities": propose_entity_names(schema)
                }
        else:
            schema = infer_schema(json_objects, idxs)
            storage = recommend_storage(schema)
            result[f"cluster_{lab}"] = {
                "indices": idxs,
                "schema": schema,
                "storage": storage,
                "proposed_entities": propose_entity_names(schema)
            }
    return labels, result

def propose_entity_names(schema: Dict[str, Dict[str, Any]]) -> List[str]:
    roots = [k.split(".")[0].replace("[]", "") for k in schema.keys()]
    freq = Counter(roots)
    common = [r for r, _ in freq.most_common(3)]
    hints = {
        "user":"User", "person":"Person", "customer":"Customer",
        "order":"Order", "item":"Item", "product":"Product",
        "sensor":"SensorReading", "reading":"SensorReading",
        "event":"Event", "log":"Log", "transaction":"Transaction",
        "image":"ImageMeta", "video":"VideoMeta", "media":"MediaMeta"
    }
    names = []
    for r in common:
        names.append(hints.get(r.lower(), r.capitalize()))
    if not names:
        names = ["Entity"]
    return list(dict.fromkeys(names))
