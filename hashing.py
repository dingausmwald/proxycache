# hashing.py

# -*- coding: utf-8 -*-

"""
Raw-хэширование: raw_prefix без ролей, только контент, разделённый двойным переводом строки.

Блоки по 100 слов, LCP по полным SHA256-хэшам.
Key = sha256(model_id + "\\n" + raw_prefix), т.е. модель включена в ключ.

Метафайлы содержат:
- key
- model_id
- prefix_len
- wpb
- blocks
- timestamp
"""

import os
import json
import hashlib
import re
import time
import glob
import logging
from typing import List, Dict, Optional, Tuple

from config import META_DIR, WORDS_PER_BLOCK

log = logging.getLogger(__name__)


def raw_prefix(messages: List[Dict]) -> str:
    parts = []
    for msg in messages or []:
        content = msg.get("content", "")
        if isinstance(content, str):
            content = content.strip()
        else:
            content = str(content).strip()
        if content:
            parts.append(content)
    text = "\n\n".join(parts).strip()
    log.debug("raw_prefix len_chars=%d", len(text))
    return text


def words_from_text(text: str) -> List[str]:
    return re.findall(r"\w+", text.lower())


def block_hashes_from_text(text: str, wpb: int = WORDS_PER_BLOCK) -> List[str]:
    words = words_from_text(text)
    hashes: List[str] = []
    for i in range(0, len(words), wpb):
        block = " ".join(words[i:i + wpb])
        h = hashlib.sha256(block.encode("utf-8")).hexdigest()
        hashes.append(h)
    log.debug("block_hashes n_blocks=%d wpb=%d", len(hashes), wpb)
    return hashes


def lcp_blocks(blocks1: List[str], blocks2: List[str]) -> int:
    n = min(len(blocks1), len(blocks2))
    i = 0
    while i < n and blocks1[i] == blocks2[i]:
        i += 1
    return i


def prefix_key_sha256(text: str) -> str:
    """
    Базовая SHA256-обёртка; для кеша в неё передаём model_id + "\\n" + raw_prefix.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def scan_all_meta() -> List[Dict]:
    files = sorted(
        glob.glob(os.path.join(META_DIR, "*.meta.json")),
        key=os.path.getmtime,
        reverse=True,
    )
    metas: List[Dict] = []
    for f in files:
        try:
            with open(f, "r", encoding="utf-8") as fd:
                meta = json.load(fd)
                metas.append(meta)
        except Exception as e:
            log.warning("scan_meta_fail %s: %s", f, e)
    log.debug("scan_meta n_found=%d", len(metas))
    return metas


def find_best_restore_candidate(
    req_blocks: List[str],
    wpb: int,
    th: float,
    model_id: str,
) -> Optional[Tuple[str, float]]:
    """
    Ищет лучший кандидат для restore среди мета-файлов ТОЛЬКО текущей модели.

    Фильтруем по:
    - meta["model_id"] == model_id
    - meta["wpb"] == wpb
    """
    metas = scan_all_meta()
    best_key: Optional[str] = None
    best_ratio = 0.0

    for meta in metas:
        if meta.get("model_id") != model_id:
            continue
        if int(meta.get("wpb") or 0) != wpb:
            continue

        cand_blocks = meta.get("blocks") or []
        lcp = lcp_blocks(req_blocks, cand_blocks)
        denom = max(1, min(len(req_blocks), len(cand_blocks)))
        ratio = lcp / denom

        if ratio >= th and ratio > best_ratio:
            best_ratio = ratio
            best_key = meta.get("key")

    return (best_key, best_ratio) if best_key else None


def write_meta(
    key: str,
    prefix_text: str,
    blocks: List[str],
    wpb: int,
    model_id: str,
) -> None:
    """
    Записывает/перезаписывает meta-файл для key, привязанный к конкретной модели.
    """
    meta = {
        "key": key,
        "model_id": model_id,
        "prefix_len": len(prefix_text),
        "wpb": wpb,
        "blocks": blocks,
        "timestamp": time.time(),
    }
    path = os.path.join(META_DIR, f"{key}.meta.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


def touch_meta(key: str) -> None:
    """
    Обновляет timestamp в существующем meta-файле key.meta.json.
    """
    path = os.path.join(META_DIR, f"{key}.meta.json")
    try:
        with open(path, "r+", encoding="utf-8") as f:
            try:
                meta = json.load(f)
            except Exception as e:
                log.warning("touch_meta_read_fail key=%s: %s", key[:16], e)
                return
            meta["timestamp"] = time.time()
            f.seek(0)
            json.dump(meta, f, indent=2, ensure_ascii=False)
            f.truncate()
        log.debug("touch_meta_ok key=%s", key[:16])
    except FileNotFoundError:
        log.warning("touch_meta_missing key=%s", key[:16])
    except Exception as e:
        log.warning("touch_meta_fail key=%s: %s", key[:16], e)

def cleanup_old_cache(
    cache_dir: str,
    meta_dir: str,
    max_age_hours: int = 168,
    max_size_gb: float = 50.0,
) -> Dict[str, int]:
    """
    Cleanup old cache files based on age and total size.
    Returns stats: {"deleted_by_age": N, "deleted_by_size": N, "total_freed_bytes": N}
    """
    import time
    
    stats = {"deleted_by_age": 0, "deleted_by_size": 0, "total_freed_bytes": 0}
    
    if not cache_dir or not os.path.isdir(cache_dir):
        log.warning("cleanup_skip: cache_dir not set or doesn't exist: %s", cache_dir)
        return stats
    
    now = time.time()
    max_age_seconds = max_age_hours * 3600 if max_age_hours > 0 else float('inf')
    max_size_bytes = max_size_gb * 1024 * 1024 * 1024
    
    # Get all cache files with their stats
    cache_files = []
    for f in os.listdir(cache_dir):
        filepath = os.path.join(cache_dir, f)
        if os.path.isfile(filepath):
            try:
                stat = os.stat(filepath)
                cache_files.append({
                    "path": filepath,
                    "basename": f,
                    "size": stat.st_size,
                    "mtime": stat.st_mtime,
                })
            except OSError:
                continue
    
    # Delete by age
    for cf in cache_files[:]:
        if now - cf["mtime"] > max_age_seconds:
            try:
                os.remove(cf["path"])
                stats["deleted_by_age"] += 1
                stats["total_freed_bytes"] += cf["size"]
                cache_files.remove(cf)
                # Also remove meta file
                meta_path = os.path.join(meta_dir, f"{cf['basename']}.meta.json")
                if os.path.exists(meta_path):
                    os.remove(meta_path)
                log.info("cleanup_age: deleted %s (age: %.1f hours)", 
                        cf["basename"][:16], (now - cf["mtime"]) / 3600)
            except OSError as e:
                log.warning("cleanup_age_fail: %s: %s", cf["basename"][:16], e)
    
    # Calculate current total size
    total_size = sum(cf["size"] for cf in cache_files)
    
    # Delete by size (oldest first) until under limit
    if total_size > max_size_bytes:
        # Sort by mtime (oldest first)
        cache_files.sort(key=lambda x: x["mtime"])
        
        for cf in cache_files:
            if total_size <= max_size_bytes:
                break
            try:
                os.remove(cf["path"])
                stats["deleted_by_size"] += 1
                stats["total_freed_bytes"] += cf["size"]
                total_size -= cf["size"]
                # Also remove meta file
                meta_path = os.path.join(meta_dir, f"{cf['basename']}.meta.json")
                if os.path.exists(meta_path):
                    os.remove(meta_path)
                log.info("cleanup_size: deleted %s (freed: %.1f MB)", 
                        cf["basename"][:16], cf["size"] / 1024 / 1024)
            except OSError as e:
                log.warning("cleanup_size_fail: %s: %s", cf["basename"][:16], e)
    
    log.info("cleanup_done: deleted_by_age=%d deleted_by_size=%d freed=%.1f MB",
             stats["deleted_by_age"], stats["deleted_by_size"], 
             stats["total_freed_bytes"] / 1024 / 1024)
    
    return stats
