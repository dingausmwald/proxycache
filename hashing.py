# hashing.py
# -*- coding: utf-8 -*-

"""
Raw-хэширование: raw_prefix без ролей, только контент, разделённый двойным переводом строки.
Блоки по 100 слов, LCP по полным SHA256-хэшам. Key=sha256(raw_prefix).
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
    # Без ролей, разделитель — пустая строка между репликами
    text = "\n\n".join(parts).strip()
    log.debug("raw_prefix len_chars=%d", len(text))
    return text

def words_from_text(text: str) -> List[str]:
    return re.findall(r'\w+', text.lower())

def block_hashes_from_text(text: str, wpb: int = WORDS_PER_BLOCK) -> List[str]:
    words = words_from_text(text)
    hashes = []
    for i in range(0, len(words), wpb):
        block = " ".join(words[i:i+wpb])
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
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def scan_all_meta() -> List[Dict]:
    files = sorted(glob.glob(os.path.join(META_DIR, "*.meta.json")), key=os.path.getmtime, reverse=True)
    metas = []
    for f in files:
        try:
            with open(f, "r", encoding="utf-8") as fd:
                meta = json.load(fd)
                metas.append(meta)
        except Exception as e:
            log.warning("scan_meta_fail %s: %s", f, e)
    log.debug("scan_meta n_found=%d", len(metas))
    return metas

def find_best_restore_candidate(req_blocks: List[str], wpb: int, th: float) -> Optional[Tuple[str, float]]:
    metas = scan_all_meta()
    best_key, best_ratio = None, 0.0
    for meta in metas:
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

def write_meta(key: str, prefix_text: str, blocks: List[str], wpb: int):
    meta = {
        "key": key,
        "prefix_len": len(prefix_text),
        "wpb": wpb,
        "blocks": blocks,
        "timestamp": time.time(),
    }
    path = os.path.join(META_DIR, f"{key}.meta.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
