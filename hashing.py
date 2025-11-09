# hashing.py
# -*- coding: utf-8 -*-
import os
import re
import hashlib
import time
import glob
import json
from typing import List, Dict, Optional

from config import SYSTEM_PROMPT_FILE, LOCAL_META_DIR

# Гарантируем существование каталога для .meta (локальные метаданные прокси)
os.makedirs(LOCAL_META_DIR, exist_ok=True)

def normalize_content(content) -> str:
    """
    Нормализация OpenAI content:
    - Если строка -> trim и вернуть.
    - Если список частей -> взять только text-части и склеить пробелом.
    - Иное -> привести к строке безопасно.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for p in content:
            # Ожидаемые элементы: {"type":"text","text":"..."} или {"type":"image_url", ...}
            if isinstance(p, dict) and p.get("type") == "text":
                t = p.get("text")
                if isinstance(t, str):
                    t = t.strip()
                    if t:
                        parts.append(t)
        return " ".join(parts).strip()
    # На всякий случай: нестроковые типы
    try:
        return str(content).strip()
    except Exception:
        return ""

def canonical_chat_prefix(messages: Optional[List[Dict]],
                          system_prompt_file: Optional[str] = SYSTEM_PROMPT_FILE,
                          add_bos: bool = True) -> str:
    """
    Канонически сериализует чат в строку префикса:
    - BOS (опционально),
    - системный промпт из файла (если задан),
    - последовательность ролей с нормализованным content,
    - маркер начала ответа ассистента.
    """
    parts: List[str] = []
    if add_bos:
        parts.append("<|bos|>\n")

    if system_prompt_file and os.path.exists(system_prompt_file):
        with open(system_prompt_file, "r", encoding="utf-8") as f:
            sys_text = f.read().strip()
        if sys_text:
            parts.append(f"<|system|>\n{sys_text}\n")

    for m in messages or []:
        role = m.get("role", "user")
        raw = m.get("content")
        content = normalize_content(raw)
        if role == "system" and content:
            parts.append(f"<|system|>\n{content}\n")
        elif role == "assistant":
            parts.append(f"<|assistant|>\n{content}\n")
        elif role == "user":
            parts.append(f"<|user|>\n{content}\n")
        else:
            parts.append(f"<|user:{role}|>\n{content}\n")

    parts.append("<|assistant|>\n")
    return "".join(parts)

def words_from_text(text: str) -> List[str]:
    # Деление по пробельным символам; убираем пустые
    return [w for w in re.split(r"\s+", text.strip()) if w]

def block_hashes_from_text(text: str, words_per_block: int) -> List[str]:
    """
    Делим текст на блоки по N словам и считаем SHA-256 каждого блока — эвристика LCP без токенизации.
    """
    ws = words_from_text(text)
    blocks: List[str] = []
    for i in range(0, len(ws), words_per_block):
        block_words = ws[i:i+words_per_block]
        block_text = " ".join(block_words)
        h = hashlib.sha256(block_text.encode("utf-8")).hexdigest()
        blocks.append(h)
    return blocks

def lcp_blocks(a: List[str], b: List[str]) -> int:
    """
    Длина общего префикса по блокам (в блоках).
    """
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i

def prefix_key_sha256(prefix_text: str) -> str:
    """
    Ключ префикса — SHA-256 от канонической строки чата.
    """
    return hashlib.sha256(prefix_text.encode("utf-8")).hexdigest()

def meta_filename_for_key_local(key: str) -> str:
    """
    Путь к локальному .meta для ключа (хранится у прокси, не в --slot-save-path сервера).
    """
    return os.path.join(LOCAL_META_DIR, f"slotcache_{key}.meta.json")

def write_meta_for_key_local(key: str,
                             prefix_text: str,
                             model_id: str,
                             words_per_block: int,
                             block_hashes: List[str]) -> None:
    """
    Запись локальной .meta: ключ, модель, длина префикса, размер блока, список хешей блоков.
    """
    meta = {
        "key": key,
        "model_id": model_id,
        "prefix_len_chars": len(prefix_text),
        "blocks": block_hashes,
        "words_per_block": words_per_block,
        "updated_at": int(time.time()),
    }
    with open(meta_filename_for_key_local(key), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False)

def scan_all_meta_local(limit: int) -> List[Dict]:
    """
    Чтение последних .meta из каталога LOCAL_META_DIR (для выбора лучшего restore-кандидата).
    """
    files = sorted(glob.glob(os.path.join(LOCAL_META_DIR, "slotcache_*.meta.json")), reverse=True)
    metas: List[Dict] = []
    for p in files[:limit]:
        try:
            with open(p, "r", encoding="utf-8") as f:
                metas.append(json.load(f))
        except Exception:
            continue
    return metas  # ВАЖНО: без опечаток
