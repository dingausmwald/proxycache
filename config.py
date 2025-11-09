# config.py
# -*- coding: utf-8 -*-
import os
import json
import logging

# Базовый адрес llama.cpp server (должен поднимать /v1/chat/completions и /slots)
LLAMA_SERVER_URL = os.getenv("LLAMA_SERVER_URL", "http://127.0.0.1:8000")  # [web:11]

# Число слотов на сервере (см. -np у llama-server)
SLOTS_COUNT = int(os.getenv("SLOTS_COUNT", "4"))  # [web:40]

# Таймауты HTTP‑клиента
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "600"))

# Системный промпт (опционально, для канонизации префикса)
SYSTEM_PROMPT_FILE = os.getenv("SYSTEM_PROMPT_FILE")

# Идентификатор модели для /v1/models
DEFAULT_MODEL_ID = os.getenv("MODEL_ID", "llama.cpp")

# Прибитые ключи, недоступные для эвикции
PINNED_PREFIX_KEYS = set(json.loads(os.getenv("PINNED_KEYS", "[]")))

# Размер блока по словам для блочной эвристики (обычно 16)
DEFAULT_WORDS_PER_BLOCK = int(os.getenv("WORDS_PER_BLOCK", "16"))

# Лимит сканирования .meta на диске
DISK_META_SCAN_LIMIT = int(os.getenv("DISK_META_SCAN_LIMIT", "200"))

# Порог «малого» запроса
THRESHOLD_MODE = os.getenv("THRESHOLD_MODE", "chars").lower()
MIN_PREFIX_CHARS = int(os.getenv("MIN_PREFIX_CHARS", "5000"))
MIN_PREFIX_WORDS = int(os.getenv("MIN_PREFIX_WORDS", "1000"))
MIN_PREFIX_BLOCKS = int(os.getenv("MIN_PREFIX_BLOCKS", "20"))

# ЕДИНЫЙ порог похожести (доля LCP относительно min(len(req), len(candidate)))
# Применяется и к active‑lcp, и к restore‑lcp. exact‑match принимается всегда.
SIMILARITY_MIN_RATIO = float(os.getenv("SIMILARITY_MIN_RATIO", "0.85"))  # [web:40]

# Каталог локальных .meta (служебные метаданные прокси для дисковой эвристики)
LOCAL_META_DIR = os.getenv("LOCAL_META_DIR", "./kvslots_meta")  # [web:40]

# Опционально: локальный путь, смонтированный к --slot-save-path у сервера, для логирования размеров .bin
SLOT_SAVE_MOUNT = os.getenv("SLOT_SAVE_MOUNT")  # [web:254][web:40]

# Логирование
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
)
