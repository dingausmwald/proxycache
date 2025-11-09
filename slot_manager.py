# slot_manager.py
# -*- coding: utf-8 -*-
"""
Менеджер слотов (SlotManager) — контролирует выбор и жизненный цикл KV‑кэша в слотах llama.cpp:
- Активные «горячие» слоты используются только при достаточной похожести LCP (единый процентный порог).
- Restore с диска — только если .meta проходит тот же порог, иначе холодный старт.
- REJECT active‑lcp => отвергнутый hot‑слот исключается из кандидатов для назначения, чтобы не перетирать его кэш.
- Параллелизм: каждый слот защищён asyncio.Lock, при полной занятости — ожидание LRU.
- save/restore: работают только с basename имен файлов в каталоге --slot-save-path у llama.cpp.
"""

import os
import time
import httpx
import asyncio
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Set

from fastapi import HTTPException

from llama_client import LlamaClient
from config import (
    DEFAULT_MODEL_ID,
    DISK_META_SCAN_LIMIT,
    PINNED_PREFIX_KEYS,
    SLOT_SAVE_MOUNT,
    SIMILARITY_MIN_RATIO,
)
from hashing import scan_all_meta_local, lcp_blocks, write_meta_for_key_local  # блочная эвристика и локальные .meta [web:40]

@dataclass
class SlotBinding:
    """
    Описание состояния слота на стороне прокси (не у сервера):
    - key: SHA256 канонического префикса (уникальный идентификатор префикса).
    - prefix_text: канонический текст (для записи .meta и диагностики).
    - block_hashes: цепочка SHA256 блоков (по словам) — для LCP‑сравнения без токенизации.
    - last_use_ts: метка LRU (экранируем «самый старый» при эвикции/ожидании).
    - words_per_block: размер блока слов для совместимости поиска.
    - pinned: слот нельзя вытеснить.
    - hot: слот содержит «пригодный» KV‑кэш (участвует в активном поиске).
    - lock: асинхронная блокировка слота (обеспечивает ожидание при полной занятости).
    """
    key: str
    prefix_text: str
    block_hashes: List[str]
    last_use_ts: float
    words_per_block: int
    pinned: bool = False
    hot: bool = True
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

class SlotManager:
    """
    Основные задачи:
    - ensure_slot_for_request: найти подходящий слот (active‑exact / active‑lcp / restore‑lcp / cold),
      применяя единый порог похожести SIMILARITY_MIN_RATIO к lcp/min(len(req), len(candidate)).
    - acquire_free_or_cold_slot: предпочтительно выбрать свободный id, затем cold, затем свободный hot,
      иначе ждать LRU; опционально исключить «запрещённые» slot_id (например, отвергнутый hot).
    - save_slot_cache/restore_slot_cache: обёртки над HTTP API сервера (/slots?action=save|restore), basename.
    """
    def __init__(self, slots_count: int, llama: LlamaClient, model_id: str = DEFAULT_MODEL_ID, logger=None):
        self.slots_count = slots_count
        self.llama = llama
        self.model_id = model_id
        self._bindings: Dict[int, SlotBinding] = {}
        self._global_lock: Optional[asyncio.Lock] = None
        self.log = logger

    async def _lock_global(self) -> asyncio.Lock:
        if self._global_lock is None:
            self._global_lock = asyncio.Lock()
        return self._global_lock

    def _now(self) -> float:
        return time.time()

    def _eligible_active(self, b: SlotBinding, wpb: int) -> bool:
        # Активный кандидат только если слот «горячий» и совпадает words_per_block
        return b.hot and b.words_per_block == wpb  # [web:40]

    def _free_slot_ids(self) -> List[int]:
        # Свободные id (слоты без биндингов в прокси)
        used = set(self._bindings.keys())
        return [i for i in range(self.slots_count) if i not in used]

    def _lrud_ids(self, exclude: Set[int]) -> List[int]:
        # Идентификаторы занятых слотов по возрастанию last_use_ts (старые — первыми), за вычетом exclude
        ids = [sid for sid in self._bindings.keys() if sid not in exclude]
        return sorted(ids, key=lambda sid: self._bindings[sid].last_use_ts)

    def choose_victim_slot(self, exclude: Set[int]) -> Optional[int]:
        # LRU‑жертва среди неpinned и не исключённых слотов
        victim_id = None
        victim_ts = float("inf")
        for sid, b in self._bindings.items():
            if sid in exclude or b.pinned:
                continue
            if b.last_use_ts < victim_ts:
                victim_id = sid
                victim_ts = b.last_use_ts
        return victim_id

    def pick_cold_or_oldest_slot_for_small(self) -> Optional[int]:
        # Для малых — сначала cold, иначе LRU
        cold_id = None
        cold_ts = float("inf")
        for sid, b in self._bindings.items():
            if b.pinned:
                continue
            if not b.hot and b.last_use_ts < cold_ts:
                cold_id = sid
                cold_ts = b.last_use_ts
        if cold_id is not None:
            return cold_id
        return self.choose_victim_slot(exclude=set())

    async def mark_slot_cold(self, slot_id: int) -> None:
        b = self._bindings.get(slot_id)
        if b:
            b.hot = False

    async def acquire_free_or_cold_slot(self, exclude: Optional[Set[int]] = None) -> Tuple[int, SlotBinding]:
        """
        Универсальный захват слота (с lock) в порядке приоритета:
        1) Свободный (небинденный) id, не в exclude — создаём Binding и берём lock.
        2) Свободный cold (lock свободен), не в exclude — берём lock.
        3) Свободный hot (lock свободен), не в exclude — берём lock.
        4) Все заняты — ждём LRU вне exclude; если их нет — ждём LRU вообще.
        """
        ex = exclude or set()

        # 1) Свободные id
        free_ids = [sid for sid in self._free_slot_ids() if sid not in ex]
        if free_ids:
            sid = free_ids[0]
            b = SlotBinding(key="(empty)", prefix_text="", block_hashes=[], last_use_ts=self._now(),
                            words_per_block=16, pinned=False, hot=False)
            self._bindings[sid] = b
            await b.lock.acquire()
            return sid, b

        # 2) Свободные cold
        for sid in self._lrud_ids(ex):
            b = self._bindings[sid]
            if b.pinned:
                continue
            if not b.hot and not b.lock.locked():
                await b.lock.acquire()
                return sid, b

        # 3) Свободные hot
        for sid in self._lrud_ids(ex):
            b = self._bindings[sid]
            if b.pinned:
                continue
            if not b.lock.locked():
                await b.lock.acquire()
                return sid, b

        # 4) Все заняты — ждём LRU вне exclude
        lrud = self._lrud_ids(ex)
        if lrud:
            sid = lrud[0]
            b = self._bindings[sid]
            await b.lock.acquire()
            return sid, b

        # 4b) Совсем без альтернатив — ждём «любой» LRU
        all_ids = self._lrud_ids(exclude=set())
        if not all_ids:
            sid = 0
            b = SlotBinding(key="(empty)", prefix_text="", block_hashes=[], last_use_ts=self._now(),
                            words_per_block=16, pinned=False, hot=False)
            self._bindings[sid] = b
            await b.lock.acquire()
            return sid, b
        sid = all_ids[0]
        b = self._bindings[sid]
        await b.lock.acquire()
        return sid, b

    async def ensure_slot_for_request(
        self,
        req_key: str,
        req_prefix_text: str,
        req_blocks: List[str],
        words_per_block: int
    ) -> Tuple[int, SlotBinding, str, int, int]:
        """
        ЕДИНАЯ функция принятия решений:
        - Active exact: принимаем всегда, если нашли. (Совпадение «все блоки»)
        - Active LCP: доля lcp / min(len(req), len(slot)) >= SIMILARITY_MIN_RATIO? да — берём; нет — REJECT.
          REJECT => запоминаем rejected_sid, чтобы не назначать туда cold/restore при наличии альтернатив.
        - Restore LCP (.meta): доля lcp / min(len(req), len(meta)) >= SIMILARITY_MIN_RATIO? да — restore; нет — пропускаем.
        - Если restore не подходит — cold start в слот, подобранный с учётом exclude.
        Во всех случаях слот захватывается (lock) до завершения генерации.
        """
        gl = await self._lock_global()
        async with gl:
            best_sid = None
            best_binding = None
            best_lcp = 0

            # 1) exact среди активных hot
            for sid, b in self._bindings.items():
                if not self._eligible_active(b, words_per_block):
                    continue
                if b.block_hashes == req_blocks:
                    best_sid = sid
                    best_binding = b
                    best_lcp = len(req_blocks)
                    break

            # 2) lcp среди активных hot
            if best_sid is None:
                for sid, b in self._bindings.items():
                    if not self._eligible_active(b, words_per_block):
                        continue
                    l = lcp_blocks(req_blocks, b.block_hashes)
                    if l > best_lcp:
                        best_lcp = l
                        best_sid = sid
                        best_binding = b

            rejected_sid: Optional[int] = None
            if best_sid is not None and best_binding is not None:
                if best_lcp == len(req_blocks):
                    # Active exact — сразу берём
                    if not best_binding.lock.locked():
                        await best_binding.lock.acquire()
                    best_binding.last_use_ts = self._now()
                    if self.log:
                        self.log.info(f"active-exact slot_id={best_sid} lcp_blocks={best_lcp}")
                    return best_sid, best_binding, "active-exact", best_lcp, len(best_binding.block_hashes)
                else:
                    # Проверяем единый процентный порог
                    denom = max(1, min(len(req_blocks), len(best_binding.block_hashes)))
                    ratio = best_lcp / denom
                    if ratio >= SIMILARITY_MIN_RATIO:
                        if not best_binding.lock.locked():
                            await best_binding.lock.acquire()
                        best_binding.last_use_ts = self._now()
                        if self.log:
                            self.log.info(f"active-lcp ACCEPT slot_id={best_sid} lcp_blocks={best_lcp}/{denom} ratio={round(ratio,3)}")
                        return best_sid, best_binding, "active-lcp", best_lcp, len(best_binding.block_hashes)
                    else:
                        rejected_sid = best_sid
                        if self.log:
                            self.log.info(f"active-lcp REJECT slot_id={best_sid} lcp_blocks={best_lcp}/{denom} ratio={round(ratio,3)} min_ratio={SIMILARITY_MIN_RATIO}")

            # 3) Поиск restore‑кандидата среди .meta
            best_meta = None
            best_meta_lcp = 0
            best_meta_blocks_total = 0
            for meta in scan_all_meta_local(DISK_META_SCAN_LIMIT):
                if meta.get("model_id") != self.model_id:
                    continue
                if int(meta.get("words_per_block", words_per_block)) != words_per_block:
                    continue
                mblocks = meta.get("blocks") or []
                l = lcp_blocks(req_blocks, mblocks)
                if l > best_meta_lcp:
                    best_meta_lcp = l
                    best_meta = meta
                    best_meta_blocks_total = len(mblocks)

            # 4) Захват целевого слота с исключением отвергнутого hot
            exclude = {rejected_sid} if rejected_sid is not None else set()
            sid, slot_binding = await self.acquire_free_or_cold_slot(exclude=exclude)

            # 5) Restore, если .meta проходит порог
            if best_meta and best_meta_lcp > 0:
                denom = max(1, min(len(req_blocks), best_meta_blocks_total))
                ratio = best_meta_lcp / denom
                if ratio >= SIMILARITY_MIN_RATIO:
                    meta_key = best_meta.get("key")
                    await self.restore_slot_cache(sid, meta_key)
                    binding = SlotBinding(
                        key=meta_key,
                        prefix_text="(from meta)",
                        block_hashes=best_meta.get("blocks") or [],
                        last_use_ts=self._now(),
                        words_per_block=words_per_block,
                        pinned=(meta_key in PINNED_PREFIX_KEYS),
                        hot=True,
                        lock=slot_binding.lock,
                    )
                    self._bindings[sid] = binding
                    if self.log:
                        self.log.info(f"restore-lcp ACCEPT slot_id={sid} lcp_blocks={best_meta_lcp}/{denom} ratio={round(ratio,3)}")
                    return sid, binding, "restore-lcp", best_meta_lcp, len(binding.block_hashes)
                else:
                    if self.log:
                        self.log.info(f"restore-lcp REJECT lcp_blocks={best_meta_lcp}/{denom} ratio={round(ratio,3)} min_ratio={SIMILARITY_MIN_RATIO}")

            # 6) Холодный старт (не портим отвергнутый hot)
            binding = SlotBinding(
                key=req_key,
                prefix_text=req_prefix_text,
                block_hashes=req_blocks,
                last_use_ts=self._now(),
                words_per_block=words_per_block,
                pinned=(req_key in PINNED_PREFIX_KEYS),
                hot=True,
                lock=slot_binding.lock,
            )
            self._bindings[sid] = binding
            if self.log:
                self.log.info(f"slot_bound_cold slot_id={sid} key={req_key}")
            return sid, binding, "cold", 0, 0

    async def save_slot_cache(self, slot_id: int, key: str) -> None:
        """
        Просит сервер сохранить KV‑состояние слота в файл slotcache_{key}.bin в каталоге --slot-save-path (basename). [web:6]
        Пишет локальную .meta у прокси для будущего restore‑поиска. [web:40]
        """
        filename_basename = f"slotcache_{key}.bin"
        try:
            res = await self.llama.save_slot(slot_id, filename_basename)
            if self.log:
                self.log.info(f"slot_saved slot_id={slot_id} key={key} filename={filename_basename} meta={res}")
            b = self._bindings.get(slot_id)
            if b:
                write_meta_for_key_local(key, b.prefix_text, model_id=self.model_id,
                                         words_per_block=b.words_per_block,
                                         block_hashes=b.block_hashes)
        except httpx.HTTPStatusError as e:
            if self.log:
                self.log.warning(f"slot_save_failed slot_id={slot_id} key={key} filename={filename_basename} status={e.response.status_code}")

    async def restore_slot_cache(self, slot_id: int, key: str) -> None:
        """
        Просит сервер восстановить KV‑состояние слота из файла slotcache_{key}.bin (basename) внутри --slot-save-path. [web:6]
        """
        filename_basename = f"slotcache_{key}.bin"
        try:
            res = await self.llama.restore_slot(slot_id, filename_basename)
            if self.log:
                self.log.info(f"slot_restored slot_id={slot_id} key={key} filename={filename_basename} meta={res}")
        except httpx.HTTPStatusError as e:
            if self.log:
                self.log.warning(f"slot_restore_failed slot_id={slot_id} key={key} filename={filename_basename} status={e.response.status_code}")

    async def touch(self, slot_id: int) -> None:
        """
        Обновляет last_use_ts слота — для LRU‑эвикции и диагностики.
        """
        if slot_id in self._bindings:
            self._bindings[slot_id].last_use_ts = time.time()
