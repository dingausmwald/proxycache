# app.py
# -*- coding: utf-8 -*-
"""
FastAPI-приложение (OpenAI-совместимый proxy) поверх llama.cpp server.

Особенности:
- /v1/models — список моделей (OpenAI-совместимо).
- /v1/chat/completions — поддержка stream и обычного JSON.
- Порог «малых» запросов: chars/words/blocks; малые идут в cold/свободные слоты без cache_prompt и помечают слот как cold.
- «Большие» используют эвристику (active-exact/active-lcp/restore-lcp/cold).
- Параллелизм: выбор слота сопровождается захватом slot.lock; в finally все пути освобождают lock.
"""

import time
import httpx
from contextlib import asynccontextmanager
from typing import AsyncIterator, Tuple, List, Dict

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse

from config import (
    LLAMA_SERVER_URL,
    SLOTS_COUNT,
    DEFAULT_MODEL_ID,
    DEFAULT_WORDS_PER_BLOCK,
    THRESHOLD_MODE,
    MIN_PREFIX_CHARS,
    MIN_PREFIX_WORDS,
    MIN_PREFIX_BLOCKS,
)
from llama_client import LlamaClient
from slot_manager import SlotManager
from hashing import (
    canonical_chat_prefix,
    block_hashes_from_text,
    prefix_key_sha256,
    words_from_text,
)
import logging

log = logging.getLogger("proxycache")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Инициализация клиентов и менеджера слотов, аккуратное закрытие по завершении.
    """
    app.state.llama_client = LlamaClient(LLAMA_SERVER_URL)
    app.state.slot_manager = SlotManager(SLOTS_COUNT, app.state.llama_client, DEFAULT_MODEL_ID, logger=log)
    log.info(f"lifespan_startup server={LLAMA_SERVER_URL} slots={SLOTS_COUNT}")
    try:
        yield
    finally:
        await app.state.llama_client.close()
        log.info("lifespan_shutdown")

app = FastAPI(
    title="OpenAI-compatible Llama Proxy (word-block heuristic + thresholds + contention-safe)",
    lifespan=lifespan,
)

def build_done_chunk() -> bytes:
    """
    Маркер завершения OpenAI‑совместимого SSE‑потока.
    """
    return b"data: [DONE]\n\n"

def resolve_block_size(request: Request) -> int:
    """
    Определяет размер блока слов для блочного хеширования:
    приоритет: X-Block-Size -> ?block_size -> DEFAULT_WORDS_PER_BLOCK.
    """
    v = request.headers.get("x-block-size") or request.query_params.get("block_size")
    if v:
        try:
            n = int(v)
            if 1 <= n <= 2048:
                return n
        except Exception:
            pass
    return DEFAULT_WORDS_PER_BLOCK

def resolve_threshold_mode(request: Request) -> str:
    """
    Режим порога «малых» запросов: chars | words | blocks.
    """
    v = (request.headers.get("x-threshold-mode") or request.query_params.get("threshold_mode") or THRESHOLD_MODE).lower()
    return v if v in ("chars", "words", "blocks") else "chars"

def resolve_min_prefix_chars(request: Request) -> int:
    v = request.headers.get("x-min-prefix-chars") or request.query_params.get("min_prefix_chars")
    if v:
        try:
            n = int(v)
            if 0 <= n <= 10_000_000:
                return n
        except Exception:
            pass
    return MIN_PREFIX_CHARS

def resolve_min_prefix_words(request: Request) -> int:
    v = request.headers.get("x-min-prefix-words") or request.query_params.get("min_prefix_words")
    if v:
        try:
            n = int(v)
            if 0 <= n <= 10_000_000:
                return n
        except Exception:
            pass
    return MIN_PREFIX_WORDS

def resolve_min_prefix_blocks(request: Request) -> int:
    v = request.headers.get("x-min-prefix-blocks") or request.query_params.get("min_prefix_blocks")
    if v:
        try:
            n = int(v)
            if 0 <= n <= 10_000_000:
                return n
        except Exception:
            pass
    return MIN_PREFIX_BLOCKS

def extract_prefix_stats(openai_body: Dict, words_per_block: int) -> Tuple[str, str, List[str], int, int]:
    """
    Готовит ключ и статистику префикса:
    - key: SHA256 канонического текста.
    - req_blocks: цепочка блочных хешей (для LCP).
    - prefix_len/words_cnt: метрики для порогов «малых» запросов.
    """
    messages = openai_body.get("messages") or []
    prefix_text = canonical_chat_prefix(messages, add_bos=True)
    key = prefix_key_sha256(prefix_text)
    blocks = block_hashes_from_text(prefix_text, words_per_block)
    words_cnt = len(words_from_text(prefix_text))
    return key, prefix_text, blocks, len(prefix_text), words_cnt

@app.get("/v1/models")
async def list_models():
    """
    OpenAI-совместимый список моделей (минимально достаточный для клиентов).
    """
    now = int(time.time())
    data = [{
        "id": DEFAULT_MODEL_ID,
        "object": "model",
        "created": now,
        "owned_by": "local",
    }]
    return {"object": "list", "data": data}

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """
    OpenAI Chat Completions:
    - stream=false: возвращает JSON (JSONResponse).
    - stream=true: SSE‑поток (StreamingResponse).
    - Малые запросы: slot по cold→free hot→ожидание, без cache_prompt, после завершения — mark cold.
    - Большие запросы: ensure_slot_for_request, затем fixed slot_id с cache_prompt=true.
    На время генерации соответствующий слот удерживается через binding.lock.
    """
    body = await request.json()
    stream_req = bool(body.get("stream", False))
    model = body.get("model", DEFAULT_MODEL_ID)
    words_per_block = resolve_block_size(request)
    mode = resolve_threshold_mode(request)
    min_chars = resolve_min_prefix_chars(request)
    min_words = resolve_min_prefix_words(request)
    min_blocks = resolve_min_prefix_blocks(request)

    req_key, req_prefix_text, req_blocks, prefix_len, words_cnt = extract_prefix_stats(body, words_per_block)
    blocks_cnt = len(req_blocks)
    log.info(f"request_received model={model} key={req_key} stream={stream_req} prefix_chars={prefix_len} words={words_cnt} blocks={blocks_cnt} wpb={words_per_block} threshold_mode={mode} min_chars={min_chars} min_words={min_words} min_blocks={min_blocks}")

    slot_manager: SlotManager = request.app.state.slot_manager
    llama_client: LlamaClient = request.app.state.llama_client

    # Условие «малого» запроса
    small = (
        (mode == "chars" and prefix_len < min_chars) or
        (mode == "words" and words_cnt < min_words) or
        (mode == "blocks" and blocks_cnt < min_blocks)
    )

    if small:
        # Малые запросы: захватываем слот через приоритет cold/free, иначе ждём LRU, без cache_prompt
        sid, binding = await slot_manager.acquire_free_or_cold_slot()
        log.info(f"small_request_use_slot slot_id={sid}")

        if stream_req:
            llama_body = dict(body)
            llama_body["stream"] = True
            llama_body["_slot_id"] = sid

            async def sse_iterator_small() -> AsyncIterator[bytes]:
                try:
                    async for raw in llama_client.chat_completions_stream(llama_body):
                        yield raw
                except httpx.ConnectError as e:
                    log.warning(f"backend_connect_error url={LLAMA_SERVER_URL} err={e}")
                    yield build_done_chunk()
                    raise HTTPException(status_code=502, detail="llama backend unavailable")
                except httpx.HTTPError as e:
                    log.warning(f"backend_http_error url={LLAMA_SERVER_URL} err={e}")
                    yield build_done_chunk()
                    raise HTTPException(status_code=502, detail="llama backend error")
                finally:
                    # пометка cold и release lock
                    await slot_manager.mark_slot_cold(sid)
                    log.info(f"small_request_mark_cold slot_id={sid}")
                    try:
                        binding.lock.release()
                    except RuntimeError:
                        pass
                    yield build_done_chunk()

            return StreamingResponse(sse_iterator_small(), media_type="text/event-stream")
        else:
            llama_body = dict(body)
            llama_body["stream"] = False
            llama_body["_slot_id"] = sid
            try:
                resp = await llama_client.chat_completions_json(llama_body)
                await slot_manager.mark_slot_cold(sid)
                log.info(f"small_request_mark_cold slot_id={sid}")
                return JSONResponse(content=resp)
            except httpx.ConnectError as e:
                log.warning(f"backend_connect_error url={LLAMA_SERVER_URL} err={e}")
                raise HTTPException(status_code=502, detail="llama backend unavailable")
            except httpx.HTTPError as e:
                log.warning(f"backend_http_error url={LLAMA_SERVER_URL} err={e}")
                raise HTTPException(status_code=502, detail="llama backend error")
            finally:
                try:
                    binding.lock.release()
                except RuntimeError:
                    pass

    # Большая ветка: ensure_slot_for_request под глобальным контролем, вернётся слот с уже захваченным lock
    slot_id, binding, source, lcp_count, binding_total = await slot_manager.ensure_slot_for_request(
        req_key, req_prefix_text, req_blocks, words_per_block
    )
    log.info(f"match_info source={source} lcp_blocks={lcp_count}/{blocks_cnt} binding_blocks={binding_total}")

    if stream_req:
        llama_body = dict(body)
        llama_body["stream"] = True
        llama_body["cache_prompt"] = True
        llama_body["_slot_id"] = slot_id
        log.info(f"generation_start stream slot_id={slot_id} key={binding.key} source={source} cache_prompt=True")

        async def sse_iterator() -> AsyncIterator[bytes]:
            try:
                async for raw in llama_client.chat_completions_stream(llama_body):
                    yield raw
                    await slot_manager.touch(slot_id)
            except httpx.ConnectError as e:
                log.warning(f"backend_connect_error url={LLAMA_SERVER_URL} err={e}")
                yield build_done_chunk()
                raise HTTPException(status_code=502, detail="llama backend unavailable")
            except httpx.HTTPError as e:
                log.warning(f"backend_http_error url={LLAMA_SERVER_URL} err={e}")
                yield build_done_chunk()
                raise HTTPException(status_code=502, detail="llama backend error")
            finally:
                log.info(f"generation_done slot_id={slot_id} key={binding.key}")
                try:
                    binding.lock.release()
                except RuntimeError:
                    pass
                yield build_done_chunk()

        return StreamingResponse(sse_iterator(), media_type="text/event-stream")
    else:
        llama_body = dict(body)
        llama_body["stream"] = False
        llama_body["cache_prompt"] = True
        llama_body["_slot_id"] = slot_id
        log.info(f"generation_start json slot_id={slot_id} key={binding.key} source={source} cache_prompt=True")
        try:
            resp = await llama_client.chat_completions_json(llama_body)
            return JSONResponse(content=resp)
        except httpx.ConnectError as e:
            log.warning(f"backend_connect_error url={LLAMA_SERVER_URL} err={e}")
            raise HTTPException(status_code=502, detail="llama backend unavailable")
        except httpx.HTTPError as e:
            log.warning(f"backend_http_error url={LLAMA_SERVER_URL} err={e}")
            raise HTTPException(status_code=502, detail="llama backend error")
        finally:
            try:
                binding.lock.release()
            except RuntimeError:
                pass
