# llama_client.py
# -*- coding: utf-8 -*-
import os
from typing import Dict, AsyncIterator, Tuple, Optional

import httpx

from config import LLAMA_SERVER_URL, REQUEST_TIMEOUT  # базовая конфигурация

class LlamaClient:
    """
    HTTP-клиент к llama.cpp server с поддержкой:
      - /slots (статус)
      - /slots/{id}?action=save|restore (KV-слоты; filename = basename)
      - /v1/chat/completions (stream=false JSON, stream=true SSE)
    Особенности:
      - При наличии _slot_id в body он будет продублирован как slot_id/id_slot на корневом уровне,
        добавлен в options.slot_id/options.id_slot и продублирован в query (?slot_id=...&id_slot=...).
      - Это повышает вероятность, что конкретная серверная сборка действительно закрепит запрос в нужный слот.
    """

    def __init__(self, base_url: str = LLAMA_SERVER_URL):
        limits = httpx.Limits(max_keepalive_connections=20, max_connections=100)
        self._client = httpx.AsyncClient(
            base_url=base_url,
            timeout=httpx.Timeout(REQUEST_TIMEOUT),
            limits=limits,
        )  # корректный способ создать HTTP клиент для FastAPI-приложения

    async def close(self):
        await self._client.aclose()  # корректное закрытие клиента по завершении приложения

    async def slots_status(self) -> Dict:
        r = await self._client.get("/slots")
        r.raise_for_status()
        return r.json()  # llama.cpp server предоставляет статус слотов по /slots

    async def save_slot(self, slot_id: int, filename_basename: str) -> Dict:
        """
        Сохранение KV-слота в файле с именем filename_basename внутри каталога --slot-save-path (на стороне сервера).
        Важно: filename должен быть только basename, без абсолютных путей.
        """
        params = {"action": "save"}
        data = {"filename": os.path.basename(filename_basename)}
        r = await self._client.post(f"/slots/{slot_id}", params=params, json=data)
        r.raise_for_status()
        return r.json() if r.content else {}  # filename = basename для --slot-save-path

    async def restore_slot(self, slot_id: int, filename_basename: str) -> Dict:
        """
        Восстановление KV-слота из файла filename_basename внутри каталога --slot-save-path (на стороне сервера).
        """
        params = {"action": "restore"}
        data = {"filename": os.path.basename(filename_basename)}
        r = await self._client.post(f"/slots/{slot_id}", params=params, json=data)
        r.raise_for_status()
        return r.json() if r.content else {}  # сервер сам читает файл из --slot-save-path

    def _inject_slot(self, body: Dict) -> Tuple[Dict, Optional[int], str]:
        """
        Если в body присутствует _slot_id, продублировать его:
          - в корне: slot_id, id_slot
          - в options: slot_id, id_slot
          - вернуть также query-строку с теми же параметрами для совместимости (?slot_id=...&id_slot=...)
        """
        b = dict(body)
        slot_id = b.pop("_slot_id", None)
        query_suffix = ""
        if slot_id is not None:
            # Дублируем в корне
            b["slot_id"] = slot_id
            b["id_slot"] = slot_id
            # Дублируем в options
            opts = dict(b.get("options") or {})
            opts["slot_id"] = slot_id
            opts["id_slot"] = slot_id
            b["options"] = opts
            # Query-параметры для некоторых ревизий сервера
            query_suffix = f"?slot_id={slot_id}&id_slot={slot_id}"
        return b, slot_id, query_suffix

    async def chat_completions_stream(self, body: Dict) -> AsyncIterator[bytes]:
        """
        stream=true — проксирование SSE чанков как есть (text/event-stream).
        При наличии _slot_id — пробрасываем slot во всех формах (root/options/query).
        """
        b, slot_id, q = self._inject_slot(body)
        # Убедимся, что stream=true не потерялся
        b["stream"] = True
        # cache_prompt должен управляться вызывающей стороной, но если нужно:
        # b.setdefault("cache_prompt", True)

        url = "/v1/chat/completions" + (q if q else "")
        async with self._client.stream("POST", url, json=b) as r:
            r.raise_for_status()
            async for chunk in r.aiter_raw():
                if chunk:
                    yield chunk  # отдаём сырые data: {...}\n\n чанки для совместимости с клиентами OpenAI

    async def chat_completions_json(self, body: Dict) -> Dict:
        """
        stream=false — вернуть цельный JSON-ответ единожды без SSE.
        При наличии _slot_id — пробрасываем slot во всех формах (root/options/query).
        """
        b, slot_id, q = self._inject_slot(body)
        b["stream"] = False
        url = "/v1/chat/completions" + (q if q else "")
        r = await self._client.post(url, json=b)
        r.raise_for_status()
        return r.json()  # обычный JSON-ответ как в OpenAI Chat Completions
