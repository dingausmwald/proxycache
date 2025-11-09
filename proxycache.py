# proxycache.py
# -*- coding: utf-8 -*-
# Запуск через: python3 proxycache.py
import os
import uvicorn
from app import app  # FastAPI instance

if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8081"))
    uvicorn.run(app, host=host, port=port, log_level="info")  # [web:107][web:137]
