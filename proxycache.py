# proxycache.py
# -*- coding: utf-8 -*-

"""
Точка запуска uvicorn.
"""

import uvicorn
from app import app
from config import PORT, LOG_LEVEL

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level=LOG_LEVEL.lower())
