# backend/routers/ws_connection.py
"""
WebSocket connection registry — tracks active connections per session.
"""

from fastapi import WebSocket


class ConnectionManager:
    """Maps session_id → active WebSocket connection."""

    def __init__(self):
        self._connections: dict[str, WebSocket] = {}

    async def connect(self, session_id: str, websocket: WebSocket) -> None:
        await websocket.accept()
        self._connections[session_id] = websocket
        print(f"[WS] Client connected: {session_id}")

    def disconnect(self, session_id: str) -> None:
        self._connections.pop(session_id, None)
        print(f"[WS] Client disconnected: {session_id}")

    async def send_json(self, session_id: str, data: dict) -> None:
        ws = self._connections.get(session_id)
        if ws:
            await ws.send_json(data)

    async def send_bytes(self, session_id: str, data: bytes) -> None:
        ws = self._connections.get(session_id)
        if ws:
            await ws.send_bytes(data)


# Shared singleton — import this wherever you need to send messages
manager = ConnectionManager()
