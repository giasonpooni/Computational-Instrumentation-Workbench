"""Loopback-only WebSocket transport for the shared instrument session."""

from __future__ import annotations

import asyncio
import json
import logging

from websockets.asyncio.server import serve
from websockets.exceptions import ConnectionClosed

from .session import Session, _reject_constant, envelope

LOG = logging.getLogger(__name__)


class WorkbenchServer:
    def __init__(self, session: Session):
        self.session = session
        self.clients = set()
        self._selection_lock = asyncio.Lock()

    async def _send(self, websocket, message: dict) -> None:
        await websocket.send(json.dumps(message, allow_nan=False))

    async def _broadcast(self, message: dict) -> None:
        async def deliver(client):
            try:
                await asyncio.wait_for(self._send(client, message), timeout=2)
            except (ConnectionClosed, TimeoutError):
                self.clients.discard(client)
                await client.close()
        await asyncio.gather(*(deliver(client) for client in tuple(self.clients)))

    async def handler(self, websocket) -> None:
        self.clients.add(websocket)
        try:
            await self._send(websocket, envelope("session.snapshot", self.session.snapshot()))
            async for raw in websocket:
                try:
                    if not isinstance(raw, str):
                        raise ValueError("Protocol v1 accepts text JSON frames only")
                    request = json.loads(raw, parse_constant=_reject_constant)
                except (ValueError, RecursionError) as exc:
                    await self._send(websocket, envelope("error", {"code": "invalid_request", "message": str(exc)}))
                    continue
                if isinstance(request, dict) and request.get("type") == "selection.update":
                    # Preserve broadcast order across concurrent clients.
                    async with self._selection_lock:
                        response = self.session.handle(request)
                        try:
                            await self._send(websocket, response)
                        except ConnectionClosed:
                            self.clients.discard(websocket)
                        if response["type"] == "response":
                            await self._broadcast(envelope("selection.changed", response["payload"]))
                else:
                    # Numerical operations and disk IO do not block socket polling.
                    response = await asyncio.to_thread(self.session.handle, request)
                    await self._send(websocket, response)
        except ConnectionClosed:
            pass
        except Exception:
            LOG.exception("Client handler failed")
            await websocket.close(code=1011, reason="Internal service error")
        finally:
            self.clients.discard(websocket)


async def run_server(session: Session, port: int = 8765) -> None:
    bridge = WorkbenchServer(session)
    # Native local clients only. Browser origins and non-loopback binds are excluded.
    async with serve(bridge.handler, "127.0.0.1", port, origins=[None],
                     max_size=1_048_576, max_queue=16, close_timeout=2):
        print(f"Computational Instrumentation Workbench: ws://127.0.0.1:{port}", flush=True)
        print(f"Session {session.session_id} | {session.run['run_id']} | Ctrl+C to stop", flush=True)
        await asyncio.Future()
