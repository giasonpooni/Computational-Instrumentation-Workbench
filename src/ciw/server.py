"""Local WebSocket transport with an explicit container bind and saved shutdown."""

from __future__ import annotations

import asyncio
import json
import logging
import signal

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


async def run_server(session: Session, port: int = 8765, bind: str = "127.0.0.1",
                     *, stop_event: asyncio.Event | None = None) -> None:
    """Drain connections and persist the workspace when the process is stopped.

    SIGINT/SIGTERM stop this process only. There is no remote shutdown operation.
    An embedding application may supply its own stop event instead of installing
    process signal handlers. SIGTERM handling is exercised on POSIX; Windows
    process termination is immediate and must be preceded by workspace.save.
    """
    if bind not in {"127.0.0.1", "0.0.0.0"}:
        raise ValueError("bind must be 127.0.0.1 or explicitly 0.0.0.0 for a container")
    bridge = WorkbenchServer(session)
    loop = asyncio.get_running_loop()
    managed_signals = stop_event is None
    stopped = stop_event if stop_event is not None else asyncio.Event()
    installed = []
    started = False
    try:
        if managed_signals:
            for signum in (signal.SIGINT, signal.SIGTERM):
                previous = signal.getsignal(signum)
                try:
                    loop.add_signal_handler(signum, stopped.set)
                    installed.append((signum, previous, True))
                except (NotImplementedError, RuntimeError):
                    # Windows event loops do not expose add_signal_handler.
                    try:
                        signal.signal(signum, lambda *_: loop.call_soon_threadsafe(stopped.set))
                        installed.append((signum, previous, False))
                    except ValueError:
                        LOG.warning("Signal handlers require the main thread; supply stop_event when embedding")
        # Native clients only; Compose publishes the explicit container bind
        # through a host-loopback port. Browser-origin connections stay rejected.
        async with serve(bridge.handler, bind, port, origins=[None],
                         max_size=1_048_576, max_queue=16, close_timeout=2):
            started = True
            print(f"Computational Instrumentation Workbench: ws://{bind}:{port}", flush=True)
            print(f"Session {session.session_id} | {session.run['run_id']} | Ctrl+C to stop", flush=True)
            await stopped.wait()
        # Exiting serve closes the listener and waits for handlers, including
        # in-flight scientific operations, before snapshotting their results.
    finally:
        try:
            if started:
                workspace = session.output_dir / "workspace.json"
                try:
                    await asyncio.to_thread(session.save_workspace, workspace)
                except Exception:
                    LOG.exception("Unable to save workspace during shutdown: %s", workspace)
                    raise
                print(f"Saved workspace: {workspace}", flush=True)
        finally:
            for signum, previous, via_loop in installed:
                if via_loop:
                    loop.remove_signal_handler(signum)
                signal.signal(signum, previous)
