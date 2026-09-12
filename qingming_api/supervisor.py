"""Replace only our native worker, never replay speech or restart other services."""
import asyncio
import logging

LOG = logging.getLogger("qingming.supervisor")


class WorkerSupervisor:
    def __init__(self, worker, lock, *, poll_interval=0.25, initial_delay=1,
                 max_delay=30, stable_seconds=60):
        self.worker = worker
        self.lock = lock
        self.poll_interval = poll_interval
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.stable_seconds = stable_seconds
        self.starts = 0

    async def run(self):
        delay = self.initial_delay
        started_at = None
        clock = asyncio.get_running_loop().time
        try:
            while True:
                if self.worker.ready:
                    await asyncio.sleep(self.poll_interval)
                    continue
                if self.starts:
                    if started_at is not None and clock() - started_at >= self.stable_seconds:
                        delay = self.initial_delay
                    LOG.warning("Native worker unavailable; recovery in %.1fs", delay)
                    await asyncio.sleep(delay)
                    delay = min(self.max_delay, delay * 2)
                # The request owns this lock until its native/encoder cleanup ends.
                # No new process may be started while old audio is still unwinding.
                async with self.lock:
                    await self.worker.close()
                    self.starts += 1
                    started_at = None
                    try:
                        await self.worker.start()
                    except Exception:
                        self.worker.ready = False
                        LOG.exception("Native startup failed; will retry")
                    else:
                        started_at = clock()
        finally:
            await self.worker.close()
