"""A background thread for rendering, so the Tk thread never waits for streamlines or heatmaps."""
import queue
import threading

import kernels


class Worker:
    """Runs jobs (functions without arguments) one at a time on a daemon thread. Only the newest waiting job
    runs: a submit replaces one that has not started yet. Results land in `results`, which the Tk thread
    polls; Tk itself is never touched from here."""

    def __init__(self):
        self.results = queue.SimpleQueue()
        self._job = None
        self._cv = threading.Condition()
        threading.Thread(target=self._loop, name="render", daemon=True).start()

    def submit(self, job):
        with self._cv:
            self._job = job
            self._cv.notify()

    def _loop(self):
        kernels.warm_up()   # compiles on the very first run (seconds), later loads from the disk cache
        while True:
            with self._cv:
                while self._job is None:
                    self._cv.wait()
                job, self._job = self._job, None
            try:
                result = job()
            except Exception as exc:   # handed to the Tk thread, so a failed job does not end the thread
                result = exc
            self.results.put(result)
