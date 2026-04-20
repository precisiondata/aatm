import time
from typing import Optional


class Timer:
    def __init__(self, description: Optional[str] = None):
        self.description = description

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end = time.perf_counter()
        self.elapsed = self.end - self.start
        print(
            f"{self.description if self.description else f'Timer {id(self)}'} - Executed in {self.elapsed:.4f} seconds"
        )
