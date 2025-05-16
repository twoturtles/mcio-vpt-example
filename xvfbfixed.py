import fcntl
import os
import random

from xvfbwrapper import Xvfb


class XvfbFixed(Xvfb):
    def _get_next_unused_display(self):
        filepath = os.path.join(self._tempdir, ".X{0}-lock")
        # RNG now depends on PID
        rng = random.Random(hash((os.getpid(), random.random())))

        while True:
            display = rng.randint(1, self.__class__.MAX_DISPLAY)

            if os.path.exists(filepath.format(display)):
                continue
            self._lock_display_file = open(filepath.format(display), "w")
            try:
                fcntl.flock(self._lock_display_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                continue

            return display
