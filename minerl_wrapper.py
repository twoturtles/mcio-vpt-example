import contextlib
import fcntl
import os
import random
from pathlib import Path
from typing import Iterator

from mcio_ctrl.envs.minerl_env import MinerlEnv
from mcio_ctrl.instance import Installer, InstanceManager
from mcio_ctrl.types import MCioMode, RunOptions
from mcio_ctrl.world import STORAGE_LOCATION, WorldManager
from xvfbwrapper import Xvfb

MCIO_DIR = "./mcio"
INSTANCE_NAME = "main"


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


@contextlib.contextmanager
def minerl_env(seed) -> Iterator[MinerlEnv]:
    mcio_dir = Path(MCIO_DIR).absolute()
    world_name = f"world_{seed}"

    im = InstanceManager(mcio_dir)
    if not im.instance_exists(INSTANCE_NAME):
        installer = Installer(INSTANCE_NAME, mcio_dir)
        installer.install()

        im.install_mod(INSTANCE_NAME, "sodium")

    wm = WorldManager(mcio_dir)
    if not wm.world_exists(STORAGE_LOCATION, world_name):
        wm.create(world_name, seed=seed)
    if wm.world_exists(INSTANCE_NAME, world_name):
        wm.delete(INSTANCE_NAME, world_name)
    wm.copy(STORAGE_LOCATION, world_name, INSTANCE_NAME)

    env = None
    with XvfbFixed():
        try:
            opts = RunOptions(
                mcio_dir=mcio_dir,
                instance_name=INSTANCE_NAME,
                world_name=world_name,
                hide_window=True,
                mcio_mode=MCioMode.SYNC,
                width=640,
                height=360,
                mc_username="MCioAgent",
            )
            env = MinerlEnv(opts, render_mode="rgb_array")
            yield env

        finally:
            if env is not None:
                env.close()
