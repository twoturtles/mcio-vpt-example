import contextlib
from pathlib import Path
from typing import Iterator

from mcio_ctrl.envs.minerl_env import MinerlEnv
from mcio_ctrl.instance import Installer, InstanceManager
from mcio_ctrl.types import MCioMode, RunOptions
from mcio_ctrl.world import STORAGE_LOCATION, WorldManager

MCIO_DIR = "./mcio"
INSTANCE_NAME = "main"


@contextlib.contextmanager
def minerl_env(seed, headless: bool) -> Iterator[MinerlEnv]:
    mcio_dir = Path(MCIO_DIR).absolute()
    mcio_dir.mkdir(exist_ok=True)
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
    if headless:
        from xvfbfixed import XvfbFixed

        context = XvfbFixed
    else:
        context = contextlib.nullcontext
    with context():
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
