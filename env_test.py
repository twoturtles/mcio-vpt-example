import copy
import fcntl
import logging
import os
import random
from pathlib import Path

import mcio_ctrl
import numpy as np
from mcio_ctrl.envs.minerl_env import MinerlEnv
from PIL import Image
from xvfbwrapper import Xvfb

VERSION = "1.21.3"
MCIO_DIR = "./mcio"
INSTANCE_NAME = f"INSTANCE_{VERSION.replace('.', '_')}"
WORLD_NAME = f"WORLD_{VERSION.replace('.', '_')}"


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


NULL_ACTION = {
    "ESC": 0,
    "attack": 0,
    "use": 0,
    "pickItem": 0,
    "forward": 0,
    "left": 0,
    "right": 0,
    "back": 0,
    "drop": 0,
    "inventory": 0,
    "jump": 0,
    "sneak": 0,
    "sprint": 0,
    "swapHands": 0,
    "hotbar.1": 0,
    "hotbar.2": 0,
    "hotbar.3": 0,
    "hotbar.4": 0,
    "hotbar.5": 0,
    "hotbar.6": 0,
    "hotbar.7": 0,
    "hotbar.8": 0,
    "hotbar.9": 0,
    "camera": np.zeros(2),
}


def save_frames_to_gif(frames: list[np.ndarray], filepath, fps: int = 20):
    imgs = [Image.fromarray(frame) for frame in frames]
    imgs[0].save(
        filepath,
        save_all=True,
        append_images=imgs[1:],
        duration=1000 // fps,
        loop=0,
    )


def main():
    logging.basicConfig(level=logging.WARNING)

    mcio_dir = Path(MCIO_DIR).absolute()

    im = mcio_ctrl.instance.InstanceManager(mcio_dir)
    if not im.instance_exists(INSTANCE_NAME):
        print("Installing Minecraft...")

        installer = mcio_ctrl.instance.Installer(
            instance_name=INSTANCE_NAME, mcio_dir=mcio_dir, mc_version=VERSION
        )
        installer.install()

    wm = mcio_ctrl.world.WorldManager(mcio_dir)
    if not wm.world_exists(mcio_ctrl.world.STORAGE_LOCATION, WORLD_NAME):
        print("Creating world...")
        wm.create(WORLD_NAME, seed=0)

    if wm.world_exists(INSTANCE_NAME, WORLD_NAME):
        wm.delete(INSTANCE_NAME, WORLD_NAME)
    print("Copying world to instance...")
    wm.copy(mcio_ctrl.world.STORAGE_LOCATION, WORLD_NAME, INSTANCE_NAME)

    env = None
    try:
        opts = mcio_ctrl.types.RunOptions(
            mcio_dir=mcio_dir,
            instance_name=INSTANCE_NAME,
            world_name=WORLD_NAME,
            hide_window=True,
            mcio_mode=mcio_ctrl.types.MCioMode.SYNC,
            width=640,
            height=360,
            mc_username="JKG2LHOkfjdg7",
        )

        frames = []
        with XvfbFixed():
            print("Launching...")
            env = MinerlEnv(opts, render_mode="rgb_array")
            obs, _ = env.reset(seed=0)
            frames.append(obs["pov"])
            env.skip_steps(10)

            for t in range(5):
                action = copy.deepcopy(NULL_ACTION)
                action["camera"] = np.array([3.0, 1.0])
                obs, _, _, _, _ = env.step(action)
                frames.append(obs["pov"])

            for t in range(10):
                action = copy.deepcopy(NULL_ACTION)
                action["camera"] = np.array([-1.0, 2.0])
                obs, _, _, _, _ = env.step(action)
                frames.append(obs["pov"])

            action = copy.deepcopy(NULL_ACTION)
            action["inventory"] = 1
            obs, _, _, _, _ = env.step(action)
            frames.append(obs["pov"])

            for t in range(5):
                action = copy.deepcopy(NULL_ACTION)
                action["camera"] = np.array([3.0, 1.0])
                obs, _, _, _, _ = env.step(action)
                frames.append(obs["pov"])

            for t in range(10):
                action = copy.deepcopy(NULL_ACTION)
                action["camera"] = np.array([-1.0, 2.0])
                obs, _, _, _, _ = env.step(action)
                frames.append(obs["pov"])

        # for t, frame in enumerate(frames):
        #     Image.fromarray(frame).save(f"frames/frame_{t}.png")
        save_frames_to_gif(frames, "frames.gif", fps=5)

    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()
