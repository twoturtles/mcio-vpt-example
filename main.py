import tqdm

from minerl_wrapper import minerl_env
from utils import save_frames_to_mp4
from vpt.agent import MineRLAgent, load_vpt


def main():
    vpt: MineRLAgent = load_vpt(
        model_filepath="data/3x.model",
        weights_filepath="data/foundation-model-3x.weights",
        device="cuda",
    )
    vpt.reset()

    with minerl_env(seed=0) as env:
        env.reset(seed=0)
        obs, _, _, _, _ = env.skip_steps(100)  # Wait for rendering...

        frames = []
        for _ in tqdm.trange(1000):
            action = vpt.get_action(obs)
            action = {k: v[0] if k == "camera" else v.item() for k, v in action.items()}

            obs, _, _, _, _ = env.step(action)
            frames.append(obs["pov"])

    save_frames_to_mp4(frames, "episode.mp4")


if __name__ == "__main__":
    main()
