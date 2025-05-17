import tqdm

from minerl_wrapper import minerl_env
from utils import save_frames_to_mp4
from vpt.agent import MineRLAgent, load_vpt


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--headless")
    parser.add_argument("--model", default="data/3x.model")
    parser.add_argument("--weights", default="data/bc-house-3x.weights")
    args = parser.parse_args()

    vpt: MineRLAgent = load_vpt(
        model_filepath=args.model, weights_filepath=args.weights, device="cuda"
    )
    vpt.reset()

    with minerl_env(seed=1, headless=args.headless) as env:
        env.reset(seed=1)
        obs, _, _, _, _ = env.skip_steps(100)  # Wait for rendering...

        frames = []
        for t in tqdm.trange(600):
            action = vpt.get_action(obs)
            action = {k: v[0] if k == "camera" else v.item() for k, v in action.items()}
            action["ESC"] = 0

            obs, _, _, _, _ = env.step(action)
            frames.append(obs["pov"])

    save_frames_to_mp4(frames, "episode.mp4")


if __name__ == "__main__":
    main()
