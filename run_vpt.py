import mcio_ctrl as mc
import tqdm

from minerl_wrapper import minerl_env
from vpt.agent import MineRLAgent, load_vpt


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--model", default="3x.model")
    parser.add_argument("--weights", default="data/bc-house-3x.weights")
    parser.add_argument("--steps", type=int, default=600)
    args = parser.parse_args()
    assert args.model in ("2x.model", "3x.model")

    vpt: MineRLAgent = load_vpt(model=args.model, weights_filepath=args.weights)
    vpt.reset()

    with minerl_env(seed=1, headless=args.headless) as env:
        obs = env.reset(seed=1)[0]

        frames = []
        for t in tqdm.trange(args.steps):
            action = vpt.get_action(obs)
            action = {k: v[0] if k == "camera" else v.item() for k, v in action.items()}
            action["ESC"] = 0

            obs, _, _, _, _ = env.step(action)
            frames.append(obs["pov"])

    mc.util.VideoWriter(frames).write("episode.mp4")


if __name__ == "__main__":
    main()
