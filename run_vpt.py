import pprint

import mcio_ctrl as mc
import tqdm

from mcio_agent.agent import MineRLAgent, load_vpt
from minerl_wrapper import minerl_env


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--model", default="3x.model")
    parser.add_argument("--weights", default="data/bc-house-3x.weights")
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument(
        "--gui",
        "-G",
        action="store_true",
        default=False,
        help="Render to screen",
    )
    parser.add_argument("--video", "-V", default=None, help="Save video to file")
    parser.add_argument(
        "--connect", "-c", action="store_true", help="Connect to running instance"
    )

    args = parser.parse_args()
    assert args.model in ("2x.model", "3x.model")

    gui = args.gui if not args.headless else False

    vpt: MineRLAgent = load_vpt(model=args.model, weights_filepath=args.weights)
    vpt.reset()

    with minerl_env(
        seed=1, headless=args.headless, gui=gui, connect=args.connect
    ) as env:
        obs = env.reset(seed=1)[0]

        if args.video:
            vid = mc.util.VideoWriter()
        for t in tqdm.trange(args.steps):
            action = vpt.get_action(obs)
            action = {k: v[0] if k == "camera" else v.item() for k, v in action.items()}
            action["ESC"] = 0

            obs, _, _, _, _ = env.step(action)
            if args.gui:
                env.render()
            if args.video:
                vid.add(obs["pov"])

    if args.video:
        vid.write(args.video)
    pprint.pprint(env.stats_cache)


if __name__ == "__main__":
    main()
