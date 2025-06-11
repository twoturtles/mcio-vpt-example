import hashlib
import pprint
import threading
from queue import SimpleQueue

import mcio_ctrl as mc
import torch
import torch.nn as nn
import tqdm
from mineclip import MineCLIP

from mcio_agent.agent import MineRLAgent
from mcio_agent.lib import torch_util
from minerl_wrapper import minerl_env

MINECLIP_CONFIG = {
    "arch": "vit_base_p16_fz.v2.t2",
    "hidden_dim": 512,
    "image_feature_dim": 512,
    "mlp_adapter_spec": "v0-2.t0",
    "pool_type": "attn.d2.nh8.glusw",
    "resolution": [160, 256],
    "ckpt": {
        "path": "data/mineclip_attn.pth",
        "checksum": "b5ece9198337cfd117a3bfbd921e56da",
    },
}


prompt_queue: SimpleQueue[str] = SimpleQueue()


def input_thread() -> None:
    while True:
        new_prompt = input("\n\nEnter new prompt: \n")
        prompt_queue.put(new_prompt)


def load_mineclip() -> None:
    cfg = MINECLIP_CONFIG.copy()
    ckpt = cfg.pop("ckpt")
    assert (
        hashlib.md5(open(ckpt["path"], "rb").read()).hexdigest() == ckpt["checksum"]
    ), "broken ckpt"

    model = MineCLIP(**cfg)
    ckpt = torch.load(ckpt["path"], weights_only=True, map_location="cpu")
    state_dict = {k.removeprefix("model."): v for k, v in ckpt["state_dict"].items()}
    model.load_state_dict(state_dict)
    return model


class TranslatorVAE(torch.nn.Module):
    def __init__(self, input_dim=512, hidden_dim=512, latent_dim=512):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_dim * 2, hidden_dim),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 2 * latent_dim),
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(latent_dim + input_dim, hidden_dim),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, input_dim),
        )

    def encode(self, visual_embeddings, text_embeddings):
        """Encode the given visual and text embeddings into a latent vector."""
        # Concatenate the visual and text embeddings.
        x = torch.cat([visual_embeddings, text_embeddings], dim=1)
        # Encode the concatenated embeddings into a latent vector.
        return self.encoder(x)

    def sample(self, mu, logvar):
        """Sample a latent vector from the given mu and logvar."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, latent_vector, text_embeddings):
        """Decode the given latent vector and text embeddings into a visual embedding."""
        # Concatenate the latent vector and text embeddings.
        x = torch.cat([latent_vector, text_embeddings], dim=1)
        # Decode the concatenated embeddings into a visual embedding.
        return self.decoder(x)

    def forward(self, text_embeddings, deterministic=False):
        """Encode the given text embeddings into a latent vector and then decode it into a visual embedding."""
        # Use the prior as the mean and logvar.
        mu = torch.zeros(text_embeddings.shape[0], self.latent_dim).to(
            text_embeddings.device
        )
        logvar = torch.zeros(text_embeddings.shape[0], self.latent_dim).to(
            text_embeddings.device
        )

        # Sample a latent vector from the mu and logvar.
        if deterministic:
            latent_vector = mu
        else:
            latent_vector = self.sample(mu, logvar)

        # Decode the latent vector into a visual embedding.
        pred_visual_embeddings = self.decode(latent_vector, text_embeddings)

        return pred_visual_embeddings


def load_vae(device="cpu", weights_path="data/steve1_prior.pt") -> TranslatorVAE:
    vae = TranslatorVAE().to(device)
    vae.load_state_dict(
        torch.load(weights_path, weights_only=True, map_location=device)
    )
    return vae


class STEVE1PromptEncoder(nn.Module):
    def __init__(self, policy_ckpt):
        super().__init__()
        self.mineclip = load_mineclip()
        self.prior = load_vae()
        self.proj_weight, self.proj_bias = (
            policy_ckpt.pop("net.mineclip_embed_linear.weight"),
            policy_ckpt.pop("net.mineclip_embed_linear.bias"),
        )

    @torch.inference_mode
    def embed_prompt(self, prompt: str):
        emb = self.mineclip.encode_text(prompt)
        emb = self.prior(emb.float())
        emb = emb @ self.proj_weight.type_as(emb).t() + self.proj_bias.type_as(emb)
        return emb


# TODO: support for guidance.


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("prompt")
    parser.add_argument("--guidance", type=float, default=0.0)
    parser.add_argument("--headless", action="store_true")
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

    gui = args.gui if not args.headless else False

    device = torch_util.default_device_type()
    policy_ckpt = torch.load(
        "data/steve1.weights", weights_only=True, map_location="cpu"
    )
    text_encoder = STEVE1PromptEncoder(policy_ckpt).to(device)

    policy_kwargs = {
        "attention_heads": 16,  # 24 for 3x.model
        "attention_mask_style": "clipped_causal",
        "attention_memory_size": 256,
        "diff_mlp_embedding": False,
        "hidsize": 2048,  # 3072 for 3x.model
        "img_shape": [128, 128, 3],
        "impala_chans": [16, 32, 32],
        "impala_kwargs": {"post_pool_groups": 1},
        "impala_width": 8,  # 12 for 3x.model
        "init_norm_kwargs": {"batch_norm": False, "group_norm_groups": 1},
        "n_recurrence_layers": 4,
        "only_img_input": True,
        "pointwise_ratio": 4,
        "pointwise_use_activation": False,
        "recurrence_is_residual": True,
        "recurrence_type": "transformer",
        "timesteps": 128,
        "use_pointwise_layer": True,
        "use_pre_lstm_ln": False,
    }
    pi_head_kwargs = {"temperature": 2.0}
    agent = MineRLAgent(
        policy_kwargs=policy_kwargs,
        pi_head_kwargs=pi_head_kwargs,
        guidance=args.guidance,
    )
    agent.policy.load_state_dict(policy_ckpt)
    agent.reset()

    prompt_queue.put(args.prompt)
    threading.Thread(target=input_thread, daemon=True).start()

    with minerl_env(
        seed=1, headless=args.headless, gui=gui, connect=args.connect
    ) as env:
        env.reset(seed=1)
        obs, _, _, _, _ = env.skip_steps(100)  # Wait for rendering...

        if args.video:
            vid = mc.util.VideoWriter()
        for t in tqdm.trange(args.steps):
            if not prompt_queue.empty():
                prompt = prompt_queue.get()
                with torch.amp.autocast(device):
                    cond_embed = text_encoder.embed_prompt(prompt).cpu().numpy()
                print(f"running with prompt={prompt!r}")

            obs["cond_embed"] = cond_embed
            action = agent.get_action(obs)
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
