## Instructions for VPT:
1. Install dependencies. An example `environment.yml` for conda users and an example
`requirements.txt` for pip users are provided.
2. Download the desired `.model` and `.weights` from the [VPT
github](https://github.com/openai/Video-Pre-Training), and place them in a directory
called `data/`. For example:
    - `mkdir data`
    - `wget -O data/3x.model
    https://openaipublic.blob.core.windows.net/minecraft-rl/models/foundation-model-3x.model`
    - `wget -P data/
    https://openaipublic.blob.core.windows.net/minecraft-rl/models/bc-house-3x.weights`
3. Run `run_vpt.py`. Add the `--headless` flag for running on headless systems.

## Instructions for STEVE-1:
1. Install dependencies (see above).
2. Download VPT's 2x.model, MineCLIP weights, STEVE-1 prior weights, and STEVE-1 policy
weights:
    - `wget -O data/2x.model
    https://openaipublic.blob.core.windows.net/minecraft-rl/models/2x.model`
    - `gdown https://drive.google.com/uc?id=1uaZM1ZLBz2dZWcn85rZmjP7LV6Sg5PZW -O
    data/mineclip_attn.pth`
    - `gdown https://drive.google.com/uc?id=1OdX5wiybK8jALVfP5_dEo0CWm9BQbDES -O
    data/steve1_prior.pt`
    - `gdown https://drive.google.com/uc?id=1E3fd_-H1rRZqMkUKHfiMhx-ppLLehQPI -O
    data/steve1.weights `
3. Run `run_steve1.py {prompt}` with the desired prompt
    - E.g. for the logs task, the prompt 'chop down the tree, gather wood, pick up wood,
    chop it down, break tree' was used in the paper.
    - Add the `--headless` flag for running on headless systems.
    - Add `--guidance {guidance}` to set the classifier-free guidance scale (default 0).
