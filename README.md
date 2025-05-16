# Instructions:
1. Install dependencies. For conda users, I've provided an example environment.yml,
which can be used via `conda env create -f environment.yml`.
2. Download the desired `.model` and `.weights` from the [VPT
github](https://github.com/openai/Video-Pre-Training), and place them in a directory
called `data/`. For example:
    - `mkdir data`
    - `wget -P data/
    https://openaipublic.blob.core.windows.net/minecraft-rl/models/foundation-model-3x.model`
    - `wget -P data/
    https://openaipublic.blob.core.windows.net/minecraft-rl/models/bc-house-3x.weights`
3. Run `main.py`. Add the `--headless` flag for running on headless systems.
