# Flux Front End

A local, self-hosted [Gradio](https://www.gradio.app/) web UI for generating images with the
[FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev) diffusion model, entirely on your own
hardware. It supports plain text-to-image generation, plus two ways of guiding a generation with a
reference image: Redux-based style transfer and classic img2img.

Everything runs locally against model weights you already have on disk — nothing is uploaded to a
third-party API, and no API key is required to generate images.

---

## Table of contents

- [How it works](#how-it-works)
- [Requirements](#requirements)
- [Installation](#installation)
- [Configuration (.env)](#configuration-env)
- [Running the app](#running-the-app)
- [Using the UI](#using-the-ui)
- [Project structure](#project-structure)
- [File-by-file reference](#file-by-file-reference)
- [Logging](#logging)
- [Troubleshooting](#troubleshooting)

---

## How it works

This project is a thin browser UI (built with Gradio) wrapped around the Hugging Face
[`diffusers`](https://github.com/huggingface/diffusers) library's Flux pipelines. When you click
"Generate," the UI collects your settings and calls into `flux_pipeline.py`, which:

1. Loads (and caches in memory) the Flux pipeline from your local model folder.
2. Optionally loads the Flux Redux "prior" pipeline, used for style-reference generation.
3. Runs text-to-image, img2img, or style-referenced generation depending on what you provided.
4. Returns the resulting image and the seed that was used, which the UI then saves to disk and
   displays.

Because loading multi-gigabyte model weights from disk is slow, the pipeline is cached in memory
after the first generation and only reloaded if you change the model path, precision, or max token
settings.

## Requirements

- **Python 3.10+** (developed against 3.12).
- **An NVIDIA GPU with CUDA** is strongly recommended. Flux is a large model; running on CPU is
  possible but extremely slow. The default configuration uses fp16 precision plus sequential CPU
  offload, which is tuned to fit the full pipeline on a single 24GB-class GPU.
- **Local Flux model weights**, already downloaded in `diffusers` format:
  - [FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev) — required.
  - [FLUX.1-Redux-dev](https://huggingface.co/black-forest-labs/FLUX.1-Redux-dev) — optional, only
    needed if you want to use the "Style (Redux)" reference mode.

  This app does **not** download models for you (it loads with `local_files_only=True`). You need
  to obtain the weights yourself (e.g. via `huggingface-cli download` or `git lfs`) and point the
  app at the folder they live in.

## Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/sgtidwellgit/FluxGradio.git
   cd FluxGradio
   ```

2. Create and activate a virtual environment (recommended):

   ```bash
   python -m venv venv
   venv\Scripts\activate      # Windows
   source venv/bin/activate   # macOS/Linux
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

   Note: `torch` as listed in `requirements.txt` will install a CPU-only build by default on most
   platforms. For GPU acceleration, install the CUDA-enabled build of PyTorch that matches your
   driver/CUDA version **before** running `pip install -r requirements.txt` — see
   [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/) for the correct
   command for your system.

## Configuration (.env)

The app reads its model paths and default output directory from environment variables rather than
hardcoding them, since these paths are specific to each machine.

1. Copy the example file:

   ```bash
   copy .env.example .env      # Windows
   cp .env.example .env        # macOS/Linux
   ```

2. Edit `.env` and fill in the paths for your machine:

   | Variable                | Description                                                                 |
   |--------------------------|------------------------------------------------------------------------------|
   | `FLUX_MODEL_PATH`        | Folder containing the FLUX.1-dev weights (diffusers format).                |
   | `FLUX_REDUX_MODEL_PATH`  | Folder containing the FLUX.1-Redux-dev weights. Only used in Style mode.    |
   | `FLUX_OUTPUT_DIR`        | Folder where generated images are saved.                                    |

`.env` is listed in `.gitignore` and will never be committed — it's meant to hold your personal,
machine-specific paths. `.env.example` is the template that ships in the repo so anyone who clones
it knows what variables to set.

These values only populate the *default* values shown in the "Model / output settings" section of
the UI — you can still override the model path, Redux path, or output directory per-session directly
in the browser without editing `.env`.

## Running the app

```bash
python app.py
```

Then open **http://127.0.0.1:7860** in your browser.

## Using the UI

- **Positive prompt** — describe the image you want.
- **Negative prompt** — describe what to avoid. Only has an effect when **True CFG scale** is above
  1.0 — Flux-dev is "guidance-distilled," so at the default of 1.0 the negative prompt is accepted
  but ignored (running it for real doubles the compute cost per step).
- **Width / Height** — output resolution, must be a multiple of 16 (the app rounds down for you).
  Flux was trained around ~1024–2048px; pushing much higher tends to produce duplicated/tiled
  artifacts.
- **Steps** — number of denoising steps. More steps generally means more detail/coherence at the
  cost of generation time.
- **Guidance scale (CFG)** — how strongly the model follows your prompt.
- **True CFG scale** — set above 1.0 to actually enable negative-prompt guidance (see above).
- **Seed** — set to `-1` for a random seed each run, or a specific integer to reproduce a result.
- **Max T5 tokens** — token budget for the T5 text encoder (hard-capped at 512 by Flux itself).
- **Use fp16** — half-precision inference; requires CUDA. Leave this on unless you're running on
  CPU.
- **Reference image (optional)** — upload an image to guide the generation, then choose a mode:
  - **Style (Redux)** — generation starts from noise and follows your *prompt's* subject and
    composition, but borrows the reference image's visual style/mood via the Flux Redux prior.
    Requires `FLUX_REDUX_MODEL_PATH` to be set to a valid Redux model folder.
  - **Img2img** — generation starts from the reference image itself and lightly modifies it, so the
    output stays anchored to the reference's composition regardless of the prompt.
  - **Reference Strength** — how strongly the reference pulls the result (see the tooltip in the UI
    for the precise behavior in each mode).
- **Model / output settings** (collapsed accordion) — lets you override the model path, Redux model
  path, and output directory for this session without touching `.env`.

Generated images are saved to your configured output directory as
`{timestamp}_{mode}_seed{seed}.png`, and the seed/save path are shown in the **Info** box. Any
errors (e.g. out-of-memory, bad model path) are shown in the **Error log** box and written to
`logs/app.log`.

## Project structure

```
FluxFrontEnd/
├── app.py              # Gradio UI: layout, inputs/outputs, wires the UI to flux_pipeline.generate()
├── flux_pipeline.py    # Model loading/caching and the actual txt2img/img2img/style generation logic
├── requirements.txt    # Python dependencies
├── .env.example         # Template for machine-specific configuration (safe to commit)
├── .env                  # Your actual machine-specific configuration (gitignored, not committed)
├── .gitignore
├── logs/
│   └── app.log          # Rolling log file written by the app (gitignored)
└── README.md
```

## File-by-file reference

### `app.py`

The Gradio front end. Responsibilities:

- Loads configuration from `.env` via `python-dotenv` (model paths, output directory).
- Sets up logging to both the console and `logs/app.log`, so a crash mid-generation (e.g. a CUDA
  error that kills the process) still leaves a trace of the last thing that happened on disk.
- Defines `run_generate()`, the function bound to the "Generate" button. It validates the prompt,
  calls `flux_pipeline.generate()`, handles/logs exceptions, saves the resulting image to the
  configured output directory with a timestamped filename, and returns the image + status text to
  the UI.
- Defines the actual page layout using `gr.Blocks` — all the sliders, textboxes, and the image
  upload widget described above in [Using the UI](#using-the-ui).

### `flux_pipeline.py`

The model-loading and inference logic, kept separate from the UI so it can be reused or tested
independently. Responsibilities:

- `PipelineCache` — holds the currently-loaded pipeline(s) in memory so multi-gigabyte weights
  aren't reloaded from disk on every single generation. It reloads automatically if you change the
  model path, precision, max-token setting, or Redux path between generations.
- Applies memory optimizations on load: attention slicing, VAE tiling, and **sequential** CPU
  offload (as opposed to whole-component offload) — this keeps VRAM usage low enough for img2img
  (where the VAE runs both before and after the transformer) to fit on a single 24GB-class GPU
  without spilling into (much slower) shared system memory.
  Loads the optional Flux Redux prior pipeline for style-reference generation, if a Redux model path
  is provided.
- `generate()` — the actual generation entry point. Branches into one of three code paths depending
  on whether a reference image was provided and, if so, which mode:
  - **Style (Redux)**: encodes the reference image into embeddings via the Redux prior, concatenates
    them with the text prompt's embeddings, and runs standard txt2img denoising from pure noise.
  - **Img2img**: runs `FluxImg2ImgPipeline`, starting from the reference image's own latent.
  - **Plain txt2img**: standard text-to-image generation.

  In all three cases, `reference_strength` is exposed as an intuitive, non-inverted 0–1 control
  (1.0 = stick close to the reference) and internally converted to whatever each mode actually
  expects.

### `requirements.txt`

Python package dependencies: `torch`, `diffusers`, `transformers`, `accelerate`, `sentencepiece`,
`protobuf`, `gradio`, `pillow`, and `python-dotenv`.

### `.env` / `.env.example`

Machine-specific configuration — see [Configuration (.env)](#configuration-env) above.

### `.gitignore`

Excludes `.env` (your local secrets/paths), Python bytecode caches, virtual environment folders,
and the rolling log file, so none of that ends up committed to the repository.

### `logs/app.log`

Plain-text log file, recreated/appended to on every run. Contains one line per denoising step plus
generation start/finish/error events — useful for diagnosing hangs or crashes after the fact, since
both log handlers flush immediately on every write.

## Logging

Every generation logs:

- The requested settings (mode, size, steps, CFG values, seed, etc.).
- Pipeline load/unload events (including when the cache reloads due to a changed setting).
- One line per denoising step, so you can tell exactly which step a hang or crash occurred on.
- Success (with the seed actually used) or a full traceback on failure.

Logs go to both the console and `logs/app.log`. The log file is excluded from git via
`.gitignore`.

## Troubleshooting

- **"Style reference mode requires a valid Redux model path."** — you selected "Style (Redux)" mode
  but `FLUX_REDUX_MODEL_PATH` (or the override in "Model / output settings") isn't set to a valid
  local Redux model folder.
- **Out of memory / CUDA OOM** — try lowering width/height, or ensure `Use fp16` is enabled. This app
  is tuned for ~24GB-class GPUs; smaller cards may need lower resolutions.
- **Very slow generation** — check `logs/app.log` for `device=cpu`, which means CUDA wasn't detected
  (no CUDA-enabled PyTorch installed, or no compatible GPU/driver). See the note in
  [Installation](#installation) about installing the correct PyTorch build.
- **Model fails to load / file-not-found errors** — double check `FLUX_MODEL_PATH` (and
  `FLUX_REDUX_MODEL_PATH` if using Style mode) in your `.env` point at a folder containing the
  model in `diffusers` format (a `model_index.json` plus component subfolders), not a single
  `.safetensors` file.
