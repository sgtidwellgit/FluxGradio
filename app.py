"""Simple local Gradio front end for Flux image generation.

Wraps flux_pipeline.py (which mirrors the pipeline setup used in
PopArt_Flux_Generator.py) with a browser UI: prompt/negative prompt,
size, steps, guidance, seed, and an optional guide image for img2img.

Run:
    python app.py
Then open http://127.0.0.1:7860
"""

import logging
import os
import traceback
from datetime import datetime
from pathlib import Path

import gradio as gr
from dotenv import load_dotenv

from flux_pipeline import generate

load_dotenv()

DEFAULT_MODEL_PATH = os.environ.get("FLUX_MODEL_PATH", "")
DEFAULT_REDUX_MODEL_PATH = os.environ.get("FLUX_REDUX_MODEL_PATH", "")
DEFAULT_OUTDIR = os.environ.get("FLUX_OUTPUT_DIR", "")

LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "app.log"

# FileHandler/StreamHandler both flush on every emit, so even if the process
# later dies outright (e.g. a CUDA crash that takes the whole app down), the
# last lines written before that point are already on disk in LOG_FILE.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("flux_frontend.app")


def run_generate(
    model_path,
    redux_model_path,
    outdir,
    prompt,
    negative_prompt,
    width,
    height,
    steps,
    cfg,
    true_cfg_scale,
    seed,
    max_tokens,
    use_fp16,
    reference_image,
    reference_strength,
    reference_mode,
):
    if not prompt or not prompt.strip():
        return None, "Enter a positive prompt first.", ""

    mode = reference_mode if reference_image is not None else "txt2img"
    logger.info(
        "Generate requested: mode=%s size=%sx%s steps=%s cfg=%s true_cfg=%s "
        "seed=%s max_tokens=%s ref_strength=%s",
        mode, width, height, steps, cfg, true_cfg_scale, seed, max_tokens,
        reference_strength if reference_image is not None else "n/a",
    )

    try:
        image, seed_used = generate(
            model_path=model_path,
            redux_model_path=redux_model_path,
            prompt=prompt,
            negative_prompt=negative_prompt or "",
            width=int(width),
            height=int(height),
            steps=int(steps),
            cfg=float(cfg),
            seed=int(seed),
            max_tokens=int(max_tokens),
            use_fp16=bool(use_fp16),
            reference_image=reference_image,
            reference_strength=float(reference_strength),
            reference_mode=reference_mode,
            true_cfg_scale=float(true_cfg_scale),
        )
    except Exception:
        tb = traceback.format_exc()
        logger.error("Generation failed:\n%s", tb)
        status = f"Generation failed: {tb.strip().splitlines()[-1]}\nFull traceback below, and in {LOG_FILE}"
        return None, status, tb

    logger.info("Generation succeeded: seed=%s", seed_used)

    out_dir = Path(outdir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    out_path = out_dir / f"{stamp}_{mode}_seed{seed_used}.png"
    image.save(out_path)
    logger.info("Saved %s", out_path.resolve())

    info = f"Seed used: {seed_used}\nSaved to: {out_path.resolve()}"
    return image, info, ""


with gr.Blocks(title="Flux Front End") as demo:
    gr.Markdown("# Flux Front End\nLocal Flux.1-dev generation with optional image guidance.")

    with gr.Row():
        with gr.Column(scale=1):
            prompt = gr.Textbox(label="Positive prompt", lines=6, placeholder="Describe the image you want...")
            negative_prompt = gr.Textbox(
                label="Negative prompt",
                lines=3,
                placeholder="Describe what to avoid...",
            )

            with gr.Row():
                width = gr.Slider(
                    256, 4096, value=1024, step=16, label="Width",
                    info="Past ~2048, expect artifacts (duplicated/tiled elements) — Flux wasn't trained at these sizes",
                )
                height = gr.Slider(
                    256, 4096, value=1024, step=16, label="Height",
                    info="Past ~2048, expect artifacts (duplicated/tiled elements) — Flux wasn't trained at these sizes",
                )

            with gr.Row():
                steps = gr.Slider(1, 100, value=24, step=1, label="Steps")
                cfg = gr.Slider(0.0, 20.0, value=5.5, step=0.1, label="Guidance scale (CFG)")

            with gr.Row():
                true_cfg_scale = gr.Slider(
                    1.0, 5.0, value=1.0, step=0.1,
                    label="True CFG scale",
                    info="Negative prompt only has an effect above 1.0 (Flux-dev is guidance-distilled)",
                )
                seed = gr.Number(value=42, precision=0, label="Seed (-1 = random)")

            max_tokens = gr.Slider(
                64, 512, value=512, step=64, label="Max T5 tokens",
                info="Flux's T5 encoder hard-caps at 512 tokens",
            )
            use_fp16 = gr.Checkbox(value=True, label="Use fp16 (requires CUDA)")

            gr.Markdown("### Reference image (optional)")
            reference_image = gr.Image(label="Reference Image", type="pil")
            reference_mode = gr.Radio(
                choices=[
                    ("Style (Redux) — new scene, borrows the look", "style"),
                    ("Img2img — restyle this exact image", "img2img"),
                ],
                value="style",
                label="Reference mode",
                info=(
                    "Style keeps your prompt's subject/composition and only pulls the visual style/mood "
                    "from the reference — generation starts from noise, not the reference's pixels, so it "
                    "won't just reproduce it. Img2img starts from the reference image itself and lightly "
                    "modifies it, so the output stays anchored to its composition regardless of the prompt."
                ),
            )
            reference_strength = gr.Slider(
                0.0, 1.0, value=0.6, step=0.01,
                label="Reference Strength",
                info=(
                    "How strongly the reference pulls the result. In Style mode: how hard the reference's "
                    "look competes with the prompt (0.2-0.4 = subtle influence, 0.8+ = strong). In Img2img "
                    "mode: how closely to stick to the reference image's exact pixels/composition (0.8+ = "
                    "stay close, 0.2-0.4 = loose starting point)."
                ),
            )

            with gr.Accordion("Model / output settings", open=False):
                model_path = gr.Textbox(value=DEFAULT_MODEL_PATH, label="Model path")
                redux_model_path = gr.Textbox(
                    value=DEFAULT_REDUX_MODEL_PATH,
                    label="Redux model path",
                    info="Only used when Reference mode is Style.",
                )
                outdir = gr.Textbox(value=DEFAULT_OUTDIR, label="Output directory")

            generate_btn = gr.Button("Generate", variant="primary")

        with gr.Column(scale=1):
            output_image = gr.Image(label="Result")
            output_info = gr.Textbox(label="Info", lines=3)
            output_error = gr.Textbox(
                label=f"Error log (also written to {LOG_FILE})",
                lines=15,
                max_lines=40,
                visible=True,
            )

    generate_btn.click(
        fn=run_generate,
        inputs=[
            model_path,
            redux_model_path,
            outdir,
            prompt,
            negative_prompt,
            width,
            height,
            steps,
            cfg,
            true_cfg_scale,
            seed,
            max_tokens,
            use_fp16,
            reference_image,
            reference_strength,
            reference_mode,
        ],
        outputs=[output_image, output_info, output_error],
    )

if __name__ == "__main__":
    demo.queue().launch()
