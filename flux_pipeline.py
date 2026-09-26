"""Pipeline loading and generation logic for the Flux front end.

Mirrors the setup used in PopArt_Flux_Generator.py (dual T5/CLIP tokenizers,
fp16 + CPU offload, attention/VAE tiling) but exposes it as reusable
text-to-image and image-guided (img2img) generation functions.
"""

import logging

import torch
from diffusers import FluxPipeline, FluxImg2ImgPipeline, FluxPriorReduxPipeline
from PIL import Image

logger = logging.getLogger("flux_frontend.pipeline")


class PipelineCache:
    """Holds the currently loaded pipeline so we don't reload multi-GB weights on every generate call."""

    def __init__(self):
        self.model_path = None
        self.redux_model_path = None
        self.use_fp16 = None
        self.max_tokens = None
        self.txt2img = None
        self.img2img = None
        self.redux = None
        self.device = None

    def get(self, model_path: str, use_fp16: bool, max_tokens: int, redux_model_path: str | None = None):
        if (
            self.txt2img is not None
            and self.model_path == model_path
            and self.use_fp16 == use_fp16
            and self.max_tokens == max_tokens
            and self.redux_model_path == redux_model_path
        ):
            return self.txt2img, self.img2img, self.redux, self.device

        self._unload()

        dtype = torch.float16 if (use_fp16 and torch.cuda.is_available()) else torch.float32
        device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info("Loading pipeline from %s (device=%s, dtype=%s)", model_path, device, dtype)
        pipe = FluxPipeline.from_pretrained(model_path, torch_dtype=dtype, local_files_only=True)
        logger.info("Pipeline weights loaded, applying memory optimizations")
        pipe.safety_checker = None
        pipe.enable_attention_slicing()
        #pipe.enable_vae_tiling()
        pipe.vae.enable_tiling()
        # The flux-dev transformer alone is ~23.8GB in fp16, leaving almost no
        # headroom on a 24GB card. enable_model_cpu_offload() swaps whole
        # components at once, and for img2img the VAE runs both before and
        # after the transformer, which can leave both resident on the GPU at
        # the same time and overflow into (catastrophically slow) shared
        # system memory. enable_sequential_cpu_offload() moves weights at the
        # individual-layer level instead, staying well under the VRAM ceiling.
        pipe.enable_sequential_cpu_offload()

        # Note: the actual T5 token budget is applied per-call via `max_sequence_length`
        # in generate() below, not via tokenizer.model_max_length. Mutating the CLIP
        # tokenizer's model_max_length here would corrupt FluxPipeline's cached
        # `tokenizer_max_length` (snapshotted at construction time and re-snapshotted
        # by from_pipe() below), causing CLIP encoding to attempt sequences longer
        # than its fixed 77-position limit.

        img2img_pipe = FluxImg2ImgPipeline.from_pipe(pipe)
        logger.info("Pipeline ready (txt2img + img2img)")

        redux_pipe = None
        if redux_model_path:
            # Redux only has an image encoder + embedder (~1GB total, no
            # transformer/VAE), so unlike the main pipe it comfortably fits
            # fully resident on the GPU without offload. It has no text
            # encoder of its own (model_index.json doesn't declare one) --
            # the actual prompt is encoded separately via `pipe.encode_prompt`
            # in generate() and concatenated with this pipe's image embedding.
            logger.info("Loading Redux style-reference prior from %s", redux_model_path)
            redux_pipe = FluxPriorReduxPipeline.from_pretrained(
                redux_model_path, torch_dtype=dtype, local_files_only=True
            )
            redux_pipe.to(device)
            logger.info("Redux prior ready")

        self.model_path = model_path
        self.redux_model_path = redux_model_path
        self.use_fp16 = use_fp16
        self.max_tokens = max_tokens
        self.txt2img = pipe
        self.img2img = img2img_pipe
        self.redux = redux_pipe
        self.device = device

        return self.txt2img, self.img2img, self.redux, self.device

    def _unload(self):
        self.txt2img = None
        self.img2img = None
        self.redux = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


_cache = PipelineCache()


def generate(
    model_path: str,
    prompt: str,
    negative_prompt: str,
    width: int,
    height: int,
    steps: int,
    cfg: float,
    seed: int,
    max_tokens: int = 1024,
    use_fp16: bool = True,
    reference_image: Image.Image | None = None,
    reference_strength: float = 0.6,
    reference_mode: str = "img2img",
    redux_model_path: str | None = None,
    true_cfg_scale: float = 1.0,
) -> tuple[Image.Image, int]:
    """Generate an image, optionally guided by a reference image.

    reference_mode selects how the reference image is used:
      - "img2img": structure-preserving. Starts from the reference image's own
        latent and denoises it -- the output's composition/layout stays
        anchored to the input regardless of what the prompt describes. Use
        this to restyle or lightly touch up an existing image.
      - "style": Flux Redux style/subject conditioning. The reference image is
        encoded into embeddings that are concatenated onto the text prompt's
        embeddings (not the image's own latent), so generation still starts
        from pure noise and follows the prompt's subject/composition, while
        picking up the reference's visual style. Use this when you want a new
        scene "in the style of" the reference, without duplicating it.

    reference_strength is the intuitive, non-inverted control: 1.0 = stay as
    close as possible to the reference image, 0.0 = treat it as a loose
    starting point and let the prompt drive most of the result.
      - In "img2img" mode this is inverted into diffusers' img2img `strength`
        (where 0 = keep the input, 1 = ignore it), and clamped away from
        exactly 0 since a diffusers strength of 0 runs zero denoising steps.
      - In "style" mode this scales the magnitude of the image embedding
        before it's concatenated with the (unscaled) text embedding, so
        higher values push the style/subject signal harder relative to the
        prompt. This scaling is empirical (Redux has no official "strength"
        knob) and may need per-prompt tuning.

    Flux-dev is guidance-distilled: negative_prompt is only actually applied
    when true_cfg_scale > 1 (it runs a real second forward pass per step).
    At true_cfg_scale == 1.0 the negative prompt is accepted but ignored.

    Returns (image, seed_used).
    """
    use_style_mode = reference_image is not None and reference_mode == "style"
    txt2img, img2img, redux, device = _cache.get(
        model_path, use_fp16, max_tokens, redux_model_path if use_style_mode else None
    )

    if seed is None or seed < 0:
        seed = torch.seed() % (2**32 - 1)
    gen = torch.Generator(device=device).manual_seed(int(seed))

    # Flux requires height/width to be multiples of 16 (VAE downsample factor 8, patch size 2).
    width = max(16, (int(width) // 16) * 16)
    height = max(16, (int(height) // 16) * 16)

    def make_log_step(actual_step_count):
        # Fires every denoising step, so a hang mid-generation shows exactly
        # which step it stalled on in the log file (vs. stuck loading, or
        # stuck in the VAE decode that happens after the last step).
        def log_step(pipe, step, timestep, callback_kwargs):
            logger.info("Denoising step %d/%d (timestep=%s)", step + 1, actual_step_count, timestep)
            return callback_kwargs
        return log_step

    if use_style_mode:
        if redux is None:
            raise ValueError("Style reference mode requires a valid Redux model path.")
        logger.info(
            "Encoding style reference via Redux (reference_strength=%.2f)", reference_strength
        )
        ref_image = reference_image.convert("RGB")
        with torch.no_grad():
            image_latents = redux.encode_image(ref_image, device, 1)
            image_embeds = redux.image_embedder(image_latents).image_embeds.to(device=device)

            prompt_embeds, pooled_prompt_embeds, _ = txt2img.encode_prompt(
                prompt=prompt,
                prompt_2=prompt,
                device=device,
                num_images_per_prompt=1,
                max_sequence_length=int(max_tokens),
            )
            # Redux's own default weighting assumes the image embedding is
            # concatenated onto a *zero* text embedding (pure image-to-image
            # variation); here it's concatenated onto our real prompt
            # embedding instead, so we scale it relative to reference_strength
            # rather than using it at Redux's default magnitude.
            image_embeds = image_embeds.to(dtype=prompt_embeds.dtype) * (float(reference_strength) * 2.0)
            combined_embeds = torch.cat([prompt_embeds, image_embeds], dim=1)

        logger.info("Starting style-referenced txt2img denoising (%d steps requested)", steps)
        result = txt2img(
            prompt_embeds=combined_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_prompt=negative_prompt,
            true_cfg_scale=float(true_cfg_scale),
            guidance_scale=float(cfg),
            num_inference_steps=int(steps),
            height=height,
            width=width,
            max_sequence_length=int(max_tokens),
            generator=gen,
            callback_on_step_end=make_log_step(int(steps)),
        )
    elif reference_image is not None:
        reference_image = reference_image.convert("RGB").resize((width, height))
        diffusers_strength = max(0.02, min(1.0, 1.0 - float(reference_strength)))
        # img2img trims the schedule based on strength (diffusers get_timesteps):
        # low diffusers strength = high reference adherence = fewer real steps
        # actually run, regardless of the requested step count.
        actual_steps = max(1, int(steps) - int(max(int(steps) - min(int(steps) * diffusers_strength, int(steps)), 0)))
        logger.info(
            "Starting img2img denoising (%d steps requested, ~%d will actually run at reference_strength=%.2f)",
            steps, actual_steps, reference_strength,
        )
        result = img2img(
            prompt=prompt,
            negative_prompt=negative_prompt,
            true_cfg_scale=float(true_cfg_scale),
            image=reference_image,
            strength=diffusers_strength,
            guidance_scale=float(cfg),
            num_inference_steps=int(steps),
            height=height,
            width=width,
            max_sequence_length=int(max_tokens),
            generator=gen,
            callback_on_step_end=make_log_step(actual_steps),
        )
    else:
        logger.info("Starting txt2img denoising (%d steps requested)", steps)
        result = txt2img(
            prompt=prompt,
            negative_prompt=negative_prompt,
            true_cfg_scale=float(true_cfg_scale),
            guidance_scale=float(cfg),
            num_inference_steps=int(steps),
            height=height,
            width=width,
            max_sequence_length=int(max_tokens),
            generator=gen,
            callback_on_step_end=make_log_step(int(steps)),
        )
    logger.info("Denoising finished, decoding image")

    return result.images[0], int(seed)
