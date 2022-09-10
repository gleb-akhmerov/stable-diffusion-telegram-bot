import argparse, os, sys, glob, random
import getopt
from dataclasses import dataclass
from functools import partial
from io import BytesIO
from math import sqrt
from typing import Literal, Optional, Union, IO

import torch
import numpy as np
import copy
import cv2
import skimage
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from itertools import islice, chain
from einops import rearrange, repeat
from torchvision.utils import make_grid
import time
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import contextmanager, nullcontext
from ldm.util import instantiate_from_config
from telegram import Update, InputMediaPhoto
from telegram.ext import Updater, CommandHandler, CallbackContext, Filters

from optimizedSD.ddpm import UNet


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def ichunk(it, size):
    it = iter(it)
    while True:
        chunk_it = islice(it, size)
        try:
            first_el = next(chunk_it)
        except StopIteration:
            return
        yield chain((first_el,), chunk_it)


def load_model_from_config(ckpt):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    return sd


def load_img(path, h0, w0) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    w, h = image.size

    print(f"loaded input image of size ({w}, {h}) from {path}")
    if h0 is not None and w0 is not None:
        h, w = h0, w0

    w, h = map(lambda x: x - x % 64, (w, h))  # resize to integer multiple of 32

    print(f"New image size ({w}, {h})")
    image = image.resize((w, h), resample=Image.Resampling.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.0 * image - 1.0


config_path = "optimizedSD/v1-inference.yaml"
ckpt = "models/ldm/stable-diffusion-v1/model.ckpt"


@dataclass
class ModelSet:
    model: UNet
    modelCS: UNet
    modelFS: UNet


def load_model_set(device: str, turbo: bool = True) -> ModelSet:
    sd = load_model_from_config(f"{ckpt}")
    li = []
    lo = []
    for key, value in sd.items():
        sp = key.split('.')
        if (sp[0]) == 'model':
            if 'input_blocks' in sp:
                li.append(key)
            elif 'middle_block' in sp:
                li.append(key)
            elif 'time_embed' in sp:
                li.append(key)
            else:
                lo.append(key)
    for key in li:
        sd['model1.' + key[6:]] = sd.pop(key)
    for key in lo:
        sd['model2.' + key[6:]] = sd.pop(key)

    config = OmegaConf.load(config_path)

    model = instantiate_from_config(config.modelUNet)
    _, _ = model.load_state_dict(sd, strict=False)
    model.eval()
    model.unet_bs = 1
    model.cdevice = device
    model.turbo = turbo

    modelCS = instantiate_from_config(config.modelCondStage)
    _, _ = modelCS.load_state_dict(sd, strict=False)
    modelCS.eval()
    modelCS.cond_stage_model.device = device

    modelFS = instantiate_from_config(config.modelFirstStage)
    _, _ = modelFS.load_state_dict(sd, strict=False)
    modelFS.eval()

    return ModelSet(model=model, modelCS=modelCS, modelFS=modelFS)


def generate(
    model_set: ModelSet,
    prompt: str,
    ddim_steps: int,
    n_samples: int,
    scale: float = 7.5,
    seed: int = 42,
    precision: Literal["autocast", "full"] = "autocast",
    fixed_code: bool = False,
    n_iter: int = 1,
    H: int = 512,
    W: int = 512,
    C: int = 4,
    f: int = 8,
    ddim_eta: float = 0.0,
    img: Optional[Union[IO[bytes], str]] = None,  # path or file-like
    img_prompt_strength: float = 0.8,
    device: str = "cuda",
    sampler: Literal["plms", "ddim"] = "ddim",
):
    seed_everything(seed)

    if device != 'cpu' and precision == "autocast":
        model_set.model.half()
        model_set.modelFS.half()
        model_set.modelCS.half()

    start_code = None
    if fixed_code:
        start_code = torch.randn([n_samples, C, H // f, W // f], device=device)

    batch_size = n_samples
    data = [batch_size * [prompt]]

    if img is not None:
        init_image = load_img(img, H, W).to(device)

        if device != 'cpu' and precision == "autocast":
            init_image = init_image.half()

        model_set.modelFS.to(device)

        init_image = repeat(init_image, '1 ... -> b ...', b=batch_size)
        init_latent = model_set.modelFS.get_first_stage_encoding(
            model_set.modelFS.encode_first_stage(init_image)
        )  # move to latent space

        if device != 'cpu':
            mem = torch.cuda.memory_allocated() / 1e6
            model_set.modelFS.to("cpu")
            while torch.cuda.memory_allocated() / 1e6 >= mem:
                time.sleep(1)

        t_enc = int(img_prompt_strength * ddim_steps)
        print(f"target t_enc is {t_enc} steps")

    if precision == "autocast" and device != "cpu":
        precision_scope = autocast
    else:
        precision_scope = nullcontext

    with torch.no_grad():

        for _ in trange(n_iter, desc="Sampling"):
            for prompts in tqdm(data, desc="data"):
                with precision_scope("cuda"):
                    model_set.modelCS.to(device)
                    uc = None
                    if scale != 1.0:
                        uc = model_set.modelCS.get_learned_conditioning(batch_size * [""])
                    if isinstance(prompts, tuple):
                        prompts = list(prompts)

                    c = model_set.modelCS.get_learned_conditioning(prompts)
                    if device != 'cpu':
                        mem = torch.cuda.memory_allocated() / 1e6
                        model_set.modelCS.to("cpu")
                        while torch.cuda.memory_allocated() / 1e6 >= mem:
                            time.sleep(1)

                    if img is not None:
                        # encode (scaled latent)
                        z_enc = model_set.model.stochastic_encode(
                            x0=init_latent,
                            t=torch.tensor([t_enc]*batch_size).to(device),
                            seed=seed,
                            ddim_eta=ddim_eta,
                            ddim_steps=ddim_steps,
                        )
                        # decode it
                        samples_ddim = model_set.model.sample(
                            t_enc,
                            c,
                            z_enc,
                            unconditional_guidance_scale=scale,
                            unconditional_conditioning=uc,
                            sampler=sampler,
                        )
                    else:
                        shape = [n_samples, C, H // f, W // f]
                        samples_ddim = model_set.model.sample(
                            S=ddim_steps,
                            conditioning=c,
                            shape=shape,
                            verbose=False,
                            unconditional_guidance_scale=scale,
                            unconditional_conditioning=uc,
                            eta=ddim_eta,
                            x_T=start_code,
                            seed=seed,
                            sampler=sampler,
                        )

                    model_set.modelFS.to(device)
                    print("saving images")
                    for i in range(batch_size):
                        x_samples_ddim = model_set.modelFS.decode_first_stage(
                            samples_ddim[i].unsqueeze(0)
                        )
                        x_sample = torch.clamp(
                            (x_samples_ddim + 1.0) / 2.0,
                            min=0.0,
                            max=1.0
                        )

                        x_sample = 255. * rearrange(
                            x_sample[0].cpu().numpy(),
                            'c h w -> h w c'
                        )
                        bio = BytesIO()
                        Image.fromarray(x_sample.astype(np.uint8)).save(bio, format="png")
                        bio.seek(0)
                        seed += 1
                        yield bio.read()

                    if device != 'cpu':
                        mem = torch.cuda.memory_allocated() / 1e6
                        model_set.modelFS.to("cpu")
                        while torch.cuda.memory_allocated() / 1e6 >= mem:
                            time.sleep(1)

                    del samples_ddim


def do_loop(
    model_set,
    prompt,
    ddim_steps,
    n_samples,
    n_iter,
    seed,
    scale,
    img,
    img_prompt_strength,
    W,
    H,
    device,
    sampler,
    loop_steps,
):
    loop_seed = seed

    # color correction
    correction_target = cv2.cvtColor(
        cv2.imdecode(np.frombuffer(img.getvalue(), dtype=np.uint8), cv2.IMREAD_COLOR),
        cv2.COLOR_BGR2LAB,
    )

    for _ in range(loop_steps):
        img_bytes = next(
            generate(
                model_set=model_set,
                prompt=prompt,
                ddim_steps=ddim_steps,
                n_samples=n_samples,
                n_iter=n_iter,
                seed=loop_seed,
                scale=scale,
                img=img,
                img_prompt_strength=img_prompt_strength,
                W=W,
                H=H,
                device=device,
                sampler=sampler,
            )
        )
        new_img = BytesIO()
        Image.fromarray(
            cv2.cvtColor(
                skimage.exposure.match_histograms(
                    cv2.cvtColor(
                        cv2.imdecode(np.frombuffer(img_bytes, dtype=np.uint8), cv2.IMREAD_COLOR),
                        cv2.COLOR_BGR2LAB
                    ),
                    correction_target,
                    channel_axis=2
                ),
                cv2.COLOR_LAB2RGB
            ).astype("uint8")
        ).save(new_img, format="png")
        img = new_img
        loop_seed += 1
        yield img.getvalue()


def txt2img_command(
    update: Update, context: CallbackContext, model_set: ModelSet, device: str
) -> None:
    print(f"Command: {update.message.text}")

    prefix_len = len("/g ")
    prompt = update.message.text[prefix_len:].strip()
    if not prompt:
        update.message.reply_text(
            "Prompt must be non-empty. Example: /g Red cat", quote=True,
        )
        return

    longopts, prompt_unjoined = getopt.getopt(
        prompt.split(),
        shortopts='',
        longopts=['seed=', 'scale=', 'steps=', 'sampler='],
    )
    longopts = dict(longopts)
    prompt = ' '.join(prompt_unjoined)

    seed = int(longopts.get("--seed", random.randint(0, 99999999)))
    scale = float(longopts.get("--scale", 7.5))
    steps = int(longopts.get("--steps", 50))
    if not 1 <= steps <= 250:
        update.message.reply_text(
            "--steps must be between 1 and 250", quote=True
        )
        return
    sampler = longopts.get("--sampler", "ddim")
    if sampler not in ["plms", "ddim"]:
        update.message.reply_text(
            "--sampler must be one of: plms, ddim", quote=True
        )
        return

    batch_size = 3
    batches = 3

    images = generate(
        model_set=model_set,
        prompt=prompt,
        ddim_steps=steps,
        n_samples=batch_size,
        n_iter=batches,
        seed=seed,
        scale=scale,
        device=device,
        sampler=sampler,
    )
    with open("loading.png", "rb") as f:
        loading_bytes = f.read()
    loading_photos = [
        InputMediaPhoto(loading_bytes) for _ in range(batches * batch_size)
    ]
    loading_photos[0].caption = f"{prompt}\n\nSeed: {seed}"
    messages = update.message.reply_media_group(
        loading_photos,
        quote=True,
        timeout=60,
    )
    for message, image in zip(messages, images):
        message.edit_media(InputMediaPhoto(image, caption=message.caption), timeout=60)

    print()


def img2img_command(
        update: Update, context: CallbackContext, model_set: ModelSet, device: str
) -> None:
    print(f"Command: {update.message.text}")

    prefix_len = len("/i ")
    prompt = update.message.text[prefix_len:].strip()

    img_bio = BytesIO()
    media_mes = update.message.reply_to_message
    media = media_mes.photo[-1] if media_mes.photo != [] else media_mes.document
    media.get_file().download(out=img_bio)
    img_bio.seek(0)

    # W and H to resize keeping aspect ratio and maintaining fixed number of pixels
    # (fixed area)
    w, h = Image.open(img_bio).size
    img_bio.seek(0)
    area = 512 * 512
    aspect_ratio = w / h
    resized_w = round(sqrt(aspect_ratio) * sqrt(area))
    resized_h = round(sqrt(area) / sqrt(aspect_ratio))

    longopts, prompt_unjoined = getopt.getopt(
        prompt.split(),
        shortopts='',
        longopts=['seed=', 'scale=', 'steps=', 'sampler=', 'loop='],
    )
    longopts = dict(longopts)
    strength, *prompt_unjoined = prompt_unjoined
    strength = int(strength)
    if not 0 < strength < 100:
        update.message.reply_text(
            "Strength must be between 1 and 100", quote=True, timeout=10
        )
        return
    strength = strength / 100
    prompt = ' '.join(prompt_unjoined)

    if not prompt:
        update.message.reply_text(
            "Prompt must be non-empty. Example: /i 50 Red cat", quote=True, timeout=10
        )
        return

    seed = int(longopts.get("--seed", random.randint(0, 99999999)))
    scale = float(longopts.get("--scale", 7.5))
    steps = int(longopts.get("--steps", 50))
    if not 1 <= steps <= 250:
        update.message.reply_text(
            "--steps must be between 1 and 250", quote=True
        )
        return
    sampler = longopts.get("--sampler", "ddim")
    if sampler not in ["plms", "ddim"]:
        update.message.reply_text(
            "--sampler must be one of: plms, ddim", quote=True
        )
        return
    loop = int(longopts.get("--loop", 1))
    if not 1 <= loop <= 50:
        update.message.reply_text(
            "--loop must be between 1 and 50", quote=True
        )
        return

    if loop == 1:
        batch_size = 3
        batches = 3
        images = generate(
            model_set=model_set,
            prompt=prompt,
            ddim_steps=steps,
            n_samples=batch_size,
            n_iter=batches,
            seed=seed,
            scale=scale,
            img=img_bio,
            img_prompt_strength=strength,
            W=resized_w,
            H=resized_h,
            device=device,
            sampler=sampler,
        )

        with open("loading.png", "rb") as f:
            loading_bytes = f.read()
        loading_photos = [
            InputMediaPhoto(loading_bytes) for _ in range(batch_size * batches)
        ]
        loading_photos[0].caption = f"{prompt}\n\nSeed: {seed}"
        messages = update.message.reply_media_group(
            loading_photos,
            quote=True,
            timeout=60,
        )
        for message, image in zip(messages, images):
            message.edit_media(InputMediaPhoto(image, caption=message.caption), timeout=60)

    else:
        batch_size = 1
        batches = 1
        img_count = loop
        loop_seed = seed
        images = do_loop(
            model_set=model_set,
            prompt=prompt,
            ddim_steps=steps,
            n_samples=batch_size,
            n_iter=batches,
            seed=loop_seed,
            scale=scale,
            img=img_bio,
            img_prompt_strength=strength,
            W=resized_w,
            H=resized_h,
            device=device,
            sampler=sampler,
            loop_steps=loop,
        )

        with open("loading.png", "rb") as f:
            loading_bytes = f.read()
        imgs_left = loop
        for imgs in ichunk(images, 9):
            loading_photos = [
                InputMediaPhoto(loading_bytes) for _ in range(min(imgs_left, 9))
            ]
            loading_photos[0].caption = f"{prompt}\n\nSeed: {seed}"
            messages = update.message.reply_media_group(
                loading_photos,
                quote=True,
                timeout=60,
            )
            for message, image in zip(messages, imgs):
                message.edit_media(InputMediaPhoto(image, caption=message.caption), timeout=60)
                imgs_left -= 1

    print()


def main() -> None:
    device = "cuda"
    model_set = load_model_set(device=device)

    updater = Updater("TOKEN")
    dispatcher = updater.dispatcher
    dispatcher.add_handler(
        CommandHandler(
            "g",
            partial(txt2img_command, model_set=model_set, device=device),
        )
    )
    dispatcher.add_handler(
        CommandHandler(
            "i",
            partial(img2img_command, model_set=model_set, device=device),
        )
    )

    updater.start_polling()
    updater.idle()


if __name__ == '__main__':
    main()
