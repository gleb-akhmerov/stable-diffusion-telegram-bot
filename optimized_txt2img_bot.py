import argparse, os, sys, glob, random
import getopt
from dataclasses import dataclass
from functools import partial
from io import BytesIO
from typing import Literal

import torch
import numpy as np
import copy
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange
from torchvision.utils import make_grid
import time
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import contextmanager, nullcontext
from ldm.util import instantiate_from_config
from telegram import Update, InputMediaPhoto
from telegram.ext import Updater, CommandHandler, CallbackContext, Filters
from telegram.error import NetworkError


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def load_model_from_config(ckpt):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    return sd


config_path = "optimizedSD/v1-inference.yaml"
ckpt = "models/ldm/stable-diffusion-v1/model.ckpt"
device = "cuda"


@dataclass
class ModelSet:
    model: object
    modelCS: object
    modelFS: object


def generate(
    model_set: ModelSet,
    prompt: str,
    ddim_steps: int,
    n_samples: int,
    scale: float = 7.5,
    seed: int = 42,
    precision: Literal["autocast", "full"] = "autocast",
    fixed_code: bool = True,
    n_iter: int = 1,
    H: int = 512,
    W: int = 512,
    C: int = 4,
    f: int = 8,
    ddim_eta: float = 0.0,
):
    seed_everything(seed)

    if precision == "autocast":
        model_set.model.half()
        model_set.modelCS.half()

    start_code = None
    if fixed_code:
        start_code = torch.randn([n_samples, C, H // f, W // f], device=device)

    batch_size = n_samples
    data = [batch_size * [prompt]]

    precision_scope = autocast if precision == "autocast" else nullcontext

    results = []
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
                    shape = [C, H // f, W // f]
                    mem = torch.cuda.memory_allocated() / 1e6
                    model_set.modelCS.to("cpu")
                    while torch.cuda.memory_allocated() / 1e6 >= mem:
                        time.sleep(1)

                    samples_ddim = model_set.model.sample(
                        S=ddim_steps,
                        conditioning=c,
                        batch_size=n_samples,
                        shape=shape,
                        verbose=False,
                        unconditional_guidance_scale=scale,
                        unconditional_conditioning=uc,
                        eta=ddim_eta,
                        x_T=start_code,
                        seed=seed,
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
                        results.append(bio.read())

                    mem = torch.cuda.memory_allocated() / 1e6
                    model_set.modelFS.to("cpu")
                    while torch.cuda.memory_allocated() / 1e6 >= mem:
                        time.sleep(1)

                    del samples_ddim

    return results


def generate_command(update: Update, context: CallbackContext, model_set: ModelSet) -> None:
    prefix_len = len("/g ")
    prompt = update.message.text[prefix_len:].strip()
    print(f"Prompt: {prompt}")
    if not prompt:
        update.message.reply_text(
            "Prompt must be non-empty. Example: /g Red cat", quote=True,
        )
        return

    for _ in range(5):
        try:
            update.message.reply_text("Generating...", quote=True, timeout=10)
            break
        except NetworkError as e:
            print(e)
            continue

    longopts, prompt_unjoined = getopt.getopt(
        prompt.split(),
        shortopts='',
        longopts=['seed=', 'scale='],
    )
    longopts = dict(longopts)
    prompt = ' '.join(prompt_unjoined)

    seed = int(longopts.get("--seed", random.randint(0, 99999999)))
    scale = float(longopts.get("--scale", 7.5))
    images = generate(
        model_set=model_set,
        prompt=prompt,
        ddim_steps=50,
        n_samples=9,
        seed=seed,
        scale=scale,
    )
    for _ in range(5):
        try:
            media_group = [InputMediaPhoto(x) for x in images]
            media_group[0].caption = f"{prompt}\n\nSeed: {seed}"
            update.message.reply_media_group(
                media_group,
                quote=True,
                timeout=60
            )
            break
        except NetworkError as e:
            print(e)
            continue

    print()


def load_model_set(ddim_steps: int, small_batch: bool = False) -> ModelSet:
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
    config.modelUNet.params.ddim_steps = ddim_steps

    if small_batch:
        config.modelUNet.params.small_batch = True
    else:
        config.modelUNet.params.small_batch = False

    model = instantiate_from_config(config.modelUNet)
    _, _ = model.load_state_dict(sd, strict=False)
    model.eval()

    modelCS = instantiate_from_config(config.modelCondStage)
    _, _ = modelCS.load_state_dict(sd, strict=False)
    modelCS.eval()

    modelFS = instantiate_from_config(config.modelFirstStage)
    _, _ = modelFS.load_state_dict(sd, strict=False)
    modelFS.eval()

    return ModelSet(model=model, modelCS=modelCS, modelFS=modelFS)


def main() -> None:
    model_set = load_model_set(ddim_steps=50)

    updater = Updater("TOKEN")
    dispatcher = updater.dispatcher
    dispatcher.add_handler(
        CommandHandler(
            "g",
            partial(generate_command, model_set=model_set),
        )
    )

    updater.start_polling()
    updater.idle()


if __name__ == '__main__':
    main()
