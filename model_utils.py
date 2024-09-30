from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn

from transformers.activations import ACT2FN
from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import (
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from transformers.models.clip.configuration_clip import CLIPConfig, CLIPTextConfig, CLIPVisionConfig
from transformers.models.clip.modeling_clip import *
from diffusers import StableDiffusionPipeline
from typing import Any, Callable, Dict, List, Optional, Union
from diffusers.pipelines.stable_diffusion.pipeline_output import StableDiffusionPipelineOutput
from diffusers.image_processor import PipelineImageInput
from tqdm import tqdm
import math
import copy
from diffusers import IFPipeline


def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
    """
    Rescale `noise_cfg` according to `guidance_rescale`. Based on findings of [Common Diffusion Noise Schedules and
    Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf). See Section 3.4
    """
    std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    # rescale the results from guidance (fixes overexposure)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    # mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
    noise_cfg = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
    return noise_cfg


def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    **kwargs,
):
    """
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used,
            `timesteps` must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
                Custom timesteps used to support arbitrary spacing between timesteps. If `None`, then the default
                timestep spacing strategy of the scheduler is used. If `timesteps` is passed, `num_inference_steps`
                must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        #scheduler.timesteps = torch.tensor([999]).cuda()
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        #scheduler.timesteps = torch.tensor([999]).cuda()
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


class CustomStableDiffusionPipeline(StableDiffusionPipeline):
    def aug_prompt(
        self,
        prompt=None,
        height=None,
        width=None,
        num_inference_steps=50,
        guidance_scale=7.5,
        negative_prompt=None,
        num_images_per_prompt=1,
        eta=0.0,
        generator=None,
        latents=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        target_steps=[0],
        lr=0.1,
        optim_iters=10,
        target_loss=None,
        print_optim=True,
        optim_epsilon=None,
        alpha=0.5,
    ):
        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        prompt_embeds = self._encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
        )

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7. Denoising loop
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = (
                    torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                )
                latent_model_input = self.scheduler.scale_model_input(
                    latent_model_input, t
                )

                if i in target_steps:
                    single_prompt_embeds = prompt_embeds[[-1], :, :].clone().detach()
                    if print_optim is True or optim_epsilon is not None:
                        init_embeds = single_prompt_embeds.clone()
                    single_prompt_embeds.requires_grad = True
                    dummy_prompt_embeds = prompt_embeds[[0], :, :].clone()

                    # optimizer
                    optimizer = torch.optim.AdamW([single_prompt_embeds], lr=lr)

                    prompt_tokens = self.tokenizer.encode(prompt)
                    prompt_tokens = prompt_tokens[1:-1]
                    prompt_tokens = prompt_tokens[:75]

                    curr_learnabel_mask = list(set(range(77)) - set([0]))

                    for j in range(optim_iters):
                        if print_optim is True or optim_epsilon is not None:
                            with torch.no_grad():
                                tmp_init_embeds = init_embeds[:, curr_learnabel_mask]
                                tmp_init_embeds = tmp_init_embeds.reshape(
                                    -1, tmp_init_embeds.shape[-1]
                                )
                                tmp_single_prompt_embeds = single_prompt_embeds[
                                    :, curr_learnabel_mask
                                ]
                                tmp_single_prompt_embeds = (
                                    tmp_single_prompt_embeds.reshape(
                                        -1, tmp_single_prompt_embeds.shape[-1]
                                    )
                                )

                                l_inf = torch.norm(
                                    tmp_init_embeds - tmp_single_prompt_embeds,
                                    p=float("inf"),
                                    dim=-1,
                                ).mean()
                                l_2 = torch.norm(
                                    tmp_init_embeds - tmp_single_prompt_embeds,
                                    p=2,
                                    dim=-1,
                                ).mean()

                        input_prompt_embeds = torch.cat(
                            [
                                dummy_prompt_embeds.repeat(num_images_per_prompt, 1, 1),
                                single_prompt_embeds.repeat(
                                    num_images_per_prompt, 1, 1
                                ),
                            ]
                        )

                        noise_pred = self.unet(
                            latent_model_input,
                            t,
                            encoder_hidden_states=input_prompt_embeds,
                            cross_attention_kwargs=None,
                            return_dict=False,
                        )[0]

                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred_text = noise_pred_text - noise_pred_uncond

                        noise_pred_text_norm = torch.norm(noise_pred_text, p=2).mean()
                        loss = noise_pred_text_norm
                        loss_item = loss.item()

                        if optim_epsilon is not None and l_2 > optim_epsilon:
                            tmp_init_embeds = init_embeds[:, curr_learnabel_mask]
                            tmp_init_embeds = tmp_init_embeds.reshape(
                                -1, tmp_init_embeds.shape[-1]
                            )
                            tmp_single_prompt_embeds = single_prompt_embeds[
                                :, curr_learnabel_mask
                            ]
                            tmp_single_prompt_embeds = tmp_single_prompt_embeds.reshape(
                                -1, tmp_single_prompt_embeds.shape[-1]
                            )

                            loss_l2 = torch.norm(
                                tmp_init_embeds - tmp_single_prompt_embeds, p=2, dim=-1
                            ).mean()

                            loss = alpha * loss + (1 - alpha) * loss_l2

                        if target_loss is not None:
                            if loss_item <= target_loss:
                                if print_optim is True:
                                    print(f"step: {j}, curr loss: {loss_item}")
                                break
                        
                        (single_prompt_embeds.grad,) = torch.autograd.grad(
                            loss, [single_prompt_embeds]
                        )
                        single_prompt_embeds.grad[:, [0]] = (
                            single_prompt_embeds.grad[:, [0]] * 0
                        )
                        
                        optimizer.step()
                        optimizer.zero_grad()

                        if print_optim is True:
                            print(f"step: {j}, curr loss: {loss_item}")

                    single_prompt_embeds = single_prompt_embeds.detach()
                    single_prompt_embeds.requires_grad = False
                    torch.cuda.empty_cache()
                    return single_prompt_embeds

                    with torch.no_grad():
                        noise_pred = (
                            noise_pred_uncond + guidance_scale * noise_pred_text
                        )
                        latents = self.scheduler.step(
                            noise_pred,
                            t,
                            latents,
                            **extra_step_kwargs,
                            return_dict=False,
                        )[0]
                else:
                    with torch.no_grad():
                        noise_pred = self.unet(
                            latent_model_input,
                            t,
                            encoder_hidden_states=prompt_embeds,
                            cross_attention_kwargs=None,
                            return_dict=False,
                        )[0]

                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred_text = noise_pred_text - noise_pred_uncond

                        noise_pred = (
                            noise_pred_uncond + guidance_scale * noise_pred_text
                        )

                        # compute the previous noisy sample x_t -> x_t-1
                        latents = self.scheduler.step(
                            noise_pred,
                            t,
                            latents,
                            **extra_step_kwargs,
                            return_dict=False,
                        )[0]

                progress_bar.update()

if __name__ == '__main__':
    breakpoint()
    print(1)