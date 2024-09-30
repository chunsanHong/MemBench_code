import argparse
#import wandb
import copy
from tqdm import tqdm
from statistics import mean
from PIL import Image

import torch

import open_clip
from optim_utils import *
from io_utils import *

from diffusers import DDIMScheduler, UNet2DConditionModel, DiffusionPipeline, IFPipeline
import pandas as pd

import requests
from io import BytesIO
import os
import pickle
from model_utils import *
import time
import concurrent.futures
from aesthetic.model import aesthetic_predictor
import random
import ast
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

def save_image(img, save_path, img_idx, i):
    img.save(f'{save_path}/img_{str(i).zfill(6)}/gen_{img_idx}.png')

def save_images(gen_images, save_path, i, num_workers=16):
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for img_idx, img in enumerate(gen_images):
            futures.append(executor.submit(save_image, img, save_path, img_idx, i))
        concurrent.futures.wait(futures)

def main(args):
    # load diffusion model
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"}

    session = requests.Session()
    retries = Retry(
        total=5,
        backoff_factor=1,  
        status_forcelist=[500, 502, 503, 504],  
        raise_on_status=False  
    )
    adapter = HTTPAdapter(max_retries=retries)
    session.mount('https://', adapter)
    
    os.makedirs(args.save_path, exist_ok = True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.rescale_attention == None:
        pipe = CustomStableDiffusionPipeline.from_pretrained(
            'CompVis/stable-diffusion-v1-4',
            torch_dtype=torch.bfloat16,
            safety_checker=None,
            requires_safety_checker=False,
        )
    else:
        from MemAttn.refactored_classes.MemAttn import MemStableDiffusionPipeline
        from MemAttn.refactored_classes.refactored_unet_2d_condition import UNet2DConditionModel as MemUNet2DConditionModel
        unet = MemUNet2DConditionModel.from_pretrained(
                'CompVis/stable-diffusion-v1-4', subfolder="unet", torch_dtype=torch.bfloat16)
        pipe = MemStableDiffusionPipeline.from_pretrained('CompVis/stable-diffusion-v1-4', unet=unet,  safety_checker=None, torch_dtype=torch.bfloat16, requires_safety_checker = False)

        args.c1 = args.rescale_attention
        args.cross_attn_mask = True
        args.miti_mem = True
        args.mask_length_minis1 = False
        args.save_numpy = False

    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)

    if args.dataset != 'COCO':
        dir_sim_model = 'sscd_disc_large.torchscript.pt'
        sim_model = torch.jit.load(dir_sim_model).to('cuda')

    aesthetic_model = aesthetic_predictor()
    set_random_seed(args.gen_seed)

    if args.dataset == 'SD1':
        dataset = pd.read_csv('SD1_final.csv')
        dataset['urls'] = dataset['urls'].apply(ast.literal_eval)
    elif args.dataset == 'COCO':
        with open('captions_val2017.json', 'rb') as f: dataset = pd.DataFrame(json.load(f)['annotations']).drop_duplicates(subset='image_id')
        dataset.columns = ['image_id', 'id', 'prompt']

    # generation
    print("generation")

    reference_model = "ViT-g-14"
    reference_model_pretrain = "laion2b_s12b_b42k"

    ref_model, _, ref_clip_preprocess = open_clip.create_model_and_transforms(
        reference_model,
        pretrained=reference_model_pretrain,
        device=device,
    )
    ref_tokenizer = open_clip.get_tokenizer(reference_model)

    total_max_sim = []
    total_mean_clip_score = []
    total_top3_mean = []
    total_above_0_5 = []
    total_aesthetic_score = []
    
    progress_bar = tqdm(range(len(dataset)), desc="Processing", unit="batch")

    for i in progress_bar:
        prompt = dataset.iloc[i]['prompt']
        seed = i + args.gen_seed
        os.makedirs(f'{args.save_path}/img_{str(i).zfill(6)}', exist_ok = True)

        if args.prompt_aug_style is not None:
            prompt = prompt_augmentation(
            prompt,
            args.prompt_aug_style,
            tokenizer=pipe.tokenizer,
            repeat_num=args.repeat_num,
            )

        if args.optim_target_loss is not None:
            set_random_seed(seed)
            auged_prompt_embeds = pipe.aug_prompt(
                prompt,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                num_images_per_prompt=args.num_images_per_prompt,
                target_steps=[args.optim_target_steps],
                lr=args.optim_lr,
                optim_iters=args.optim_iters,
                target_loss=args.optim_target_loss,
            )

            ### generation
            set_random_seed(seed)
            outputs = pipe(
                prompt_embeds=auged_prompt_embeds,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                num_images_per_prompt=args.num_images_per_prompt,
            )
        elif args.rescale_attention != None:
            outputs = pipe(
                prompt=prompt,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                num_images_per_prompt=args.num_images_per_prompt,
                args = args,
            )
        else:
            outputs = pipe(
                prompt=prompt,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                num_images_per_prompt=args.num_images_per_prompt,
            )

        gen_images = outputs.images
        save_images(gen_images, args.save_path, i, num_workers=16)

        feats = load_and_process_images(gen_images, aesthetic_model.preprocess, device, num_workers=16)
        aesthetic_score = aesthetic_model.predict_2(feats).cpu().numpy()
        total_aesthetic_score.append(aesthetic_score.mean())

        clip_score = measure_CLIP_similarity(gen_images, prompt, ref_model, ref_clip_preprocess, ref_tokenizer, device).cpu().numpy()
        mean_clip_score = clip_score.mean()
        total_mean_clip_score.append(mean_clip_score)
        metadata = {'CLIP': clip_score, 'prompt': prompt, 'Aesthetic Score': aesthetic_score}

        mean_clip = sum(total_mean_clip_score) / len(total_mean_clip_score)
        mean_aesthetic_score = sum(total_aesthetic_score) / len(total_aesthetic_score)

        if args.dataset == 'COCO':
            progress_bar.set_postfix({"mean Aesthetic Score": f"{mean_aesthetic_score:.4f}",  "mean CLIP Score": f"{mean_clip:.4f}"})
        else:
            urls = dataset.iloc[i]['urls']
            gen_feats = get_SSCD_feature(gen_images, sim_model, device)
            gt_images = []
            for url in urls:
                try:
                    response = session.get(url, verify=False, headers=headers)
                    response.raise_for_status()  # 상태 코드가 200이 아니면 예외 발생
                    image = Image.open(BytesIO(response.content)).convert('RGB')
                    gt_images.append(image)
                except requests.exceptions.RequestException as e:
                    print(f"Failed to retrieve image from {url}: {e}")
                    breakpoint()
            gt_feats = get_SSCD_feature(gt_images, sim_model, device)
            
            sim = torch.mm(gen_feats, gt_feats.T).cpu().numpy()
            max_sim = sim.max()
            top3_mean = np.sort(sim.max(axis=1))[-3:].mean()
            above_0_5 = (np.sort(sim.max(axis=1)) > 0.5).mean()
            total_max_sim.append(max_sim)
            total_top3_mean.append(top3_mean)
            total_above_0_5.append(above_0_5)
            metadata['SSCD'] = sim
            mean_sscd = sum(total_max_sim) / len(total_max_sim)
            mean_top3_sscd = sum(total_top3_mean)/ len(total_top3_mean)
            mean_sscd_above_0_5 = sum(total_above_0_5)/ len(total_above_0_5)
            progress_bar.set_postfix({"mean SSCD": f"{mean_sscd:.4f}","Top3 SSCD": f"{mean_top3_sscd:.4f}","SSCD>0.5": f"{mean_sscd_above_0_5:.4f}","mean Aesthetic Score": f"{mean_aesthetic_score:.4f}",  "mean CLIP Score": f"{mean_clip:.4f}"})

        with open(f'{args.save_path}/img_{str(i).zfill(6)}/metadata.pickle', 'wb') as f: pickle.dump(metadata, f)

    if args.dataset == 'COCO':
        res = {'CLIP': mean_clip, 'Aesthetic Score': mean_aesthetic_score}
    else:
        res = {'SSCD': mean_sscd, 'CLIP': mean_clip, 'Top 3 SSCD': mean_top3_sscd, "SSCD > 0.5": mean_sscd_above_0_5, 'Aesthetic Score': mean_aesthetic_score}
    with open(f'{args.save_path}/result.pickle', 'wb') as f: pickle.dump(res, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="diffusion memorization")
    parser.add_argument("--image_length", default=512, type=int)
    parser.add_argument("--num_images_per_prompt", default=10, type=int)
    parser.add_argument("--guidance_scale", default=7.5, type=float)
    parser.add_argument("--num_inference_steps", default=50, type=int)
    parser.add_argument("--gen_seed", default=0, type=int)
    parser.add_argument("--save_path", default='exps_mitigation', type=str)
    parser.add_argument("--dataset", default='SD1', type=str)

    # mitigation strategy
    # baseline
    parser.add_argument(
        "--prompt_aug_style", default=None
    )  # rand_numb_add, rand_word_add, rand_word_repeat
    parser.add_argument("--repeat_num", default=1, type=int)

    # ours
    parser.add_argument("--optim_target_steps", default=0, type=int)
    parser.add_argument("--optim_lr", default=0.05, type=float)
    parser.add_argument("--optim_iters", default=10, type=int)
    parser.add_argument("--optim_target_loss", default=None, type=float)

    parser.add_argument("--rescale_attention", default=None, type=float)

    parser.add_argument('--load_path', type=str, default=None)

    args = parser.parse_args()

    main(args)
