from utils import create_logger,set_seed
import os
import time
import argparse
import json
from PIL import Image
import torch

from gen_utils import generate_caption
from transformers import AutoModelForMaskedLM, AutoTokenizer
from diffusers import StableDiffusionPipeline, DDIMScheduler
import pickle
import pandas as pd
import torch.nn.functional as F
import numpy as np
from sim_utils import get_SSCD_feature, cluster_images


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=203)
    parser.add_argument("--batch_size", type=int, default=1, help = "Only supports batch_size=1 currently.")
    parser.add_argument("--device", type=str,
                        default='cuda',choices=['cuda','cpu'])
    parser.add_argument('--prompt',
                        default='Image of a',type=str)
    parser.add_argument('--order',
                        default='random',
                        nargs='?',
                        choices=['random', 'aug'],
                        help="Generation order of text")
    parser.add_argument('--samples_num',
                        default=100,type=int)
    parser.add_argument("--sentence_len", type=int, default=8)
    parser.add_argument("--approx_num", type=int, default=1)
    parser.add_argument("--candidate_k", type=int, default=200)
    parser.add_argument("--lm_temperature", type=float, default=0.1)
    parser.add_argument("--num_iterations", type=int, default=10, help="predefined iterations for Gibbs Sampling")
    parser.add_argument("--threshold", type=float, default=0)
    parser.add_argument("--SD_temperature", type=float, default=0.1)
    parser.add_argument("--lm_model", type=str, default='bert-base-uncased',
                        help="Path to language model") # bert,roberta
    parser.add_argument("--stop_words_path", type=str, default='stop_words.txt',
                        help="Path to stop_words.txt")
    parser.add_argument("--init_sentence", type=str, default=None)
    parser.add_argument("--SD_prefix", type=str, default=None)
    parser.add_argument("--dir_sim_model", type=str, default='sscd_disc_large.torchscript.pt')
    args = parser.parse_args()

    return args

def run_caption(args, lm_model, lm_tokenizer, token_mask, logger, pipe = None, sim_model = None, save_path = None):

    sample_id_list = list(range(args.samples_num))

    t_start = time.time()
    
    for sample_id in sample_id_list:
        logger.info(f"Sample {sample_id}: ")
        
        gen_texts, SD_scores = generate_caption(lm_model, lm_tokenizer, token_mask, logger,pipe, prompt=args.prompt, max_len=args.sentence_len, top_k=args.candidate_k, temperature=args.lm_temperature, max_iter=args.num_iterations, generate_order = args.order, init_sentence = args.init_sentence, threshold = args.threshold, SD_temperature = args.SD_temperature, SD_prefix = args.SD_prefix, approx_num = args.approx_num)

        if args.order == 'aug':
            res = gen_texts
            print(f'Total token length of token: {len(gen_texts.keys())}')
            for thres in [5,10,20]:
                print(f'Total number of prompts exceeding threshold {thres}')
                for key in gen_texts.keys(): print(f'Different word number {key}: {(np.array([item[1] for item in gen_texts[key].items()])>thres).sum()}')
        else:
            res = [(a,b) for (a,b) in zip(gen_texts, SD_scores)]
        with open(f'{save_path}/{str(sample_id).zfill(6)}.pickle','wb') as f:
            pickle.dump(res,f)

        if args.order == 'random':
            res = sorted(res, key = lambda x: x[1])
            best_res = res[-1]
            best_score = best_res[-1]
            temp_path = save_path + '/gen_images/' + (f'{best_score:.4f}').zfill(9)
            dup_num = 1

            while True:
                if os.path.exists(temp_path):
                    dup_num += 1
                    if dup_num == 2:
                        temp_path = f'{temp_path}_{dup_num}'
                    else:
                        temp_path = temp_path.split('_')
                        temp_path[-1] = str(dup_num)
                        temp_path = '_'.join(temp_path)
                else:
                    break
            
            os.makedirs(temp_path)
            imgs = []
            
            for i in range(50):
                img = pipe(best_res[0]).images[0]
                imgs.append(img)
            for img_idx, img in enumerate(imgs):
                img.save(f'{temp_path}/{img_idx}.png')
            f = open(f'{temp_path}/metadata.txt', 'w')
            f.write(f'Text is from {sample_id} iteration:')
            f.write(best_res[0])
            f.close()
            feats = get_SSCD_feature(imgs, sim_model, device)
            sim = torch.mm(feats, feats.T).cpu().numpy()
            sim_images = cluster_images(sim, 0.5, 3, sim.shape[0])
            
            if len(sim_images) > 0:
                for sel_idx, sel_indices in enumerate(sim_images):
                    temp_sel_path = sel_path = save_path + '/sel_images/' + (f'{best_score:.4f}').zfill(9) + f'/{sel_idx}'
                    os.makedirs(temp_sel_path)
                    for img_id in sel_indices:
                        imgs[img_id].save(f'{temp_sel_path}/{img_id}.png')

if __name__ == "__main__":
    args = get_args()
    
    set_seed(args.seed)
    save_path = 'exps_generation/demo_{}_{}_approx{}_len{}_topk{}_sdtemp_{}_sdth_{}'.format(
        time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()), args.order, args.approx_num, args.sentence_len,
        args.candidate_k, args.SD_temperature,
        args.threshold)
    os.makedirs(save_path)
    logger = create_logger(
        save_path, 'log.log')

    logger.info(f"Generating order:{args.order}")
    logger.info(args)

    # Load pre-trained model (weights)
    lm_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    lm_model = AutoModelForMaskedLM.from_pretrained('bert-base-uncased')
    lm_model.eval()
    lm_model = lm_model.to(args.device)
    
    model_id = "CompVis/stable-diffusion-v1-4"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to('cuda')
    
    device = "cuda"

    pipe.safety_checker = None
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to("cuda")

    sim_model = torch.jit.load(args.dir_sim_model).to('cuda')

    with open(args.stop_words_path,'r',encoding='utf-8') as stop_words_file:
        stop_words = stop_words_file.readlines()
        stop_words_ = [stop_word.rstrip('\n') for stop_word in stop_words]
        stop_ids = lm_tokenizer.convert_tokens_to_ids(stop_words_)
        token_mask = torch.ones((1,lm_tokenizer.vocab_size))
        for stop_id in stop_ids:
            token_mask[0,stop_id]=0
        token_mask = token_mask.to(args.device)

    with torch.no_grad():
        run_caption(args, lm_model, lm_tokenizer, token_mask, logger, pipe, sim_model, save_path)

