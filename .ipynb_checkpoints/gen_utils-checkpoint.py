import numpy as np
import torch
import torch.nn.functional as F
import random
from utils import get_init_text, update_token_mask
import time
from tqdm import tqdm
import cv2
import numpy as np

import time
import pickle
from tqdm import tqdm

def get_SD_score(pipe, top_texts, SD_temperature = 0.1, SD_prefix = None, approx_num = 1):
    if SD_prefix != None:
        for idx in range(len(top_texts)):
            top_texts[idx] = f'{SD_prefix} {top_texts[idx]}'
            
    pipe_text_uncond = pipe.encode_prompt('', 'cuda', 1, False)[0]
    num_latents_per_prompt = approx_num
    num_batch_prompt = 200//approx_num
    latents = pipe.prepare_latents(num_latents_per_prompt,4,512,512,torch.float16,'cuda',None)
    latents = latents.repeat(num_batch_prompt,1,1,1)

    target_probs = []

    with torch.no_grad():
        uncond = pipe.unet(latents, 999, pipe_text_uncond.repeat(len(latents),1,1)).sample
        for text_iter in range(len(top_texts)//num_batch_prompt):
            temp_text = top_texts[text_iter*num_batch_prompt:(text_iter+1)*num_batch_prompt]
            temp_text = pipe.encode_prompt(temp_text, 'cuda', num_latents_per_prompt, False)[0]
            cond = pipe.unet(latents, 999, temp_text).sample
            target_probs += (abs(cond-uncond)**2).sum(axis=[1,2,3]).reshape(-1,num_latents_per_prompt).mean(axis=-1).tolist()
    target_probs = torch.tensor(target_probs).cuda()
    target_probs = target_probs.unsqueeze(0)
    target_probs_logit = F.softmax(target_probs/SD_temperature, dim = -1)

    return target_probs_logit, target_probs



def generate_step(out, gen_idx,  temperature=None, top_k=0, sample=False, return_list=True):
    logits = out[:, gen_idx]
    if temperature is not None:
        logits = logits / temperature
    if top_k > 0:
        kth_vals, kth_idx = logits.topk(top_k, dim=-1)
        dist = torch.distributions.categorical.Categorical(logits=kth_vals)
        idx = kth_idx.gather(dim=1, index=dist.sample().unsqueeze(-1)).squeeze(-1)
    elif sample:
        dist = torch.distributions.categorical.Categorical(logits=logits)
        idx = dist.sample().squeeze(-1)
    else:
        idx = torch.argmax(logits, dim=-1)
    return idx.tolist() if return_list else idx

def generate_caption_step(out, gen_idx, mask, temperature=None, top_k=100, banned_tokens = None, include_id = None):
    logits = out[:, gen_idx]
    if banned_tokens != None:
        logits[0][banned_tokens] = -1e+5 * torch.ones(len(banned_tokens)).type(logits.dtype).to(logits.device)
        logits[0][torch.where(mask[0] == 0)[0]] = -1e+5 * torch.ones(len(torch.where(mask[0] == 0)[0])).type(logits.dtype).to(logits.device)
    if temperature is not None:
        logits = logits / temperature
    probs = F.softmax(logits, dim=-1)
    top_k_probs, top_k_ids = probs.topk(top_k, dim=-1)
    if include_id != None:
        include_id = [id for id in include_id if id not in top_k_ids[0].tolist()]
        if len(include_id) > 0:
            top_k_ids[0][-len(include_id):] = torch.tensor(include_id).to('cuda')
            top_k_probs[0][-len(include_id):] = probs[0][include_id]
    
    return top_k_probs, top_k_ids




def MCMC_generation(model, tokenizer, token_mask, prompt, logger, pipe, max_len=15, top_k=0, temperature=None,max_iters=300,verbose=True, banned_tokens = None, init_token = None, SD_temperature = 0.1, SD_prefix = None, approx_num = 4):

    seed_len = len(tokenizer.encode(prompt))-1
    init = get_init_text(tokenizer, prompt, max_len, 1)
    if init_token != None:
        init = [init_token]
    
    SD_score_sequence = []
    inp = torch.tensor(init).to('cuda')
    gen_texts_list = []
    
    convert_list = torch.tensor([0] + torch.arange(seed_len,len(init[0])).tolist())
    ii = 0

    for ii in tqdm(range(max_iters)):
        kk = np.random.randint(0, max_len)
        token_mask = update_token_mask(tokenizer, token_mask, max_len, kk)
        prev_inp = inp.clone().detach()
        inp[:,seed_len + kk] = tokenizer.mask_token_id
        inp_ = inp.clone().detach()
        out = model(inp).logits
        
        if init_token != None:
            include_id = list(set([init_token[seed_len+kk], prev_inp[0][seed_len+kk].item()]))
        else:
            include_id = list(set([tokenizer.mask_token_id, prev_inp[0][seed_len+kk].item()]))
        
        probs, idxs = generate_caption_step(out,gen_idx=seed_len + kk,mask=token_mask, top_k=top_k, temperature=temperature, banned_tokens = banned_tokens, include_id = include_id)
        topk_inp = inp_.unsqueeze(1).repeat(1,top_k,1)
        idxs_ = (idxs * token_mask[0][idxs]).long()
        topk_inp[:,:,kk + seed_len] = idxs_ 
        topk_inp_batch = topk_inp.view(-1,topk_inp.shape[-1])
        batch_text_list= tokenizer.batch_decode(topk_inp_batch , skip_special_tokens=True)
        batch_text_list_for_SD = tokenizer.batch_decode(topk_inp_batch[:, convert_list], skip_special_tokens=True)
        
        SD_score, SD_ref = get_SD_score(pipe, batch_text_list_for_SD, SD_temperature = SD_temperature, SD_prefix = SD_prefix, approx_num = approx_num)
        
        sel_SD_id = torch.multinomial(SD_score[0], num_samples = 1).view(-1,1)

        inp[:,seed_len + kk] = idxs_.gather(1, sel_SD_id).squeeze(-1)
        current_SD_score = SD_ref.gather(1, sel_SD_id).squeeze(-1)
        current_SD_score = current_SD_score.cpu().detach().numpy().tolist()

        for_print = tokenizer.batch_decode(inp)
        
        gen_texts_list += batch_text_list_for_SD
        SD_score_sequence += SD_ref[0].tolist()

    return gen_texts_list, SD_score_sequence


def MCMC_aug(model, tokenizer, token_mask, prompt, logger, pipe, max_len=15, top_k=0, temperature=None,max_iters=300, verbose=True, banned_tokens = None, init_token = None, threshold = 0, SD_temperature = 0.1, SD_prefix = None, approx_num = 4):

    seed_len = len(tokenizer.encode(prompt))-1
    
    SD_score_sequence = []
    gen_texts_list = []
    
    prev_score = 0

    convert_list = torch.tensor([0] + torch.arange(seed_len,len(init_token)).tolist())

    total_texts = []
    total_scores = []

    for init_kk in range(0, max_len):
        print(f'MCMC chain starting from {init_kk+1}th/{max_len} token started..')
        init = [init_token]
        inp = torch.tensor(init).to('cuda')
        early_stop_cnt = 0
        for ii in tqdm(range(max_iters)):
            if ii == 0:
                kk = init_kk
            else:
                kk = np.random.randint(0, max_len)
            token_mask = update_token_mask(tokenizer, token_mask, max_len, kk)
            prev_inp = inp.clone().detach()
            inp[:,seed_len + kk] = tokenizer.mask_token_id
            inp_ = inp.clone().detach()
            out = model(inp).logits
            include_id = list(set([init_token[seed_len+kk], prev_inp[0][seed_len+kk].item()]))
            probs, idxs = generate_caption_step(out,gen_idx=seed_len + kk,mask=token_mask, top_k=top_k, temperature=temperature, banned_tokens = banned_tokens, include_id = include_id)
            topk_inp = inp_.unsqueeze(1).repeat(1,top_k,1)
            idxs_ = (idxs * token_mask[0][idxs]).long()
            topk_inp[:,:,kk + seed_len] = idxs_ 
            topk_inp_batch = topk_inp.view(-1,topk_inp.shape[-1])
            batch_text_list= tokenizer.batch_decode(topk_inp_batch , skip_special_tokens=True)
            batch_text_list_for_SD = tokenizer.batch_decode(topk_inp_batch[:, convert_list], skip_special_tokens=True)
            SD_score, SD_ref = get_SD_score(pipe, batch_text_list_for_SD, SD_temperature = SD_temperature, SD_prefix = SD_prefix)

            sel_SD_id = torch.multinomial(SD_score[0], num_samples = 1).view(-1,1)
    
            prev_score = SD_ref[0][sel_SD_id.item()].item()
            inp[:,seed_len + kk] = idxs_.gather(1, sel_SD_id).squeeze(-1)
            current_SD_score = SD_ref.gather(1,sel_SD_id).squeeze(-1)
            SD_score_sequence_batch = current_SD_score.cpu().detach().numpy().tolist()
            
            cur_text_batch= tokenizer.batch_decode(inp,skip_special_tokens=True)
    
            for_print_batch = tokenizer.batch_decode(inp)    
            
            total_texts.append(topk_inp_batch[:, convert_list].cpu())
            total_scores += SD_ref[0].tolist()

            if SD_ref[0][sel_SD_id.item()].item() < threshold:
                early_stop_cnt +=1
            else:
                early_stop_cnt = 0
            if early_stop_cnt == 3:
                break
    
    total_texts = torch.cat(total_texts)

    res_dict = {i: {} for i in range(len(convert_list)-1)}

    for text, score in zip(total_texts, total_scores):
        diff_num = (text != torch.tensor(init_token)[convert_list]).sum().item()
        prompt = tokenizer.decode(text, skip_special_tokens = True)
        if prompt not in res_dict[diff_num].keys():
            res_dict[diff_num][prompt] = [score]
        else:
            res_dict[diff_num][prompt].append(score)

    for diff_num in res_dict.keys():
        prompts = list(res_dict[diff_num].keys())
        for prompt in prompts:
            res_dict[diff_num][prompt] = sum(res_dict[diff_num][prompt])/len(res_dict[diff_num][prompt])
            if res_dict[diff_num][prompt] < 5: del res_dict[diff_num][prompt]
    
    return res_dict, None


def generate_caption(model, tokenizer,token_mask,logger,pipe,
                     prompt="", max_len=15,
                     top_k=100, temperature=1.0, max_iter=500,
                     generate_order="random", init_sentence = None, threshold = 0, SD_temperature = 0.1, SD_prefix = None, approx_num = 4):
    
    start_time = time.time()
    
    with open('banned_tokens.pickle','rb') as f: banned_tokens = pickle.load(f)
    banned_tokens = list(set(banned_tokens))
    banned_tokens = [tokenizer.encode(token) for token in banned_tokens]
    banned_tokens = [token[1] for token in banned_tokens if len(token) == 3]

    if init_sentence != None:
        init_token = tokenizer.encode(f'{prompt} {init_sentence}')
        max_len = len(init_token) - len(tokenizer.encode(prompt))
    else:
        init_token = None

    if generate_order=="random":
        generate_texts, SD_scores = MCMC_generation(model, tokenizer, token_mask, prompt, logger, pipe, max_len=max_len, top_k=top_k, temperature=temperature, max_iters=max_iter, verbose=True, banned_tokens = banned_tokens, init_token = init_token, SD_temperature = SD_temperature, SD_prefix = SD_prefix, approx_num = approx_num)
    elif generate_order=="aug":
        generate_texts, SD_scores = MCMC_aug(model, tokenizer, token_mask, prompt, logger, pipe, max_len=max_len, top_k=top_k, temperature=temperature, max_iters=max_iter, verbose=True, banned_tokens = banned_tokens, init_token = init_token, threshold = threshold, SD_temperature = SD_temperature, SD_prefix = SD_prefix, approx_num = approx_num)

    logger.info("Finished in %.3fs" % (time.time() - start_time))
    
    return generate_texts, SD_scores