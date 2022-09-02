import torch
from dataloader import prepare_dataloader_vctk
import numpy as np

def test(args, model, step):    
    # Get dataset
    data_loader = prepare_dataloader_vctk(args.test_path, "test.txt", batch_size=16, shuffle=False) 
 
    # Get loss function
    Loss = model.get_criterion()

    # Evaluation
    mel_l_list = []
    d_l_list = []
    current_step = 0
    for i, batch in enumerate(data_loader):
        # Get Data
        id_ = batch["id"]
        sid, text, mel_target, D, log_D, f0, energy, \
            src_len, mel_len, max_src_len, max_mel_len = model.parse_batch(batch)
    
        with torch.no_grad():
            # Forward
            mel_output, _, _, log_duration_output, f0_output, energy_output, src_mask, mel_mask, out_mel_len = model(
                            text, src_len, mel_target, mel_len, D, None, None, max_src_len, max_mel_len)
            # Cal Loss
            mel_loss, d_loss, _, _ = Loss(mel_output,  mel_target, 
                    log_duration_output, log_D, f0_output, f0, energy_output, energy, src_len, mel_len)

            # Logger
            m_l = mel_loss.item()
            d_l = d_loss.item()

            mel_l_list.append(m_l)
            d_l_list.append(d_l)

        current_step += 1            
    
    mel_l = sum(mel_l_list) / len(mel_l_list)
    d_l = sum(d_l_list) / len(d_l_list)

    return mel_l, d_l