import torch
"""
def calculate(noisy, sen, in_arr, out_arr):
    noisy_bg = torch.cat((noisy[:,:,:20,:20],
                        noisy[:,:,-20:,:20],
                        noisy[:,:,:20,-20:],
                        noisy[:,:,-20:,-20:]),3)
    
    noisy_in = noisy_bg[in_arr,...]
    noisy_out = noisy_bg[out_arr,...]
    
    in_var = torch.var(noisy_in)
    out_var = torch.var(noisy_out)
    cov =  torch.mean((noisy_in - noisy_in.mean())*(noisy_out - noisy_out.mean()))
    #print(cov)
    alpha = - cov / torch.sqrt(in_var * out_var - cov ** 2 + 1e-12)
    
    if alpha == 0 :
        beta = 1
    else:
        beta = in_var / torch.sqrt(in_var * out_var - cov ** 2 + 1e-12)
    #print(torch.abs(alpha).max())
    return alpha, beta





"""


def calculate(noisy, sen, in_arr, out_arr):
    noisy_bg = torch.cat((noisy[:,:,:20,:20],
                        noisy[:,:,-20:,:20],
                        noisy[:,:,:20,-20:],
                        noisy[:,:,-20:,-20:]),3)
    
    noisy_mean = noisy_bg.mean([1,2,3])
    
    cov_map = torch.zeros_like(sen[0:3,...])
    
    for i in in_arr:
        sen_i = sen[i,...] 
        for j in out_arr:
            sen_j = sen[j,...] 
            cov11 = torch.var(noisy_bg[i,...])
            cov12 = torch.mean((noisy_bg[i,...] - noisy_mean[i])*(noisy_bg[j,...] - noisy_mean[j]))
            cov22 = torch.var(noisy_bg[j,...])
            cov_map[0,...] = cov_map[0,...] + cov11 * torch.abs(sen_i) * torch.abs(sen_j)
            cov_map[1,...] = cov_map[1,...] + cov12 * torch.abs(sen_i) * torch.abs(sen_j)
            cov_map[2,...] = cov_map[2,...] + cov22 * torch.abs(sen_i) * torch.abs(sen_j)
    
    alpha = - cov_map[1,...] / torch.sqrt(cov_map[0,...] * cov_map[2,...] - cov_map[1,...] ** 2 + 1e-12)

    if alpha.max() == 0:
        beta = torch.ones_like(alpha)

    else:
        beta = cov_map[0,...] / torch.sqrt(cov_map[0,...] * cov_map[2,...] - cov_map[1,...] ** 2 + 1e-12)

    print(torch.abs(alpha).max(),end = ' ')
    a = noisy_bg[in_arr,...]
    b = noisy_bg[out_arr,...]
    cor =  torch.mean((a - a.mean())*(b - b.mean()))
    print(cor / torch.sqrt(torch.var(a) * torch.var(b) - torch.mean((a - a.mean())*(b - b.mean())) ** 2) )
    return alpha, beta