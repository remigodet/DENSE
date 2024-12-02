# single file to compute all synthetic data metrics !
import torch
import numpy as np
import precision_recall
# from helpers.utils import loader_to_array
from torchmetrics.image.fid import FrechetInceptionDistance
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

# TODO finetune this 
METRIC_SAMPLING_LIM = 5000
# print("METRIC_LOOP_LIM", METRIC_LOOP_LIM)
# print("METRIC_SAMPLING_LIM", METRIC_SAMPLING_LIM)

def compute_metrics_loaders(synthetic_dataloader, original_dataloader, args=None):
    '''
    Returns every metric implemented as a list of tuples : (metric, value)
    '''
    metrics = []
    # TODO KL div NEED TO CLUSTER 
    # kldiv = KLDivLoss(reduce=False)
    # res = []
    # for (original, synthetic) in zip(synthetic_dataloader, original_dataloader): 
    #     print(original, synthetic)
    #     if type(original) == list :
    #             original = original[0] 
    #     original = original.cuda()   
    #     synthetic = synthetic.cuda()
              
    #     with torch.no_grad():
    #         res.append(kldiv(synthetic, original))
    # res = torch.mean(torch.stack(res))
    # metrics.append(("KLDiv", res))
    
    # =======================
    # FID (taken from torchmetrics)
    # =======================
    
    
    # hyperparams and cast to gpu 
    features = 64
    fid = FrechetInceptionDistance(feature=features).cuda()
    fid.inception.to(fid.device)
    # print("fid device", fid.device)
    
    i = 0
    # collect all feature representations 
    for images in original_dataloader:
        images = images[:METRIC_SAMPLING_LIM-i]
        i+=len(images)
        images = images * 255 # back in the range of uint integers
        images = images.type(torch.uint8)# no .cuda ! 
        images = images.cuda()
        fid.update(images, real=True)
        
        if i>=METRIC_SAMPLING_LIM : break
    i=0
    for images in synthetic_dataloader:
        images = images[:METRIC_SAMPLING_LIM-i]
        i+=len(images)
        images = images * 255 
        images = images.type(torch.uint8)
        images = images.cuda()
        fid.update(images, real=False)
        
        if i>=METRIC_SAMPLING_LIM : break
    # compute fid - may have more real data -> better distribution estimate
    fid_res = fid.compute()
    metrics.append((f"FID on {features} features", fid_res.item()))
    
    # =======================
    # precision recall   (taken from google gan metrics)
    # =======================
    
    # TODO loader_to_array might be too slow ...
    
    # original_data = loader_to_array(original_dataloader)
    # synthetic_data = loader_to_array(synthetic_dataloader)
    # # if n too low -> a lot of bias in the PRD curve 
    
    
    # using Inception features     
    synthetic_data = loader_to_array(fid.fake_features)
    original_data = loader_to_array(fid.real_features)
    
    n = min(len(synthetic_data), len(original_data)) # because we generate much less synthetic data -> it will skew the clusters 
    synthetic_data = synthetic_data[:n]
    original_data = original_data[:n]
    
    precision, recall = precision_recall.compute_prd_from_embedding(synthetic_data,
                                                                    original_data,
                                                                    num_runs=10)
    fmax_score, fmax_inv_score = precision_recall.prd_to_max_f_beta_pair(precision, recall)  # can tune paramter beta=8
    metrics.append(("PRD F score (beta=8)", fmax_score))
    metrics.append(("PRD 1/F score (beta=8)", fmax_inv_score))
    
    metrics.append(("precision", precision))
    metrics.append(("recall", recall))
    
    # precision_recall.plot(list(zip(precision, recall)), out_path="test_prd_curve") #doesn't work !  
    if args:
        plt.plot(precision, recall) 
        plt.xlim(0,1)
        plt.ylim(0,1)
        plt.xlabel('recall')
        plt.ylabel('precision')
        plt.savefig(f"run/{args.run_name}/figures/{args.cur_ep}_all_prd_curve")
        plt.close()
    return metrics

def compute_metrics_federated(synthetic_dataloader, client_loaders, args=None):
    '''
    Compute all metrics implemented for 1 synthetic dataset to a list of client synthetic datasets
    '''
    res = []
    i = 0
    for client_loader in client_loaders:
        i += 1
        # print(f'client {i}')
        res.append(compute_metrics_loaders(synthetic_dataloader, client_loader, args=None))
    return res

def loader_to_array(loader): 
    arr = []
    for images in loader:
        images = images.cpu().detach().numpy().reshape(images.shape[0], -1) # [bs, .*.*.]
        arr.append(images)
    return np.vstack(arr) #this may explode memory -> put some sampling or len reqs 

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    dataset_a = torch.ones(100,3,32,32)
    dataset_b = torch.ones(100,3,32,32)
    
    loader_a = torch.utils.data.DataLoader(dataset_a, batch_size=32,
                                              shuffle=False, num_workers=4)
    loader_b = torch.utils.data.DataLoader(dataset_b, batch_size=32,
                                              shuffle=False, num_workers=4)
    
    metrics = compute_metrics_loaders(loader_a, loader_b)
    for name, value in metrics:
        print(name, " ", value)
    
    