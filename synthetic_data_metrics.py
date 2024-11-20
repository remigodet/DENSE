# single file to compute all synthetic data metrics !
from torch.nn import KLDivLoss
import torch
import precision_recall
from helpers.utils import loader_to_array
from torchmetrics.image.fid import FrechetInceptionDistance
from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore')

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
    
    # FID (taken from torchmetrics)
    features = 64
    fid = FrechetInceptionDistance(feature=features)
    for images in tqdm(original_dataloader, desc="FID step 1"):
        images = images * 5 
        images = images.type(torch.uint8)
        # print(images.shape) #(.,3,32,32)
        fid.update(images, real=True)
    for images in tqdm(synthetic_dataloader, desc="FID step 2"):
        images = images * 5 
        images = images.type(torch.uint8)
        fid.update(images, real=False)
    fid_res = fid.compute()
    metrics.append((f"FID on {features} features", fid_res.item()))
    
    # precision recall   (taken from google gan metrics)
    precision, recall = precision_recall.compute_prd_from_embedding(loader_to_array(synthetic_dataloader),
                                                   loader_to_array(original_dataloader),
                                                   num_runs=10)
    metrics.append(("PRD F score (beta=8)", precision_recall.prd_to_max_f_beta_pair(precision, recall))) # can tune paramter beta=8
    # precision_recall.plot(list(zip(precision, recall)), out_path="test_prd_curve") #doesnt work !
    
    
    if args:
        plt.plot(precision, recall) 
        plt.savefig(f"run/{args.run_name}/figures/prd_curve")
        plt.close()
    
    
    return metrics

def compute_metrics_federated(synthetic_dataloader, client_loaders, args=None):
    '''
    Compute all metrics implemented for 1 synthetic dataset to a list of client synthetic datasets
    '''
    res = []
    for client_loader in client_loaders:
        res.append(compute_metrics_loaders(synthetic_dataloader, client_loader, args=None))
    return res


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
    
    