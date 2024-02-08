#please cite the following paper:
#Kakavand, R., Palizi, M., Tahghighi, P. et al. 
#Integration of Swin UNETR and statistical shape modeling for a semi-automated segmentation of the knee and biomechanical modeling of articular cartilage. 
#Sci Rep 14, 2748 (2024). https://doi.org/10.1038/s41598-024-52548-9
import numpy as np
import torch

#===============================================================
def window_center_adjustment(img, max_val):

    hist = np.histogram(img.ravel(), bins = int(max_val))[0];
    hist = hist / hist.sum();
    hist = np.cumsum(hist);

    hist_thresh = ((1-hist) < 5e-4);
    max_intensity = np.where(hist_thresh == True)[0][0];
    adjusted_img = img * (255/(max_intensity + 1e-4));
    adjusted_img = np.where(adjusted_img > 255, 255, adjusted_img).astype("uint8");

    return adjusted_img;
#===============================================================

#===============================================================
def dice_loss(logits, 
                true, 
                eps=1e-7, 
                sigmoid = False,
                arange_logits = False,
                spatial_dims = 2):

    if sigmoid is True:
        logits = torch.sigmoid(logits);
    
    if arange_logits is True:
        a = tuple(d+1 for d in range(1,spatial_dims+1));
        permutation = (0,) + a + (1,);
        logits = logits.permute(permutation);
    if logits.dim != true.dim:
        true = true.unsqueeze(dim = -1);

    dims = tuple(d+1 for d in range(spatial_dims+1));

    intersection = torch.sum(true * logits, dims);
    union = torch.sum(true + logits, dims);
    d_loss = torch.mean((2.0*intersection) / (union + eps));
    return 1-d_loss;
#===============================================================

#===============================================================
def calculate_metrics(cm):
    tn = cm[0][0];
    fn = cm[1][0];
    tp = cm[1][1];
    fp = cm[0][1];

    precision = tp / (tp+fp+1e-6);
    recall = tp / (tp + fn+1e-6);
    f1 = 2*precision*recall / (precision + recall+1e-6);
    return precision.item(), recall.item(), f1.item();
#===============================================================