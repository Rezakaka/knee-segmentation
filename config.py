#please cite the following paper:
#Kakavand, R., Palizi, M., Tahghighi, P. et al. 
#Integration of Swin UNETR and statistical shape modeling for a semi-automated segmentation of the knee and biomechanical modeling of articular cartilage. 
#Sci Rep 14, 2748 (2024). https://doi.org/10.1038/s41598-024-52548-9
import torch
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu';
BATCH_SIZE = 2;
VIRTUAL_BATCH_SIZE = 4;
LEARNING_RATE = 1e-3;
EPOCHS = 1000;
CROP_SIZE_D = 96;
CROP_SIZE_H = 96;
CROP_SIZE_L = 96;

RESIZE_SIZE_D = 160;
RESIZE_SIZE_H = 128;
RESIZE_SIZE_L = 128;

FOLDS = 5;
EARLY_STOPPING_TOLERANCE = 10;
NUM_WORKERS = 4;
DBG = False;