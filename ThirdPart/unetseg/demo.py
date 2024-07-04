l'simport os, time
from operator import add
import numpy as np
from glob import glob
import cv2
from tqdm import tqdm
import imageio
from PIL import Image
import torch
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, precision_score, recall_score

from model import build_unet
from utils import create_dir, seeding
from torch.utils.data import DataLoader
import torch.nn as nn
import torchvision.transforms as transforms
from data import DriveDataset2 ,DriveDataset



def mask_parse(mask):
    mask = np.expand_dims(mask, axis=-1)    ## (512, 512, 1)
    mask = np.concatenate([mask, mask, mask], axis=-1)  ## (512, 512, 3)
    return mask


if __name__ == "__main__":
    # process_gif()

    """ Seeding """
    seeding(42)

    """ Folders """
    create_dir("results")

    """ Load dataset """
    test_x = sorted(glob("../../../../data/DRIVE/test/images/images/*"))
    test_y = sorted(glob("../../../../data/DRIVE/test/1st_manual/tifimg/*"))

    img_transform = transforms.Compose([
       transforms.Resize((512, 512)), 
       transforms.ToTensor(),
    #    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    mask_transform = transforms.Compose([
       transforms.Resize((512, 512)), 
       transforms.ToTensor(),
    #    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    train_dataset = DriveDataset2(test_x, test_y,img_transform,mask_transform)


    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=2
    )

    """ Hyperparameters """
    H = 512
    W = 512
    size = (W, H)
    checkpoint_path = "./checkpoint.pth"

    """ Load the checkpoint """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = build_unet()
    model = model.to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    metrics_score = [0.0, 0.0, 0.0, 0.0, 0.0]
    time_taken = []

    for i,(img,mask) in tqdm(enumerate(train_loader)):
        # x = img.to(device)*0.5+1.0
        x = img.to(device)
        print(x.shape,'....')
        x = x[:,[2,1,0],:,:]
        y = mask.to(device)

        with torch.no_grad():
            """ Prediction and Calculating FPS """
            start_time = time.time()
            pred_y = model(x)
           
            pred_y = torch.sigmoid(pred_y)
            total_time = time.time() - start_time
            time_taken.append(total_time)


            pred_y = pred_y[0].cpu().numpy()        ## (1, 512, 512)
            pred_y = np.squeeze(pred_y, axis=0)     ## (512, 512)
            pred_y = pred_y > 0.5
            pred_y = np.array(pred_y, dtype=np.uint8)

        """ Saving masks """
        ori_mask = mask_parse(mask)
        pred_y = mask_parse(pred_y)
        line = np.ones((size[1], 10, 3)) * 128

        # oriimg = x[0].cpu().numpy()*2.0-1.0
        oriimg = x[0].cpu().numpy()
        # oriimg = img[0].numpy()*2.0-1.0
        oriimg = np.transpose(oriimg,(1,2,0))
        oriimg = (oriimg*255).astype(np.uint8)[:,:,[2,1,0]]
        # oriimg = cv2.cvtColor(oriimg,cv2.COLOR_RGB2BGR)*255
        print(oriimg.shape,pred_y.shape,np.min(oriimg),np.min(pred_y),np.max(oriimg),np.max(pred_y))
        cat_images = np.concatenate(
            [oriimg,line, line, pred_y * 255], axis=1
        )
        print(i)
        cv2.imwrite(f"results/{i}.png", cat_images)


