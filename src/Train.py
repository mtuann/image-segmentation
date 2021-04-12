import os
import sys
import argparse
import tqdm

from torch.utils.data import DataLoader, random_split
from Dataset import *

from UNet import *
from torch import optim

from datetime import datetime
import random


def getTimeStr():
    now = datetime.now() 
    return now.strftime("%Y%m%d__%H%M%S")
#     return now.strftime("%m/%d/%Y, %H:%M:%S")


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=5,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=1,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.0001,
                        help='Learning rate', dest='lr')
    parser.add_argument('-f', '--load', dest='load', type=str, default=False,
                        help='Load model from a .pth file')
    parser.add_argument('-s', '--scale', dest='scale', type=float, default=0.5,
                        help='Downscaling factor of the images')
    parser.add_argument('-v', '--validation', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')

    return parser.parse_args()


dir_img = './../data/imgs/'
dir_mask = './../data/masks/'
dir_checkpoints = './../checkpoints/'

def CriticIoU(y_pred, masks, threshold=0.5):
    '''compute mean IoU exaclty using prediction and target'''
    y_pred, y_true = (y_pred >= threshold).float(), (masks >= threshold).float()
    intersection = (y_true * y_pred).sum((2, 3))
    union = y_true.sum((2, 3)) + y_pred.sum((2, 3)) - intersection
    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou.mean()


def train_net(net, device, epochs=5, batch_size=50, lr=0.001, val_percent=0.2, save_cp=True, img_scale=1, model_path=None):
    
    def train_one_epoch(epoch, net, train_loader, optimizer, is_train=True):
        # lap qua all dataset
        if is_train:
            net.train()
        else:
            net.eval()

        pbar = tqdm.tqdm(enumerate(train_loader), total=len(train_loader))

        epoch_loss = 0
        
        epoch_loss_arr = []
        
        for step, batch in pbar:
            # lap qua tung batch
            imgs = batch['image']
            true_masks = batch['mask']
            imgs = imgs.to(device=device, dtype=torch.float32)

            mask_type = torch.float32 if net.n_classes == 1 else torch.long
            true_masks = true_masks.to(device=device, dtype=mask_type)

            if is_train:
                masks_pred = net(imgs)
                loss = criterion(masks_pred, true_masks)
                
                iou = CriticIoU(torch.sigmoid(masks_pred), true_masks)
                
                epoch_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_value_(net.parameters(), 0.1)
                optimizer.step()
                
            else:
                with torch.no_grad():
                    masks_pred = net(imgs)
                    loss = criterion(masks_pred, true_masks)
                    iou = CriticIoU(torch.sigmoid(masks_pred), true_masks)
                    epoch_loss += loss.item()
            
            epoch_loss_arr.append(loss.item())
            if is_train:
                pbar.set_description(f"train epoch {epoch + 1} step {step + 1} batch loss: {loss:.4f} iou: {iou:.4f} epoch loss: {np.array(epoch_loss_arr).mean():.4f}")
            else:
                pbar.set_description(f"eval epoch {epoch + 1} step {step + 1} batch loss: {loss:.4f} iou: {iou:.4f} epoch loss: {np.array(epoch_loss_arr).mean():.4f}")
                
        return np.array(epoch_loss_arr).mean()
    
    dataset = BasicDataset(dir_img, dir_mask, img_scale)
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    print("n_val: {} n_train: {}".format(n_val, n_train))
    train, val = random_split(dataset, [n_train, n_val]) # , generator=torch.Generator().manual_seed(42)
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True, drop_last=True)
    
    optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min' if net.n_classes > 1 else 'max', patience=2)
    
    if model_path is not None:
        rel_path = f'{dir_checkpoints}/{model_path}'
#         cp_epoch3__04122021__111930__0.06.pth
        net.load_state_dict(torch.load(rel_path))
        print("Done load model path: {}".format(rel_path))
        
    
    if net.n_classes > 1:
        criterion = nn.CrossEntropyLoss()
    else:
        print("n_class = 1 use BCEWithLogitsLoss")
        
    criterion = nn.BCEWithLogitsLoss()
                                 
    best_eval_iou = 0.0
    
    for epoch in range(epochs):
        train_one_epoch(epoch, net, train_loader, optimizer, is_train=True)
        eval_iou = train_one_epoch(epoch, net, val_loader, optimizer, is_train=False)
        
        if best_eval_iou < eval_iou:
            best_eval_iou = eval_iou
            
            # Save check_point
            if save_cp:
                try:
                    os.mkdir(dir_checkpoints)
                    logging.info('Created checkpoint directory')
                except OSError:
                    pass
                
                torch.save(net.state_dict(),
                           dir_checkpoints + f'cp_epoch{epoch + 1}__{getTimeStr()}__{best_eval_iou:.2f}.pth')
                
                logging.info(f'Checkpoint {epoch + 1} iou: {best_eval_iou:.6f} saved !')

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enable = True
    
def main():
    seed_everything(42)
    
    unet = UNet(n_channels=3, n_classes=1, bilinear=True)
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cuda:0')
#     print(device)
    unet.to(device=device)
    train_net(net=unet, device=device, model_path="cp_epoch3__04122021__111930__0.06.pth")

if __name__=="__main__":
    
    main()