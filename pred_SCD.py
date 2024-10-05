import os
import time
import numpy as np
import torch.autograd
from skimage import io, exposure
from torch.nn import functional as F
from torch.utils.data import DataLoader
from datasets import RS_ST as RS
from models.SCanNet import SCanNet as Net


class PredOptions:
    def __init__(self):
        self.pred_batch_size = 1
        self.test_dir = './static/'
        self.pred_dir = './PRED_DIR/'
        self.chkpt_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                       'checkpoints/SCanNet_psd_50e_mIoU70.47_Sek18.38_Fscd58.14_OA85.93_Loss0.59.pth')


def predict_main(pred_batch_size, test_dir, pred_dir, chkpt_path):
    begin_time = time.time()

    opt = PredOptions()
    opt.pred_batch_size = pred_batch_size
    opt.test_dir = test_dir
    opt.pred_dir = pred_dir
    opt.chkpt_path = chkpt_path

    net = Net(3, RS.num_classes).cuda()
    checkpoint = torch.load(opt.chkpt_path)
    state_dict = checkpoint['model_state_dict']
    net.load_state_dict(state_dict)
    net.eval()

    test_set = RS.Data_test(opt.test_dir)
    test_loader = DataLoader(test_set, batch_size=opt.pred_batch_size)
    predict(net, test_set, test_loader, opt.pred_dir)

    time_use = time.time() - begin_time
    print('Total time: %.2fs' % time_use)


def predict(net, pred_set, pred_loader, pred_dir, flip=False, index_map=False, intermediate=False):
    # pred_A_dir_rgb = os.path.join(pred_dir, 'im1')
    # pred_B_dir_rgb = os.path.join(pred_dir, 'im2')
    pred_A_dir_rgb = os.path.join(pred_dir)
    pred_B_dir_rgb = os.path.join(pred_dir)
    os.makedirs(pred_A_dir_rgb, exist_ok=True)
    os.makedirs(pred_B_dir_rgb, exist_ok=True)

    for vi, data in enumerate(pred_loader):
        imgs_A, imgs_B = data
        imgs_A = imgs_A.cuda().float()
        imgs_B = imgs_B.cuda().float()
        mask_name = pred_set.get_mask_name(vi)

        with torch.no_grad():
            out_change, outputs_A, outputs_B = net(imgs_A, imgs_B)
            out_change = F.sigmoid(out_change)

        outputs_A = outputs_A.cpu().detach()
        outputs_B = outputs_B.cpu().detach()
        pred_A = torch.argmax(outputs_A, dim=1).squeeze().numpy().astype(np.uint8)
        pred_B = torch.argmax(outputs_B, dim=1).squeeze().numpy().astype(np.uint8)

        pred_A_path = os.path.join(pred_A_dir_rgb, mask_name)
        pred_B_path = os.path.join(pred_B_dir_rgb, mask_name)
        io.imsave(pred_A_path, RS.Index2Color(pred_A))
        io.imsave(pred_B_path, RS.Index2Color(pred_B))
        # print(imgs_A.shape)  # Xem kích thước của imgs_A
        # print(outputs_A.shape)  # Xem kích thước của outputs_A
        # print(pred_A_path)

# Đoạn mã Flask của bạn
