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
                                       'checkpoints/SCanNet_psd_50e_mIoU71.64_Sek19.16_Fscd63.17_OA87.37_Loss0.48_lastLandset256.pth')


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


# def predict(net, pred_set, pred_loader, pred_dir, flip=False, index_map=False, intermediate=False):
#     # pred_A_dir_rgb = os.path.join(pred_dir, 'im1')
#     # pred_B_dir_rgb = os.path.join(pred_dir, 'im2')
#     pred_A_dir_rgb = os.path.join(pred_dir)
#     pred_B_dir_rgb = os.path.join(pred_dir)
#     os.makedirs(pred_A_dir_rgb, exist_ok=True)
#     os.makedirs(pred_B_dir_rgb, exist_ok=True)
#
#     for vi, data in enumerate(pred_loader):
#         imgs_A, imgs_B = data
#         imgs_A = imgs_A.cuda().float()
#         imgs_B = imgs_B.cuda().float()
#         mask_name = pred_set.get_mask_name(vi)
#
#         with torch.no_grad():
#             out_change, outputs_A, outputs_B = net(imgs_A, imgs_B)
#             out_change = F.sigmoid(out_change)
#
#         outputs_A = outputs_A.cpu().detach()
#         outputs_B = outputs_B.cpu().detach()
#         pred_A = torch.argmax(outputs_A, dim=1).squeeze().numpy().astype(np.uint8)
#         pred_B = torch.argmax(outputs_B, dim=1).squeeze().numpy().astype(np.uint8)
#
#         pred_A_path = os.path.join(pred_A_dir_rgb, mask_name)
#         pred_B_path = os.path.join(pred_B_dir_rgb, mask_name)
#         io.imsave(pred_A_path, RS.Index2Color(pred_A))
#         io.imsave(pred_B_path, RS.Index2Color(pred_B))

# For models with 3 outputs: 1 change map + 2 semantic maps.
# Parameters: flip->test time augmentation     index_map->"False" means rgb results      intermediate->whether to outputs the intermediate maps
def predict(net, pred_set, pred_loader, pred_dir, flip=False, index_map=False, intermediate=False):
    pred_A_dir_rgb = os.path.join(pred_dir, 'im1_rgb')
    pred_B_dir_rgb = os.path.join(pred_dir, 'im2_rgb')
    if not os.path.exists(pred_A_dir_rgb): os.makedirs(pred_A_dir_rgb)
    if not os.path.exists(pred_B_dir_rgb): os.makedirs(pred_B_dir_rgb)
    if index_map:
        pred_A_dir = os.path.join(pred_dir, 'im1')
        pred_B_dir = os.path.join(pred_dir, 'im2')
        if not os.path.exists(pred_A_dir): os.makedirs(pred_A_dir)
        if not os.path.exists(pred_B_dir): os.makedirs(pred_B_dir)
    if intermediate:
        pred_mA_dir = os.path.join(pred_dir, 'im1_semantic')
        pred_mB_dir = os.path.join(pred_dir, 'im2_semantic')
        pred_change_dir = os.path.join(pred_dir, 'change')
        if not os.path.exists(pred_mA_dir): os.makedirs(pred_mA_dir)
        if not os.path.exists(pred_mB_dir): os.makedirs(pred_mB_dir)
        if not os.path.exists(pred_change_dir): os.makedirs(pred_change_dir)

    for vi, data in enumerate(pred_loader):
        imgs_A, imgs_B = data
        # imgs = torch.cat([imgs_A, imgs_B], 1)
        imgs_A = imgs_A.cuda().float()
        imgs_B = imgs_B.cuda().float()
        mask_name = pred_set.get_mask_name(vi)
        with torch.no_grad():
            out_change, outputs_A, outputs_B = net(imgs_A, imgs_B)  # ,aux
            out_change = F.sigmoid(out_change)
        if flip:
            outputs_A = F.softmax(outputs_A, dim=1)
            outputs_B = F.softmax(outputs_B, dim=1)

            imgs_A_v = torch.flip(imgs_A, [2])
            imgs_B_v = torch.flip(imgs_B, [2])
            out_change_v, outputs_A_v, outputs_B_v = net(imgs_A_v, imgs_B_v)
            outputs_A_v = torch.flip(outputs_A_v, [2])
            outputs_B_v = torch.flip(outputs_B_v, [2])
            out_change_v = torch.flip(out_change_v, [2])
            outputs_A += F.softmax(outputs_A_v, dim=1)
            outputs_B += F.softmax(outputs_B_v, dim=1)
            out_change += F.sigmoid(out_change_v)

            imgs_A_h = torch.flip(imgs_A, [3])
            imgs_B_h = torch.flip(imgs_B, [3])
            out_change_h, outputs_A_h, outputs_B_h = net(imgs_A_h, imgs_B_h)
            outputs_A_h = torch.flip(outputs_A_h, [3])
            outputs_B_h = torch.flip(outputs_B_h, [3])
            out_change_h = torch.flip(out_change_h, [3])
            outputs_A += F.softmax(outputs_A_h, dim=1)
            outputs_B += F.softmax(outputs_B_h, dim=1)
            out_change += F.sigmoid(out_change_h)

            imgs_A_hv = torch.flip(imgs_A, [2, 3])
            imgs_B_hv = torch.flip(imgs_B, [2, 3])
            out_change_hv, outputs_A_hv, outputs_B_hv = net(imgs_A_hv, imgs_B_hv)
            outputs_A_hv = torch.flip(outputs_A_hv, [2, 3])
            outputs_B_hv = torch.flip(outputs_B_hv, [2, 3])
            out_change_hv = torch.flip(out_change_hv, [2, 3])
            outputs_A += F.softmax(outputs_A_hv, dim=1)
            outputs_B += F.softmax(outputs_B_hv, dim=1)
            out_change += F.sigmoid(out_change_hv)
            out_change = out_change / 4

        outputs_A = outputs_A.cpu().detach()
        outputs_B = outputs_B.cpu().detach()
        change_mask = out_change.cpu().detach() > 0.5
        change_mask = change_mask.squeeze()
        pred_A = torch.argmax(outputs_A, dim=1).squeeze()
        pred_B = torch.argmax(outputs_B, dim=1).squeeze()

        if intermediate:
            pred_A_path = os.path.join(pred_mA_dir, mask_name)
            pred_B_path = os.path.join(pred_mB_dir, mask_name)
            pred_change_path = os.path.join(pred_change_dir, mask_name)
            io.imsave(pred_A_path, RS.Index2Color(pred_A.numpy()))
            io.imsave(pred_B_path, RS.Index2Color(pred_B.numpy()))
            change_map = exposure.rescale_intensity(change_mask.numpy(), 'image', 'dtype')
            io.imsave(pred_change_path, change_map)
        pred_A = (pred_A * change_mask.long()).numpy()
        pred_B = (pred_B * change_mask.long()).numpy()
        pred_A_path = os.path.join(pred_A_dir_rgb, mask_name)
        pred_B_path = os.path.join(pred_B_dir_rgb, mask_name)
        io.imsave(pred_A_path, RS.Index2Color(pred_A))
        io.imsave(pred_B_path, RS.Index2Color(pred_B))
        print(pred_A_path)
        if index_map:
            pred_A_path = os.path.join(pred_A_dir, mask_name)
            pred_B_path = os.path.join(pred_B_dir, mask_name)
            io.imsave(pred_A_path, pred_A.astype(np.uint8))
            io.imsave(pred_B_path, pred_B.astype(np.uint8))

        # For models that directly produce 2 SCD maps.
