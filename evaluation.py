import os
import cv2
import numpy as np
import argparse
from tqdm import tqdm
import shutil
import scipy.ndimage
from skimage.measure import label
import scipy.ndimage.morphology

def compute_mse_loss(pred, target, trimap):
    error_map = (pred - target) / 255.0
    loss = np.sum((error_map ** 2) * (trimap == 128)) / (np.sum(trimap == 128) + 1e-8)

    # # if test on whole image (Disitinctions-646), please uncomment this line
    # loss = loss = np.sum(error_map ** 2) / (pred.shape[0] * pred.shape[1])

    return loss


def compute_sad_loss(pred, target, trimap):
    error_map = np.abs((pred - target) / 255.0)
    loss = np.sum(error_map * (trimap == 128))

    # # if test on whole image (Disitinctions-646), please uncomment this line
    # loss = np.sum(error_map)

    return loss / 1000, np.sum(trimap == 128) / 1000

def evaluate(args):
    img_names = []
    mse_loss_unknown = []
    sad_loss_unknown = []
    grad_loss_unknown = []
    conn_loss_unknown = []


    bad_case = []

    for i, img in tqdm(enumerate(os.listdir(args.label_dir))):

        if not((os.path.isfile(os.path.join(args.pred_dir, img)) and
                os.path.isfile(os.path.join(args.label_dir, img)) and
                os.path.isfile(os.path.join(args.trimap_dir, img)))):
            print('[{}/{}] "{}" skipping'.format(i, len(os.listdir(args.label_dir)), img))
            continue

        pred = cv2.imread(os.path.join(args.pred_dir, img), 0).astype(np.float32)
        label = cv2.imread(os.path.join(args.label_dir, img), 0).astype(np.float32)
        trimap = cv2.imread(os.path.join(args.trimap_dir, img), 0).astype(np.float32)

        # calculate loss
        mse_loss_unknown_ = compute_mse_loss(pred, label, trimap)
        sad_loss_unknown_ = compute_sad_loss(pred, label, trimap)[0]


        # save for average
        img_names.append(img)

        mse_loss_unknown.append(mse_loss_unknown_)  # mean l2 loss per unknown pixel
        sad_loss_unknown.append(sad_loss_unknown_)  # l1 loss on unknown area

    print('* Unknown Region: MSE:', np.array(mse_loss_unknown).mean(), ' SAD:', np.array(sad_loss_unknown).mean())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred-dir', type=str, default='/home/yihan.hu/workdir/DiffusionMattingV2/predDIM_S/noise_1step_linear1_mat')
    parser.add_argument('--label-dir', type=str, default='/weka/users/yihan.hu/Matting/Composition-1k-testset/alpha_copy', help="/weka/users/yihan.hu/Matting/Composition-1k-testset/alpha_copy")
    parser.add_argument('--trimap-dir', type=str, default='/weka/users/yihan.hu/Matting/Composition-1k-testset/trimaps', help="/weka/users/yihan.hu/Matting/Composition-1k-testset/trimaps")
    parser.add_argument('--merged-dir', type=str, default='/weka/users/yihan.hu/Matting/Composition-1k-testset/merged', help="/weka/users/yihan.hu/Matting/Composition-1k-testset/merged")

    args = parser.parse_args()

    evaluate(args)

"""
CUDA_VISIBLE_DEVICES=2 python evaluation.py \
    --pred-dir /home/yihan.hu/workdir/DiffusionMattingV2/predDIM/SIMD/ddim1/base_x0_linear0.2_mm_sa_stage2only_lr5_long/model_0215487.pth \
    --label-dir /weka/users/yihan.hu/Matting/SIMD/Test/alpha \
    --trimap-dir /weka/users/yihan.hu/Matting/SIMD/Test/trimaps

CUDA_VISIBLE_DEVICES=2 python evaluation.py \
    --pred-dir /home/yihan.hu/workdir/DiffusionMattingV2/predDIM/D646/ddim10/res34_x0_linear0.2_mm_sa_stage2only_lr_bs/model_final.pth \
    --label-dir /weka/users/yihan.hu/Matting/D646/Test/GT \
    --trimap-dir /weka/users/yihan.hu/Matting/D646/Test/trimaps

CUDA_VISIBLE_DEVICES=2 python evaluation.py \
    --pred-dir /home/yihan.hu/workdir/DiffusionMattingV2/predDIM/D646/ddim1/res34_x0_linear0.2_mm_sa_stage2only_lr_bs/model_final.pth \
    --label-dir /weka/users/yihan.hu/Matting/D646/Test/GT \
    --trimap-dir /weka/users/yihan.hu/Matting/D646/Test/trimaps

python evaluation.py \
    --pred-dir /home/yihan.hu/workdir/DiffusionMattingV2/predDIM/D646/ddim10/x0_linear0.2_mm_sa_stage2only/model_0202019.pth \
    --label-dir /weka/users/yihan.hu/Matting/D646/Test/GT \
    --trimap-dir /weka/users/yihan.hu/Matting/D646/Test/trimaps

python evaluation.py \
    --pred-dir /home/yihan.hu/workdir/DiffusionMattingV2/predDIM/D646/ddim10/base_x0_linear0.2_mm_sa_stage2only_lr5_long/model_0215487.pth \
    --label-dir /weka/users/yihan.hu/Matting/D646/Test/GT \
    --trimap-dir /weka/users/yihan.hu/Matting/D646/Test/trimaps
    

CUDA_VISIBLE_DEVICES=0 python evaluation.py \
    --pred-dir /home/yihan.hu/workdir/DiffusionMattingV2/predD646_S/gca \
    --label-dir /weka/users/yihan.hu/Matting/D646/Test/GT \
    --trimap-dir /weka/users/yihan.hu/Matting/D646/Test/trimaps

CUDA_VISIBLE_DEVICES=7 python evaluation.py \
    --pred-dir /home/yihan.hu/workdir/DiffusionMattingV2/predD646_S/matteformer/pred_alpha \
    --label-dir /weka/users/yihan.hu/Matting/D646/Test/GT \
    --trimap-dir /weka/users/yihan.hu/Matting/D646/Test/trimaps

    
CUDA_VISIBLE_DEVICES=7 python evaluation.py \
    --pred-dir /home/yihan.hu/workdir/DiffusionMattingV2/predAIM_S/vits_1024 \
    --label-dir /weka/users/yihan.hu/Matting/AIM-500/alpha_copy \
    --trimap-dir /weka/users/yihan.hu/Matting/AIM-500/trimaps

CUDA_VISIBLE_DEVICES=2 python evaluation.py \
    --pred-dir /home/yihan.hu/workdir/DiffusionMattingV2/test_zone/test_result/step_000 \
    --label-dir /weka/users/yihan.hu/Matting/Composition-1k-testset/alpha_copy \
    --trimap-dir /weka/users/yihan.hu/Matting/Composition-1k-testset/trimaps

CUDA_VISIBLE_DEVICES=2 python evaluation.py \
    --pred-dir /home/yihan.hu/workdir/DiffusionMattingV2/test_zone/test_result/step_004 \
    --label-dir /weka/users/yihan.hu/Matting/Composition-1k-testset/alpha_copy \
    --trimap-dir /weka/users/yihan.hu/Matting/Composition-1k-testset/trimaps

CUDA_VISIBLE_DEVICES=2 python evaluation.py \
    --pred-dir /home/yihan.hu/workdir/DiffusionMattingV2/test_zone/test_result/step_005 \
    --label-dir /weka/users/yihan.hu/Matting/Composition-1k-testset/alpha_copy \
    --trimap-dir /weka/users/yihan.hu/Matting/Composition-1k-testset/trimaps

"""