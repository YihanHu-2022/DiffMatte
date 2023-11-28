import torch
import torch.nn as nn
import torch.nn.functional as F

class DiffusionMattingCriterion(nn.Module):
    def __init__(self,
                 *,
                 use_mse,
                 use_mat,
                 use_sa='',
                 sa_weight=1
                 ):
        super(DiffusionMattingCriterion, self).__init__()
        losses = {"mse":use_mse, "matting":use_mat, "SA":use_sa}
        self.losses = losses
        self.sa_weight = sa_weight

    def loss_gradient_penalty(self, sample_map ,preds, targets):

        #sample_map for unknown area
        scale = sample_map.shape[0]*262144/torch.sum(sample_map)

        #gradient in x
        sobel_x_kernel = torch.tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]]).type(dtype=preds.type())
        delta_pred_x = F.conv2d(preds, weight=sobel_x_kernel, padding=1)
        delta_gt_x = F.conv2d(targets, weight=sobel_x_kernel, padding=1)

        #gradient in y 
        sobel_y_kernel = torch.tensor([[[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]]).type(dtype=preds.type())
        delta_pred_y = F.conv2d(preds, weight=sobel_y_kernel, padding=1)
        delta_gt_y = F.conv2d(targets, weight=sobel_y_kernel, padding=1)

        #loss
        loss = (F.l1_loss(delta_pred_x*sample_map, delta_gt_x*sample_map)* scale + \
            F.l1_loss(delta_pred_y*sample_map, delta_gt_y*sample_map)* scale + \
            0.01 * torch.mean(torch.abs(delta_pred_x*sample_map))* scale +  \
            0.01 * torch.mean(torch.abs(delta_pred_y*sample_map))* scale)

        return dict(loss_gradient_penalty=loss)

    def loss_pha_laplacian(self, preds, targets):
        loss = laplacian_loss(preds, targets)

        return dict(loss_pha_laplacian=loss)

    def unknown_l1_loss(self, sample_map, preds, targets):
        
        scale = sample_map.shape[0]*262144/torch.sum(sample_map)
        # scale = 1

        loss = F.l1_loss(preds*sample_map, targets*sample_map)*scale
        return dict(unknown_l1_loss=loss)

    def known_l1_loss(self, sample_map, preds, targets):
        new_sample_map = torch.zeros_like(sample_map)
        new_sample_map[sample_map==0] = 1
        
        if torch.sum(new_sample_map) == 0:
            scale = 0
        else:
            scale = new_sample_map.shape[0]*262144/torch.sum(new_sample_map)
        # scale = 1

        loss = F.l1_loss(preds*new_sample_map, targets*new_sample_map)*scale
        return dict(known_l1_loss=loss)

    def composition_loss(self, sample_map, alpha, fg, bg, image):
        scale = sample_map.shape[0]*262144/torch.sum(sample_map)
        merged = fg * alpha + bg * (1 - alpha)
        loss = F.l1_loss(merged*sample_map, image*sample_map, reduction='sum')*scale
        return dict(composition_loss=loss)
    
    def mse_loss(self, model_output, target):
        loss = torch.mean(mean_flat((target - model_output) ** 2)) 
        return dict(mse_loss=loss)
    
    def sa_loss_v1(self, sample_map, preds, targets, previous, weight=1):
        scale = sample_map.shape[0]*262144/torch.sum(sample_map)

        epsilon1 = torch.abs(preds*sample_map - targets*sample_map)
        epsilon2 = torch.abs(previous*sample_map - targets*sample_map)
        
        drift = F.smooth_l1_loss(epsilon1, epsilon2, reduction='mean')
        loss = -torch.log(drift) * scale
        tau = torch.mean(mean_flat((previous - targets) ** 2))

        return dict(sa_loss = weight * tau * loss)

    def sa_loss_v2(self, sample_map, preds, targets, previous, weight=1):
        scale = sample_map.shape[0]*262144/torch.sum(sample_map)

        # epsilon1 = torch.abs(preds*sample_map - targets*sample_map)
        # epsilon2 = torch.abs(previous*sample_map - targets*sample_map)
        
        drift = F.smooth_l1_loss(preds*sample_map, previous*sample_map, reduction='mean')
        loss = -torch.log(drift) * scale
        tau = torch.mean(mean_flat((previous - targets) ** 2))

        return dict(sa_loss = weight * tau * loss)

    def sa_loss_v3(self, sample_map, preds, targets, previous, weight=1):
        scale = sample_map.shape[0]*262144/torch.sum(sample_map)

        epsilon1 = torch.abs(cal_gradient(preds)*sample_map - cal_gradient(targets)*sample_map)
        epsilon2 = torch.abs(cal_gradient(previous)*sample_map - cal_gradient(targets)*sample_map)
        
        drift = F.smooth_l1_loss(epsilon1, epsilon2, reduction='mean')
        loss = -torch.log(drift) * scale
        tau = torch.mean(mean_flat((previous - targets) ** 2))

        return dict(sa_loss = weight * tau * loss)


    def forward(self, sample_map, model_output, pred_alpha, targets, pha, fg, bg, image, previous):
        losses = dict()

        self.sa_loss = {
            'v1': self.sa_loss_v1, 'v2': self.sa_loss_v2, 'v3': self.sa_loss_v3
        }

        if self.losses["mse"] == True:
            losses.update(self.mse_loss(model_output, targets))
        if self.losses["matting"] == True:
            losses.update(self.unknown_l1_loss(sample_map, pred_alpha, pha)) 
            losses.update(self.known_l1_loss(sample_map, pred_alpha, pha))
            losses.update(self.loss_gradient_penalty(sample_map, pred_alpha, pha))
            losses.update(self.loss_pha_laplacian(pred_alpha, pha))
            # losses.update(self.composition_loss(sample_map, pred_alpha, fg, bg, image))
        if self.losses["SA"] != '' and previous is not None:
            previous = previous.detach()
            losses.update(self.sa_loss[self.losses["SA"]](sample_map, pred_alpha, pha, previous, self.sa_weight))


        # for k in self.losses:
        #     if k=='unknown_l1_loss' or k=='known_l1_loss' or k=='loss_gradient_penalty' or k == 'composition_loss':
        #         losses.update(getattr(self, k)(sample_map, model_output, targets))
        #     else:
        #         losses.update(getattr(self, k)(model_output, targets))
        return losses


class MattingCriterion(nn.Module):
    def __init__(self,
                 *,
                 losses,
                 ):
        super(MattingCriterion, self).__init__()
        self.losses = losses

    def loss_gradient_penalty(self, sample_map ,preds, targets):
        preds = preds['phas']
        targets = targets['phas']

        #sample_map for unknown area
        scale = sample_map.shape[0]*262144/torch.sum(sample_map)

        #gradient in x
        sobel_x_kernel = torch.tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]]).type(dtype=preds.type())
        delta_pred_x = F.conv2d(preds, weight=sobel_x_kernel, padding=1)
        delta_gt_x = F.conv2d(targets, weight=sobel_x_kernel, padding=1)

        #gradient in y 
        sobel_y_kernel = torch.tensor([[[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]]).type(dtype=preds.type())
        delta_pred_y = F.conv2d(preds, weight=sobel_y_kernel, padding=1)
        delta_gt_y = F.conv2d(targets, weight=sobel_y_kernel, padding=1)

        #loss
        loss = (F.l1_loss(delta_pred_x*sample_map, delta_gt_x*sample_map)* scale + \
            F.l1_loss(delta_pred_y*sample_map, delta_gt_y*sample_map)* scale + \
            0.01 * torch.mean(torch.abs(delta_pred_x*sample_map))* scale +  \
            0.01 * torch.mean(torch.abs(delta_pred_y*sample_map))* scale)

        return dict(loss_gradient_penalty=loss)

    def loss_pha_laplacian(self, preds, targets):
        assert 'phas' in preds and 'phas' in targets
        loss = laplacian_loss(preds['phas'], targets['phas'])

        return dict(loss_pha_laplacian=loss)

    def unknown_l1_loss(self, sample_map, preds, targets):
        
        scale = sample_map.shape[0]*262144/torch.sum(sample_map)
        # scale = 1

        loss = F.l1_loss(preds['phas']*sample_map, targets['phas']*sample_map)*scale
        return dict(unknown_l1_loss=loss)

    def known_l1_loss(self, sample_map, preds, targets):
        new_sample_map = torch.zeros_like(sample_map)
        new_sample_map[sample_map==0] = 1
        
        if torch.sum(new_sample_map) == 0:
            scale = 0
        else:
            scale = new_sample_map.shape[0]*262144/torch.sum(new_sample_map)
        # scale = 1

        loss = F.l1_loss(preds['phas']*new_sample_map, targets['phas']*new_sample_map)*scale
        return dict(known_l1_loss=loss)


    def forward(self, sample_map, preds, targets):
        losses = dict()
        for k in self.losses:
            if k=='unknown_l1_loss' or k=='known_l1_loss' or k=='loss_gradient_penalty':
                losses.update(getattr(self, k)(sample_map, preds, targets))
            else:
                losses.update(getattr(self, k)(preds, targets))
        return losses

def cal_gradient(img):
    sobel_x_kernel = torch.tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]]).type(img.type())
    sobel_y_kernel = torch.tensor([[[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]]).type(img.type())

    delta_x = F.conv2d(img, weight=sobel_x_kernel, padding=1)
    delta_y = F.conv2d(img, weight=sobel_y_kernel, padding=1)

    delta = torch.abs(torch.abs(delta_x) + torch.abs(delta_y))
    return delta

def get_unknown_tensor(trimap):
    """
    get 1-channel unknown area tensor from the 3-channel/1-channel trimap tensor
    """
    if trimap.shape[1] == 3:
        weight = trimap[:, 1:2, :, :].float()
    else:
        weight = trimap.eq(0.5).float()
    return weight

def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


#-----------------Laplacian Loss-------------------------#
def laplacian_loss(pred, true, max_levels=5):
    kernel = gauss_kernel(device=pred.device, dtype=pred.dtype)
    pred_pyramid = laplacian_pyramid(pred, kernel, max_levels)
    true_pyramid = laplacian_pyramid(true, kernel, max_levels)
    loss = 0
    for level in range(max_levels):
        loss += (2 ** level) * F.l1_loss(pred_pyramid[level], true_pyramid[level])
    return loss / max_levels



def laplacian_pyramid(img, kernel, max_levels):
    current = img
    pyramid = []
    for _ in range(max_levels):
        current = crop_to_even_size(current)
        down = downsample(current, kernel)
        up = upsample(down, kernel)
        diff = current - up
        pyramid.append(diff)
        current = down
    return pyramid

def gauss_kernel(device='cpu', dtype=torch.float32):
    kernel = torch.tensor([[1,  4,  6,  4, 1],
                        [4, 16, 24, 16, 4],
                        [6, 24, 36, 24, 6],
                        [4, 16, 24, 16, 4],
                        [1,  4,  6,  4, 1]], device=device, dtype=dtype)
    kernel /= 256
    kernel = kernel[None, None, :, :]
    return kernel

def gauss_convolution(img, kernel):
    B, C, H, W = img.shape
    img = img.reshape(B * C, 1, H, W)
    img = F.pad(img, (2, 2, 2, 2), mode='reflect')
    img = F.conv2d(img, kernel)
    img = img.reshape(B, C, H, W)
    return img

def downsample(img, kernel):
    img = gauss_convolution(img, kernel)
    img = img[:, :, ::2, ::2]
    return img

def upsample(img, kernel):
    B, C, H, W = img.shape
    out = torch.zeros((B, C, H * 2, W * 2), device=img.device, dtype=img.dtype)
    out[:, :, ::2, ::2] = img * 4
    out = gauss_convolution(out, kernel)
    return out

def crop_to_even_size(img):
    H, W = img.shape[2:]
    H = H - H % 2
    W = W - W % 2
    return img[:, :, :H, :W]