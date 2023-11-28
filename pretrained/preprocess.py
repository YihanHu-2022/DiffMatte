import torch
import wget

def preprocess(model, name='dino', shape = None):
    new_model = {}
    for k in model.keys():
        if 'patch_embed.proj.weight' in k or 'module.conv1.module.weight_bar' in k: # vit
            x = torch.zeros(shape[0], shape[1], shape[2], shape[3])
            x[:, :3] = model[k]
            new_model['model.backbone.'+k] = x
        elif 'module.conv1.module.weight_v' in k:
            x = torch.zeros(4, 3, 3)
            x[:3, :, :] = model[k].view(3, 3, 3)
            new_model['model.backbone.'+k] = x.view(-1)
        else:
            new_model['model.backbone.'+k] = model[k]

    torch.save(new_model, name+'_fna.pth')


def remove_prefix_state_dict(state_dict, prefix="module"):
    """
    remove prefix from the key of pretrained state dict for Data-Parallel
    """
    new_state_dict = {}
    first_state_name = list(state_dict.keys())[0]
    if not first_state_name.startswith(prefix):
        for key, value in state_dict.items():
            new_state_dict[key] = state_dict[key].float()
    else:
        for key, value in state_dict.items():
            new_state_dict[key[len(prefix)+1:]] = state_dict[key].float()
    return new_state_dict

def res34_preprocess(checkpoint):
    trimap_channel = 3
    state_dict = remove_prefix_state_dict(checkpoint['state_dict'])
    for key, value in state_dict.items():
        state_dict[key] = state_dict[key].float()


    weight_u = state_dict["conv1.module.weight_u"]
    weight_v = state_dict["conv1.module.weight_v"]
    weight_bar = state_dict["conv1.module.weight_bar"]

    new_weight_v = torch.zeros((3+trimap_channel), 3, 3).cuda()
    new_weight_bar = torch.zeros(32, (3+trimap_channel), 3, 3).cuda()

    new_weight_v[:3, :, :].copy_(weight_v.view(3, 3, 3))
    new_weight_bar[:, :3, :, :].copy_(weight_bar)


    state_dict["conv1.module.weight_v"] = new_weight_v.view(-1)
    state_dict["conv1.module.weight_bar"] = new_weight_bar

    new_model = {}
    for k in state_dict.keys():
        new_model['model.backbone.'+k] = state_dict[k]

    
    torch.save(new_model, "res34"+'_fna.pth')

def swin_preprocess(weight):
    weight_ = {}
    for i, (k, v) in enumerate(weight.items()):
        head = k.split('.')[0]
        if head in ['patch_embed', 'layers']:
            if 'attn_mask' in k:
                print('[{}/{}] {} will be ignored'.format(i, len(weight.items()), k))
                continue
            weight_.update({'model.backbone.'+k: v})
        else:
            print('[{}/{}] {} will be ignored'.format(i, len(weight.items()), k))

    patch_embed_weight = weight_['model.backbone.patch_embed.proj.weight']
    patch_embed_weight_new = torch.nn.init.xavier_normal_(torch.randn(96, (3 + 3), 4, 4).cuda())
    patch_embed_weight_new[:, :3, :, :].copy_(patch_embed_weight)
    weight_['model.backbone.'+'patch_embed.proj.weight'] = patch_embed_weight_new

    attn_layers = [k for k, v in weight_.items() if 'attn.relative_position_bias_table' in k]
    for layer_name in attn_layers:
        pos_bias = weight_[layer_name]
        n_bias, n_head = pos_bias.shape

        layer_idx, block_idx = int(layer_name.split('.')[3]), int(layer_name.split('.')[5])
        n_prior = block_idx + 1
        pos_bias_new = torch.nn.init.xavier_normal_(torch.randn(n_bias + n_prior*3, n_head))

        pos_bias_new[:n_bias, :] = pos_bias
        weight_[layer_name] = pos_bias_new

    attn_layers = [k for k, v in weight_.items() if 'attn.relative_position_index' in k]
    for layer_name in attn_layers:
        pos_index = weight_[layer_name]

        layer_idx, block_idx = int(layer_name.split('.')[3]), int(layer_name.split('.')[5])
        n_prior = block_idx + 1

        num_patch = 49
        last_idx = 169
        pos_index_new = torch.ones((num_patch, num_patch + n_prior*3)).long() * last_idx
        pos_index_new[:num_patch, :num_patch] = pos_index
        for i in range(n_prior):
            for j in range(3):
                pos_index_new[:, num_patch + i*3 + j:num_patch + i*3 +j +1] = last_idx + i*3 + j
        weight_[layer_name] = pos_index_new

    torch.save(weight_, "swin_t"+'_fna.pth')
    print('load pretrained model done')

if __name__ == "__main__":

    # wget.download('https://dl.fbaipublicfiles.com/dino/dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth')
    # wget.download('https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth')

    # dino_model = torch.load('/home/yihan.hu/workdir/DiffusionMattingV2/pretrained/dino_deitsmall16_pretrain.pth')
    # mae_model = torch.load('/home/yihan.hu/workdir/DiffusionMattingV2/pretrained/mae_pretrain_vit_base.pth')['model']
    # res_model = torch.load('/home/yihan.hu/workdir/DiffusionMattingV2/pretrained/model_best_resnet34_En_nomixup.pth')
    # swin_model = torch.load('/home/yihan.hu/workdir/DiffusionMattingV2/pretrained/swin_tiny_patch4_window7_224.pth')['model']
    # res34_preprocess(res_model)
    # swin_preprocess(swin_model)
    res_model = torch.load("/home/yihan.hu/workdir/DiffusionMattingV2/pretrained/res34_fna.pth")
    swin_model = torch.load("/home/yihan.hu/workdir/DiffusionMattingV2/pretrained/swin_t_fna.pth")
    print("Done")

