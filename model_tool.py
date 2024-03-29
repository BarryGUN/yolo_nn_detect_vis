import torch

from utils.downloads import attempt_download


def set_scale(weight_path, scale):
    weights = attempt_download(weight_path)
    ckpt = torch.load(weights, map_location='cpu')
    ckpt['model'].yaml['scale'] = scale
    torch.save(ckpt, weight_path)
    print('ok')

# weights = 'run/train/1/weights/bestks.pt'
# # weights = 'run/train/1/weights/best.pt'
# weights = attempt_download(weights)  # download if not found locally
# # new = 'run/train/1/weights/bestks.pt'
# ckpt = torch.load(weights, map_location='cpu')  # load checkpoint to CPU to avoid CUDA memory leak
# print(ckpt['model'].yaml)
# ckpt['model'].yaml['ks'] = 1
# torch.save(ckpt, new)