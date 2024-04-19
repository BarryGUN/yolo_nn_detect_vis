import argparse

import torch
import yaml

from utils.downloads import attempt_download


def set_scale(opt):
    weight_path, scale = opt.weight_path, opt.scale
    weights = attempt_download(weight_path)
    ckpt = torch.load(weights, map_location='cpu')
    ckpt['model'].yaml['scale'] = scale
    torch.save(ckpt, weight_path)
    torch.cuda.empty_cache()
    print('ok')


def get_state_dict(opt):
    weight_path = opt.weight_path
    weights = attempt_download(weight_path)
    ckpt = torch.load(weights, map_location='cpu')
    return ckpt['model']

def get_model_attr(opt):
    weight_path, attr = opt.weight_path, opt.attr
    weights = attempt_download(weight_path)
    ckpt = torch.load(weights, map_location='cpu')
    return ckpt['opt'][attr]

def set_model_opt(opt):
    weight_path, opt_path = opt.weight_path, opt.opts_path
    weights = attempt_download(weight_path)
    ckpt = torch.load(weights, map_location='cpu')
    with open(opt_path, 'r', encoding='utf-8') as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    for k, v in data.items():
        ckpt['opt'][k] = v
    torch.save(ckpt, weight_path)
    torch.cuda.empty_cache()
    print('ok')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight-path', type=str,
                        help="weight path")
    parser.add_argument('--scale', type=str, default='n',
                        help="model scale")
    parser.add_argument('--attr', type=str, default='name',
                        help="model scale")
    parser.add_argument('--opts-path', type=str,
                        help="opt path")
    args = parser.parse_args()


    # set_scale(args)
    # set_model_opt(args)
    print(get_model_attr(args))
