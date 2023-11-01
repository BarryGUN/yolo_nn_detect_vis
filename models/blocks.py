# ConvNextV2Block

import numpy as np
import torch
from torch import nn as nn
from torch.nn import functional

from models.conv import SRepConv, Conv, GhostConv, DWConv, autopad


class RepVGGBlock(nn.Module):
    '''RepVGGBlock is a basic rep-style block, including training and deploy status
    This code is based on https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py
    '''

    default_act = nn.SiLU()

    # default siluo
    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=1, padding=1, dilation=1, groups=1, padding_mode='zeros', act=True, deploy=False,
                 use_se=False):
        super(RepVGGBlock, self).__init__()
        """ Initialization of the class.
        Args:
            in_channels (int): Number of channels in the input image
            out_channels (int): Number of channels produced by the convolution
            kernel_size (int or tuple): Size of the convolving kernel
            stride (int or tuple, optional): Stride of the convolution. Default: 1
            padding (int or tuple, optional): Zero-padding added to both sides of
                the input. Default: 1
            dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
            groups (int, optional): Number of blocked connections from input
                channels to output channels. Default: 1
            padding_mode (string, optional): Default: 'zeros'
            deploy: Whether to be deploy status or training status. Default: False
            use_se: Whether to use se. Default: False
        """
        self.deploy = deploy
        # self.groups = groups
        # self.in_channels = in_channels
        # self.out_channels = out_channels
        # self.kernel_size = kernel_size
        # self.stride = stride
        # self.padding = padding
        # self.dilation = dilation

        assert kernel_size == 3
        assert padding == 1

        padding_11 = padding - kernel_size // 2

        self.act = None if act is True else act if isinstance(act, nn.Module) else nn.Identity()

        if use_se:
            raise NotImplementedError("se block not supported yet")
        else:
            self.se = nn.Identity()

        if deploy:
            self.rbr_reparam = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                         stride=stride,
                                         padding=padding, dilation=dilation, groups=groups, bias=True,
                                         padding_mode=padding_mode)
        else:
            self.rbr_identity = nn.BatchNorm2d(
                num_features=in_channels) if out_channels == in_channels and stride == 1 else None
            self.rbr_dense = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, autopad(kernel_size, padding), groups=groups,
                          bias=False),
                nn.BatchNorm2d(num_features=out_channels),
            )

            self.rbr_1x1 = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, padding_11, groups=groups, bias=False),
                nn.BatchNorm2d(num_features=out_channels),
            )

    def forward(self, inputs):
        '''Forward process'''
        if hasattr(self, 'rbr_reparam'):
            return RepVGGBlock.default_act(self.se(self.rbr_reparam(inputs))) if self.act is None else self.act(
                self.se(self.rbr_reparam(inputs)))

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)

        return RepVGGBlock.default_act(
            self.se(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out)) if self.act is None else self.act(
            self.se(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out))

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, nn.Sequential):
            kernel = branch[0].weight
            running_mean = branch[1].running_mean
            running_var = branch[1].running_var
            gamma = branch[1].weight
            beta = branch[1].bias
            eps = branch[1].eps
            # kernel = branch.conv.weight
            # running_mean = branch.bn.running_mean
            # running_var = branch.bn.running_var
            # gamma = branch.bn.weight
            # beta = branch.bn.bias
            # eps = branch.bn.eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, 'id_tensor'):
                input_dim = self.in_channels // self.groups
                kernel_value = np.zeros((self.in_channels, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                    # print(branch.weight.device)
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def switch_to_half(self):
        if not hasattr(self, 'rbr_reparam'):
            self.switch_to_deploy()
        self.kernel_float32 = self.rbr_reparam.weight
        self.rbr_reparam.weight.data = self.rbr_reparam.weight.half()

    def switch_to_deploy(self):
        if hasattr(self, 'rbr_reparam'):
            if hasattr(self, 'kernel_float32'):
                self.rbr_reparam.weight.data = self.kernel_float32
                self.__delattr__('kernel_float32')
            return
        kernel, bias = self.get_equivalent_kernel_bias()

        self.rbr_reparam = nn.Conv2d(in_channels=self.rbr_dense[0].in_channels,
                                     out_channels=self.rbr_dense[0].out_channels,
                                     kernel_size=self.rbr_dense[0].kernel_size, stride=self.rbr_dense[0].stride,
                                     padding=self.rbr_dense[0].padding, dilation=self.rbr_dense[0].dilation,
                                     groups=self.rbr_dense[0].groups, bias=True)
        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__('rbr_dense')
        self.__delattr__('rbr_1x1')
        if hasattr(self, 'rbr_identity'):
            self.__delattr__('rbr_identity')
        if hasattr(self, 'id_tensor'):
            self.__delattr__('id_tensor')
        self.deploy = True
    # Represented convolution
    # https://arxiv.org/abs/2101.03697


class FastRepVGGBlock(nn.Module):
    '''RepVGGBlock is a basic rep-style block, including training and deploy status
    This code is based on https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py
    '''

    default_act = nn.SiLU()

    # default silu
    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=1, padding=1, dilation=1, groups=1, padding_mode='zeros', act=True, deploy=False,
                 use_se=False):
        super(FastRepVGGBlock, self).__init__()
        """ Initialization of the class.
        Args:
            in_channels (int): Number of channels in the input image
            out_channels (int): Number of channels produced by the convolution
            kernel_size (int or tuple): Size of the convolving kernel
            stride (int or tuple, optional): Stride of the convolution. Default: 1
            padding (int or tuple, optional): Zero-padding added to both sides of
                the input. Default: 1
            dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
            groups (int, optional): Number of blocked connections from input
                channels to output channels. Default: 1
            padding_mode (string, optional): Default: 'zeros'
            deploy: Whether to be deploy status or training status. Default: False
            use_se: Whether to use se. Default: False
        """
        self.deploy = deploy
        # self.in_channels = in_channels
        # self.out_channels = out_channels
        # self.kernel_size = kernel_size
        # self.stride = stride
        # self.padding = padding
        # self.dilation = dilation
        # self.groups = groups

        assert kernel_size == 3
        assert padding == 1
        assert in_channels == out_channels

        padding_11 = padding - kernel_size // 2

        # self.act = FastRepVGGBlock.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
        self.act = None if act is True else act if isinstance(act, nn.Module) else nn.Identity()

        if use_se:
            raise NotImplementedError("se block not supported yet")
        else:
            self.se = nn.Identity()

        if deploy:
            self.rbr_reparam = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                         stride=stride,
                                         padding=padding, dilation=dilation, groups=groups, bias=True,
                                         padding_mode=padding_mode)
        else:

            self.rbr_identity = nn.BatchNorm2d(num_features=out_channels)
            self.rbr_dense = nn.Conv2d(in_channels, out_channels, kernel_size, stride, autopad(kernel_size, padding),
                                       groups=groups,
                                       bias=False)

            self.rbr_1x1 = nn.Conv2d(in_channels, out_channels, 1, stride, padding_11, groups=groups, bias=False)

            self.bn = nn.BatchNorm2d(num_features=out_channels)

    def forward(self, inputs):
        '''Forward process'''
        if hasattr(self, 'rbr_reparam'):
            return FastRepVGGBlock.default_act(self.se(self.rbr_reparam(inputs))) if self.act is None else self.act(
                self.se(self.rbr_reparam(inputs)))

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)

        return FastRepVGGBlock.default_act(
            self.se(self.bn(self.rbr_dense(inputs) + self.rbr_1x1(inputs)) + id_out)) if self.act is None else self.act(
            self.se(self.bn(self.rbr_dense(inputs) + self.rbr_1x1(inputs)) + id_out))

    def get_equivalent_kernel_bias(self):
        kernel3x3_fused = self._fuse_1x1_and_3x3_kernel(self._pad_1x1_to_3x3_tensor(self.rbr_1x1.weight),
                                                        self.rbr_dense.weight)
        kernel3x3_main, bias3x3_main = self._fuse_bn_tensor(kernel3x3_fused,
                                                            *self._get_bn_params(self.bn))

        kernelid, biasid = self._fuse_bn_tensor(self._get_id_tensor(self.rbr_identity),
                                                *self._get_bn_params(self.rbr_identity))
        return kernel3x3_main + kernelid, bias3x3_main + biasid

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return functional.pad(kernel1x1, [1, 1, 1, 1])

    def _get_id_tensor(self, branch):
        assert isinstance(branch, nn.BatchNorm2d)
        if not hasattr(self, 'id_tensor'):
            input_dim = self.rbr_dense.in_channels // self.rbr_dense.groups
            kernel_value = np.zeros((self.rbr_dense.in_channels, input_dim, 3, 3), dtype=np.float32)
            for i in range(self.rbr_dense.in_channels):
                kernel_value[i, i % input_dim, 1, 1] = 1
                # print(branch.weight.device)
            self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            return self.id_tensor

    def _get_bn_params(self, branch):
        assert isinstance(branch, nn.BatchNorm2d)
        running_mean = branch.running_mean
        running_var = branch.running_var
        gamma = branch.weight
        beta = branch.bias
        eps = branch.eps
        return running_mean, running_var, gamma, beta, eps

    def _fuse_1x1_and_3x3_kernel(self, k1x1, k3x3):
        return k1x1 + k3x3

    def _fuse_bn_tensor(self, kernel, running_mean, running_var, gamma, beta, eps):
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def switch_to_half(self):
        if not hasattr(self, 'rbr_reparam'):
            self.switch_to_deploy()
        self.kernel_float32 = self.rbr_reparam.weight
        self.rbr_reparam.weight.data = self.rbr_reparam.weight.half()

    def switch_to_deploy(self):
        if hasattr(self, 'rbr_reparam'):
            if hasattr(self, 'kernel_float32'):
                self.rbr_reparam.weight.data = self.kernel_float32
                self.__delattr__('kernel_float32')
            return
        kernel, bias = self.get_equivalent_kernel_bias()

        self.rbr_reparam = nn.Conv2d(in_channels=self.rbr_dense.in_channels,
                                     out_channels=self.rbr_dense.out_channels,
                                     kernel_size=self.rbr_dense.kernel_size, stride=self.rbr_dense.stride,
                                     padding=self.rbr_dense.padding, dilation=self.rbr_dense.dilation,
                                     groups=self.rbr_dense.groups, bias=True)
        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__('rbr_dense')
        self.__delattr__('rbr_1x1')
        if hasattr(self, 'rbr_identity'):
            self.__delattr__('rbr_identity')
        if hasattr(self, 'id_tensor'):
            self.__delattr__('id_tensor')
        if hasattr(self, 'bn'):
            self.__delattr__('bn')
        self.deploy = True
    # Represented convolution
    # https://arxiv.org/abs/2101.03697


class FastRepVGGBlockV2(nn.Module):
    '''RepVGGBlock is a basic rep-style block, including training and deploy status
    This code is based on https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py
    '''

    default_act = nn.SiLU()

    # default silu
    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=1, padding=1, dilation=1, groups=1, padding_mode='zeros', act=True, deploy=False,
                 use_se=False):
        super(FastRepVGGBlockV2, self).__init__()
        """ Initialization of the class.
        Args:
            in_channels (int): Number of channels in the input image
            out_channels (int): Number of channels produced by the convolution
            kernel_size (int or tuple): Size of the convolving kernel
            stride (int or tuple, optional): Stride of the convolution. Default: 1
            padding (int or tuple, optional): Zero-padding added to both sides of
                the input. Default: 1
            dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
            groups (int, optional): Number of blocked connections from input
                channels to output channels. Default: 1
            padding_mode (string, optional): Default: 'zeros'
            deploy: Whether to be deploy status or training status. Default: False
            use_se: Whether to use se. Default: False
        """
        self.deploy = deploy
        # self.in_channels = in_channels
        # self.out_channels = out_channels
        # self.kernel_size = kernel_size
        # self.stride = stride
        # self.padding = padding
        # self.dilation = dilation
        # self.groups = groups

        assert kernel_size == 3
        assert padding == 1
        assert in_channels == out_channels

        padding_11 = padding - kernel_size // 2

        # self.act = FastRepVGGBlock.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
        self.act = None if act is True else act if isinstance(act, nn.Module) else nn.Identity()

        if use_se:
            raise NotImplementedError("se block not supported yet")
        else:
            self.se = nn.Identity()

        if deploy:
            self.rbr_reparam = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                         stride=stride,
                                         padding=padding, dilation=dilation, groups=groups, bias=True,
                                         padding_mode=padding_mode)
        else:

            # self.rbr_identity = nn.BatchNorm2d(num_features=out_channels)
            self.rbr_identity = nn.Identity()
            self.rbr_dense = nn.Conv2d(in_channels, out_channels, kernel_size, stride, autopad(kernel_size, padding),
                                       groups=groups,
                                       bias=False)

            self.rbr_1x1 = nn.Conv2d(in_channels, out_channels, 1, stride, padding_11, groups=groups, bias=False)

            self.bn = nn.BatchNorm2d(num_features=out_channels)

    def forward(self, inputs):
        '''Forward process'''
        if hasattr(self, 'rbr_reparam'):
            return FastRepVGGBlock.default_act(self.se(self.rbr_reparam(inputs))) if self.act is None else self.act(
                self.se(self.rbr_reparam(inputs)))

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)

        return FastRepVGGBlock.default_act(
            self.se(self.bn(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out))) if self.act is None else self.act(
            self.se(self.bn(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out)))

    def get_equivalent_kernel_bias(self):
        kernel3x3_fused = self._fuse_1x1_and_3x3_kernel(self._pad_1x1_to_3x3_tensor(self.rbr_1x1.weight),
                                                        self.rbr_dense.weight) + self._get_id_tensor()
        kernel3x3_main, bias3x3_main = self._fuse_bn_tensor(kernel3x3_fused, *self._get_bn_params(self.bn))
        return kernel3x3_main, bias3x3_main

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return functional.pad(kernel1x1, [1, 1, 1, 1])

    def _get_id_tensor(self):
        # assert isinstance(branch, nn.BatchNorm2d)
        if not hasattr(self, 'id_tensor'):
            input_dim = self.rbr_dense.in_channels // self.rbr_dense.groups
            kernel_value = np.zeros((self.rbr_dense.in_channels, input_dim, 3, 3), dtype=np.float32)
            for i in range(self.rbr_dense.in_channels):
                kernel_value[i, i % input_dim, 1, 1] = 1
                # print(branch.weight.device)
            self.id_tensor = torch.from_numpy(kernel_value).to(self.rbr_dense.weight.device)
            return self.id_tensor

    def _get_bn_params(self, branch):
        assert isinstance(branch, nn.BatchNorm2d)
        running_mean = branch.running_mean
        running_var = branch.running_var
        gamma = branch.weight
        beta = branch.bias
        eps = branch.eps
        return running_mean, running_var, gamma, beta, eps

    def _fuse_1x1_and_3x3_kernel(self, k1x1, k3x3):
        return k1x1 + k3x3

    def _fuse_bn_tensor(self, kernel, running_mean, running_var, gamma, beta, eps):
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def switch_to_half(self):
        if not hasattr(self, 'rbr_reparam'):
            self.switch_to_deploy()
        self.kernel_float32 = self.rbr_reparam.weight
        self.rbr_reparam.weight.data = self.rbr_reparam.weight.half()

    def switch_to_deploy(self):
        if hasattr(self, 'rbr_reparam'):
            if hasattr(self, 'kernel_float32'):
                self.rbr_reparam.weight.data = self.kernel_float32
                self.__delattr__('kernel_float32')
            return
        kernel, bias = self.get_equivalent_kernel_bias()

        self.rbr_reparam = nn.Conv2d(in_channels=self.rbr_dense.in_channels,
                                     out_channels=self.rbr_dense.out_channels,
                                     kernel_size=self.rbr_dense.kernel_size, stride=self.rbr_dense.stride,
                                     padding=self.rbr_dense.padding, dilation=self.rbr_dense.dilation,
                                     groups=self.rbr_dense.groups, bias=True)
        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__('rbr_dense')
        self.__delattr__('rbr_1x1')
        if hasattr(self, 'rbr_identity'):
            self.__delattr__('rbr_identity')
        if hasattr(self, 'id_tensor'):
            self.__delattr__('id_tensor')
        if hasattr(self, 'bn'):
            self.__delattr__('bn')
        self.deploy = True
    # Represented convolution
    # https://arxiv.org/abs/2101.03697


class FastRepVGGBlockV3(nn.Module):
    '''RepVGGBlock is a basic rep-style block, including training and deploy status
    This code is based on https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py
    '''

    default_act = nn.SiLU()

    # default silu
    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=1, padding=1, dilation=1, groups=1, padding_mode='zeros', act=True, deploy=False,
                 use_se=False):
        super(FastRepVGGBlockV3, self).__init__()
        """ Initialization of the class.
        Args:
            in_channels (int): Number of channels in the input image
            out_channels (int): Number of channels produced by the convolution
            kernel_size (int or tuple): Size of the convolving kernel
            stride (int or tuple, optional): Stride of the convolution. Default: 1
            padding (int or tuple, optional): Zero-padding added to both sides of
                the input. Default: 1
            dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
            groups (int, optional): Number of blocked connections from input
                channels to output channels. Default: 1
            padding_mode (string, optional): Default: 'zeros'
            deploy: Whether to be deploy status or training status. Default: False
            use_se: Whether to use se. Default: False
        """
        self.deploy = deploy
        # self.in_channels = in_channels
        # self.out_channels = out_channels
        # self.kernel_size = kernel_size
        # self.stride = stride
        # self.padding = padding
        # self.dilation = dilation
        # self.groups = groups

        assert kernel_size == 3
        assert padding == 1
        assert in_channels == out_channels

        padding_11 = padding - kernel_size // 2

        # self.act = FastRepVGGBlock.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
        # self.act = None if act is True else act if isinstance(act, nn.Module) else nn.Identity()
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
        if use_se:
            raise NotImplementedError("se block not supported yet")
        else:
            self.se = nn.Identity()

        if deploy:
            self.rbr_reparam = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                         stride=stride,
                                         padding=padding, dilation=dilation, groups=groups, bias=True,
                                         padding_mode=padding_mode)
        else:

            # self.rbr_identity = nn.BatchNorm2d(num_features=out_channels)
            self.rbr_identity = nn.Identity()
            self.rbr_dense = nn.Conv2d(in_channels, out_channels, kernel_size, stride, autopad(kernel_size, padding),
                                       groups=groups,
                                       bias=False)

            self.rbr_1x1 = nn.Conv2d(in_channels, out_channels, 1, stride, padding_11, groups=groups, bias=False)

            self.bn = nn.BatchNorm2d(num_features=out_channels)

    def forward(self, inputs):
        '''Forward process'''
        if hasattr(self, 'rbr_reparam'):
            # return FastRepVGGBlock.default_act(self.se(self.rbr_reparam(inputs))) if self.act is None else self.act(
            #     self.se(self.rbr_reparam(inputs)))
            return self.act(self.se(self.rbr_reparam(inputs)))

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)

        # return FastRepVGGBlock.default_act(
        #     self.se(self.bn(self.rbr_dense(inputs) + self.rbr_1x1(inputs)) + id_out)) if self.act is None else self.act(
        #     self.se(self.bn(self.rbr_dense(inputs) + self.rbr_1x1(inputs)) + id_out))
        return self.act(self.se(self.bn(self.rbr_dense(inputs) + self.rbr_1x1(inputs)) + id_out))

    def get_equivalent_kernel_bias(self):
        kernel3x3_fused = self._fuse_1x1_and_3x3_kernel(self._pad_1x1_to_3x3_tensor(self.rbr_1x1.weight),
                                                        self.rbr_dense.weight)
        kernel3x3_main, bias3x3_main = self._fuse_bn_tensor(kernel3x3_fused, *self._get_bn_params(self.bn))
        return kernel3x3_main + self._get_id_tensor(), bias3x3_main

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return functional.pad(kernel1x1, [1, 1, 1, 1])

    def _get_id_tensor(self):
        # assert isinstance(branch, nn.BatchNorm2d)
        if not hasattr(self, 'id_tensor'):
            input_dim = self.rbr_dense.in_channels // self.rbr_dense.groups
            kernel_value = np.zeros((self.rbr_dense.in_channels, input_dim, 3, 3), dtype=np.float32)
            for i in range(self.rbr_dense.in_channels):
                kernel_value[i, i % input_dim, 1, 1] = 1
                # print(branch.weight.device)
            self.id_tensor = torch.from_numpy(kernel_value).to(self.rbr_dense.weight.device)
            return self.id_tensor

    def _get_bn_params(self, branch):
        assert isinstance(branch, nn.BatchNorm2d)
        running_mean = branch.running_mean
        running_var = branch.running_var
        gamma = branch.weight
        beta = branch.bias
        eps = branch.eps
        return running_mean, running_var, gamma, beta, eps

    def _fuse_1x1_and_3x3_kernel(self, k1x1, k3x3):
        return k1x1 + k3x3

    def _fuse_bn_tensor(self, kernel, running_mean, running_var, gamma, beta, eps):
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def switch_to_half(self):
        if not hasattr(self, 'rbr_reparam'):
            self.switch_to_deploy()
        self.kernel_float32 = self.rbr_reparam.weight
        self.rbr_reparam.weight.data = self.rbr_reparam.weight.half()

    def switch_to_deploy(self):
        if hasattr(self, 'rbr_reparam'):
            if hasattr(self, 'kernel_float32'):
                self.rbr_reparam.weight.data = self.kernel_float32
                self.__delattr__('kernel_float32')
            return
        kernel, bias = self.get_equivalent_kernel_bias()

        self.rbr_reparam = nn.Conv2d(in_channels=self.rbr_dense.in_channels,
                                     out_channels=self.rbr_dense.out_channels,
                                     kernel_size=self.rbr_dense.kernel_size, stride=self.rbr_dense.stride,
                                     padding=self.rbr_dense.padding, dilation=self.rbr_dense.dilation,
                                     groups=self.rbr_dense.groups, bias=True)
        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__('rbr_dense')
        self.__delattr__('rbr_1x1')
        if hasattr(self, 'rbr_identity'):
            self.__delattr__('rbr_identity')
        if hasattr(self, 'id_tensor'):
            self.__delattr__('id_tensor')
        if hasattr(self, 'bn'):
            self.__delattr__('bn')
        self.deploy = True
    # Represented convolution
    # https://arxiv.org/abs/2101.03697

class FastRepVGGBlockV4(nn.Module):
    '''RepVGGBlock is a basic rep-style block, including training and deploy status
    This code is based on https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py
    '''

    default_act = nn.SiLU()

    # default silu
    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=1, padding=1, dilation=1, groups=1, padding_mode='zeros', act=True, deploy=False):
        super(FastRepVGGBlockV4, self).__init__()
        """ Initialization of the class.
        Args:
            in_channels (int): Number of channels in the input image
            out_channels (int): Number of channels produced by the convolution
            kernel_size (int or tuple): Size of the convolving kernel
            stride (int or tuple, optional): Stride of the convolution. Default: 1
            padding (int or tuple, optional): Zero-padding added to both sides of
                the input. Default: 1
            dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
            groups (int, optional): Number of blocked connections from input
                channels to output channels. Default: 1
            padding_mode (string, optional): Default: 'zeros'
            deploy: Whether to be deploy status or training status. Default: False
            use_se: Whether to use se. Default: False
        """
        self.deploy = deploy
        # self.in_channels = in_channels
        # self.out_channels = out_channels
        # self.kernel_size = kernel_size
        # self.stride = stride
        # self.padding = padding
        # self.dilation = dilation
        # self.groups = groups
        assert kernel_size == 3
        assert padding == 1
        assert in_channels == out_channels

        padding_11 = padding - kernel_size // 2

        # self.act = FastRepVGGBlock.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
        # self.act = None if act is True else act if isinstance(act, nn.Module) else nn.Identity()
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()


        if deploy:
            self.rbr_reparam = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                         stride=stride,
                                         padding=padding, dilation=dilation, groups=groups, bias=True,
                                         padding_mode=padding_mode)
        else:

            self.rbr_dense = nn.Conv2d(in_channels, out_channels, kernel_size, stride, autopad(kernel_size, padding),
                                       groups=groups,
                                       bias=False)

            self.rbr_1x1 = nn.Conv2d(in_channels, out_channels, 1, stride, padding_11, groups=groups, bias=False)

            self.bn_3x3 = nn.BatchNorm2d(num_features=out_channels)
            self.bn_main = nn.BatchNorm2d(num_features=out_channels)

    def forward(self, inputs):
        '''Forward process'''
        if hasattr(self, 'rbr_reparam'):
            # return FastRepVGGBlock.default_act(self.se(self.rbr_reparam(inputs))) if self.act is None else self.act(
            #     self.se(self.rbr_reparam(inputs)))
            return self.act(self.bn_main(self.rbr_reparam(inputs)))

        else:
            return self.act(self.bn_main(self.bn_3x3(self.rbr_dense(inputs)) + self.rbr_1x1(inputs)))


    def get_equivalent_kernel_bias(self):
        kernel1x1_3x3 = self._pad_1x1_to_3x3_tensor(self.rbr_1x1.weight)
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense.weight, *self._get_bn_params(self.bn_3x3))
        kernel3x3_main, bias3x3_main = self._fuse_1x1_and_3x3_kernel(kernel3x3, kernel1x1_3x3), bias3x3

        return kernel3x3_main, bias3x3_main

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return functional.pad(kernel1x1, [1, 1, 1, 1])


    def _get_bn_params(self, branch):
        assert isinstance(branch, nn.BatchNorm2d)
        running_mean = branch.running_mean
        running_var = branch.running_var
        gamma = branch.weight
        beta = branch.bias
        eps = branch.eps
        return running_mean, running_var, gamma, beta, eps

    def _fuse_1x1_and_3x3_kernel(self, k1x1, k3x3):
        return k1x1 + k3x3

    def _fuse_bn_tensor(self, kernel, running_mean, running_var, gamma, beta, eps):
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def switch_to_half(self):
        if not hasattr(self, 'rbr_reparam'):
            self.switch_to_deploy()
        self.kernel_float32 = self.rbr_reparam.weight
        self.rbr_reparam.weight.data = self.rbr_reparam.weight.half()

    def switch_to_deploy(self):
        if hasattr(self, 'rbr_reparam'):
            if hasattr(self, 'kernel_float32'):
                self.rbr_reparam.weight.data = self.kernel_float32
                self.__delattr__('kernel_float32')
            return
        kernel, bias = self.get_equivalent_kernel_bias()

        self.rbr_reparam = nn.Conv2d(in_channels=self.rbr_dense.in_channels,
                                     out_channels=self.rbr_dense.out_channels,
                                     kernel_size=self.rbr_dense.kernel_size, stride=self.rbr_dense.stride,
                                     padding=self.rbr_dense.padding, dilation=self.rbr_dense.dilation,
                                     groups=self.rbr_dense.groups, bias=True)
        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__('rbr_dense')
        self.__delattr__('rbr_1x1')
        if hasattr(self, 'bn_3x3'):
            self.__delattr__('bn_3x3')
        self.deploy = True
    # Represented convolution
    # https://arxiv.org/abs/2101.03697

class FastRepVGGBlockD(nn.Module):
    '''RepVGGBlock is a basic rep-style block, including training and deploy status
    This code is based on https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py
    '''

    default_act = nn.SiLU()

    # default silu
    def __init__(self, in_channels, out_channels, width1x1=1, kernel_size=3,
                 stride=1, padding=1, dilation=1, groups=1, padding_mode='zeros', act=True, deploy=False,
                 use_se=False):
        super(FastRepVGGBlockD, self).__init__()
        """ Initialization of the class.
        Args:
            in_channels (int): Number of channels in the input image
            out_channels (int): Number of channels produced by the convolution
            kernel_size (int or tuple): Size of the convolving kernel
            stride (int or tuple, optional): Stride of the convolution. Default: 1
            padding (int or tuple, optional): Zero-padding added to both sides of
                the input. Default: 1
            dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
            groups (int, optional): Number of blocked connections from input
                channels to output channels. Default: 1
            padding_mode (string, optional): Default: 'zeros'
            deploy: Whether to be deploy status or training status. Default: False
            use_se: Whether to use se. Default: False
        """
        self.deploy = deploy
        # self.in_channels = in_channels
        # self.out_channels = out_channels
        # self.kernel_size = kernel_size
        # self.stride = stride
        # self.padding = padding
        # self.dilation = dilation
        # self.groups = groups

        assert kernel_size == 3
        assert padding == 1
        assert in_channels == out_channels

        padding_11 = padding - kernel_size // 2

        # self.act = FastRepVGGBlock.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
        self.act = None if act is True else act if isinstance(act, nn.Module) else nn.Identity()

        if use_se:
            raise NotImplementedError("se block not supported yet")
        else:
            self.se = nn.Identity()

        if deploy:
            self.rbr_reparam = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                         stride=stride,
                                         padding=padding, dilation=dilation, groups=groups, bias=True,
                                         padding_mode=padding_mode)
        else:

            # self.rbr_identity = nn.BatchNorm2d(num_features=out_channels)
            self.rbr_identity = nn.Identity()
            self.rbr_dense = nn.ModuleList(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, autopad(kernel_size, padding),
                          groups=groups,
                          bias=False)

                for _ in range(width1x1))

            # self.rbr_1x1 =
            # self.rbr_1x1_list = nn.ModuleList(
            #     nn.Conv2d(in_channels, out_channels, 1, stride, padding_11, groups=groups, bias=False)

            # for _ in range(width1x1))

            self.bn = nn.BatchNorm2d(num_features=out_channels)

    def forward(self, inputs):
        '''Forward process'''
        if hasattr(self, 'rbr_reparam'):
            return FastRepVGGBlock.default_act(self.se(self.rbr_reparam(inputs))) if self.act is None else self.act(
                self.se(self.rbr_reparam(inputs)))
        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)

        return FastRepVGGBlock.default_act(
            self.se(self.bn(self.dense_forward(inputs)) + id_out)) if self.act is None else self.act(
            self.se(self.bn(self.dense_forward(inputs)) + id_out))

    def dense_forward(self, x):
        result = 0
        for el in self.rbr_dense:
            result += el(x)
        return result

    def get_equivalent_kernel_bias(self):
        rbr_3x3_weights = [m.weights for m in self.rbr_dense]  # get dense weights
        kernel3x3_fused = self._fuse_kernel(rbr_3x3_weights)  # fuse 3x3
        kernel3x3_main, bias3x3_main = self._fuse_bn_tensor(kernel3x3_fused, *self._get_bn_params(self.bn))  # fuse bn
        return kernel3x3_main + self._get_id_tensor(), bias3x3_main  # fuse 3x3 and id


    def _get_id_tensor(self):
        # assert isinstance(branch, nn.BatchNorm2d)
        if not hasattr(self, 'id_tensor'):
            input_dim = self.rbr_dense.in_channels // self.rbr_dense.groups
            kernel_value = np.zeros((self.rbr_dense.in_channels, input_dim, 3, 3), dtype=np.float32)
            for i in range(self.rbr_dense.in_channels):
                kernel_value[i, i % input_dim, 1, 1] = 1
                # print(branch.weight.device)
            self.id_tensor = torch.from_numpy(kernel_value).to(self.rbr_dense.weight.device)
            return self.id_tensor

    def _get_bn_params(self, branch):
        assert isinstance(branch, nn.BatchNorm2d)
        running_mean = branch.running_mean
        running_var = branch.running_var
        gamma = branch.weight
        beta = branch.bias
        eps = branch.eps
        return running_mean, running_var, gamma, beta, eps

    def _fuse_kernel(self, kernel_weights):
        result = 0
        for val in kernel_weights:
            result += val
        return result

    def _fuse_bn_tensor(self, kernel, running_mean, running_var, gamma, beta, eps):
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def switch_to_half(self):
        if not hasattr(self, 'rbr_reparam'):
            self.switch_to_deploy()
        self.kernel_float32 = self.rbr_reparam.weight
        self.rbr_reparam.weight.data = self.rbr_reparam.weight.half()

    def switch_to_deploy(self):
        if hasattr(self, 'rbr_reparam'):
            if hasattr(self, 'kernel_float32'):
                self.rbr_reparam.weight.data = self.kernel_float32
                self.__delattr__('kernel_float32')
            return
        kernel, bias = self.get_equivalent_kernel_bias()

        self.rbr_reparam = nn.Conv2d(in_channels=self.rbr_dense.in_channels,
                                     out_channels=self.rbr_dense.out_channels,
                                     kernel_size=self.rbr_dense.kernel_size, stride=self.rbr_dense.stride,
                                     padding=self.rbr_dense.padding, dilation=self.rbr_dense.dilation,
                                     groups=self.rbr_dense.groups, bias=True)
        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__('rbr_dense')
        if hasattr(self, 'rbr_identity'):
            self.__delattr__('rbr_identity')
        if hasattr(self, 'id_tensor'):
            self.__delattr__('id_tensor')
        if hasattr(self, 'bn'):
            self.__delattr__('bn')
        self.deploy = True
    # Represented convolution
    # https://arxiv.org/abs/2101.03697

class RepNeXtBlock(nn.Module):
    '''RepVGGBlock is a basic rep-style block, including training and deploy status
    This code is based on https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py
    '''

    default_act = nn.SiLU()

    # default silu
    def __init__(self, dim, kernel_size_dense=5, kernel_size_tiny=3,
                 stride=1, padding=None, dilation=1, padding_mode='zeros', act=True, deploy=False):
        super(RepNeXtBlock, self).__init__()
        """ Initialization of the class.
        Args:
            in_channels (int): Number of channels in the input image
            out_channels (int): Number of channels produced by the convolution
            kernel_size (int or tuple): Size of the convolving kernel
            stride (int or tuple, optional): Stride of the convolution. Default: 1
            padding (int or tuple, optional): Zero-padding added to both sides of
                the input. Default: 1
            dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
            groups (int, optional): Number of blocked connections from input
                channels to output channels. Default: 1
            padding_mode (string, optional): Default: 'zeros'
            deploy: Whether to be deploy status or training status. Default: False
            use_se: Whether to use se. Default: False
        """
        self.deploy = deploy


        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()


        if deploy:
            self.rbr_reparam = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=kernel_size_dense,
                                         stride=stride,
                                         padding=autopad(kernel_size_dense, padding), dilation=dilation,
                                         groups=dim,
                                         bias=True,
                                         padding_mode=padding_mode)
        else:

            self.rbr_dense = nn.Conv2d(dim, dim, kernel_size_dense, stride, autopad(kernel_size_dense, padding),
                                       groups=dim,
                                       bias=False)

            self.rbr_3x3 = nn.Conv2d(dim, dim, kernel_size_tiny, stride, autopad(kernel_size_tiny, padding),
                                     groups=dim,
                                     bias=False)

            self.bn_5x5 = nn.BatchNorm2d(num_features=dim)
            self.bn_main = nn.BatchNorm2d(num_features=dim)

    def forward(self, inputs):
        '''Forward process'''
        if hasattr(self, 'rbr_reparam'):

            return self.act(self.bn_main(self.rbr_reparam(inputs)))

        else:
            return self.act(self.bn_main(self.bn_5x5(self.rbr_dense(inputs)) + self.rbr_3x3(inputs)))


    def get_equivalent_kernel_bias(self):
        kernel3x3_5x5 = self._pad_3x3_to_5x5_tensor(self.rbr_3x3.weight)
        kernel5x5, bias5x5 = self._fuse_bn_tensor(self.rbr_dense.weight, *self._get_bn_params(self.bn_5x5))
        kernel5x5_main, bias5x5_main = self._fuse_3x3_and_5x5_kernel(kernel5x5, kernel3x3_5x5), bias5x5

        return kernel5x5_main, bias5x5_main

    def _pad_3x3_to_5x5_tensor(self, kernel3x3):
        if kernel3x3 is None:
            return 0
        else:
            return functional.pad(kernel3x3, [1, 1, 1, 1])


    def _get_bn_params(self, branch):
        assert isinstance(branch, nn.BatchNorm2d)
        running_mean = branch.running_mean
        running_var = branch.running_var
        gamma = branch.weight
        beta = branch.bias
        eps = branch.eps
        return running_mean, running_var, gamma, beta, eps

    def _fuse_3x3_and_5x5_kernel(self, k3x3, k5x5):
        return k3x3 + k5x5

    def _fuse_bn_tensor(self, kernel, running_mean, running_var, gamma, beta, eps):
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def switch_to_half(self):
        if not hasattr(self, 'rbr_reparam'):
            self.switch_to_deploy()
        self.kernel_float32 = self.rbr_reparam.weight
        self.rbr_reparam.weight.data = self.rbr_reparam.weight.half()

    def switch_to_deploy(self):
        if hasattr(self, 'rbr_reparam'):
            if hasattr(self, 'kernel_float32'):
                self.rbr_reparam.weight.data = self.kernel_float32
                self.__delattr__('kernel_float32')
            return
        kernel, bias = self.get_equivalent_kernel_bias()

        self.rbr_reparam = nn.Conv2d(in_channels=self.rbr_dense.in_channels,
                                     out_channels=self.rbr_dense.out_channels,
                                     kernel_size=self.rbr_dense.kernel_size, stride=self.rbr_dense.stride,
                                     padding=self.rbr_dense.padding, dilation=self.rbr_dense.dilation,
                                     groups=self.rbr_dense.groups, bias=True)
        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__('rbr_dense')
        self.__delattr__('rbr_3x3')
        if hasattr(self, 'bn_5x5'):
            self.__delattr__('bn_5x5')
        self.deploy = True
    # Represented convolution
    # https://arxiv.org/abs/2101.03697


class QARepVGGBlock(RepVGGBlock):
    """
    RepVGGBlock is a basic rep-style block, including training and deploy status
    This code is based on https://arxiv.org/abs/2212.01593
    """

    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=1, padding=1, dilation=1, groups=1, padding_mode='zeros', act=nn.SiLU, deploy=False,
                 use_se=False):
        super(QARepVGGBlock, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups,
                                            padding_mode, act, deploy, use_se)
        if not deploy:
            self.bn = nn.BatchNorm2d(out_channels)
            self.rbr_1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, groups=groups, bias=False)
            self.rbr_identity = nn.Identity() if out_channels == in_channels and stride == 1 else None
        self._id_tensor = None

    def forward(self, inputs):
        if hasattr(self, 'rbr_reparam'):
            return self.act(self.bn(self.se(self.rbr_reparam(inputs))))

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)

        return self.act(self.bn(self.se(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out)))

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel = kernel3x3 + self._pad_1x1_to_3x3_tensor(self.rbr_1x1.weight)
        bias = bias3x3

        if self.rbr_identity is not None:
            input_dim = self.in_channels // self.groups
            kernel_value = np.zeros((self.in_channels, input_dim, 3, 3), dtype=np.float32)
            for i in range(self.in_channels):
                kernel_value[i, i % input_dim, 1, 1] = 1
            id_tensor = torch.from_numpy(kernel_value).to(self.rbr_1x1.weight.device)
            kernel = kernel + id_tensor
        return kernel, bias

    def _fuse_extra_bn_tensor(self, kernel, bias, branch):
        assert isinstance(branch, nn.BatchNorm2d)

        running_mean = branch.running_mean - bias  # remove bias
        running_var = branch.running_var
        gamma = branch.weight
        beta = branch.bias
        eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def switch_to_deploy(self):
        if hasattr(self, 'rbr_reparam'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.rbr_reparam = nn.Conv2d(in_channels=self.in_channels,
                                     out_channels=self.out_channels,
                                     kernel_size=self.kernel_size, stride=self.stride,
                                     padding=self.padding, dilation=self.dilation,
                                     groups=self.groups, bias=True)
        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__('rbr_dense')
        self.__delattr__('rbr_1x1')
        if hasattr(self, 'rbr_identity'):
            self.__delattr__('rbr_identity')
        if hasattr(self, 'id_tensor'):
            self.__delattr__('id_tensor')
        # keep post bn for QAT
        # if hasattr(self, 'bn'):
        #     self.__delattr__('bn')
        self.deploy = True


class RepBlock(nn.Module):
    '''
        RepBlock is a stage block with rep-style basic block
    '''

    # default silu
    def __init__(self, in_channels, out_channels, n=1, block=RepVGGBlock, act=nn.SiLU):
        super().__init__()

        self.conv1 = block(in_channels, out_channels, act=act)
        self.block = nn.Sequential(
            *(block(out_channels, out_channels, act=act) for _ in range(n - 1))) if n > 1 else None

    def forward(self, x):
        x = self.conv1(x)
        if self.block is not None:
            x = self.block(x)
        return x


class FastRepBlock(nn.Module):
    '''
        RepBlock is a stage block with rep-style basic block
    '''

    # default silu
    def __init__(self, in_channels, out_channels, n=1, block=FastRepVGGBlock, act=True):
        super().__init__()

        self.conv1 = block(in_channels, out_channels, act=act)
        self.block = nn.Sequential(
            *(block(out_channels, out_channels, act=act) for _ in range(n - 1))) if n > 1 else None

    def forward(self, x):
        x = self.conv1(x)
        if self.block is not None:
            x = self.block(x)
        return x


class FastRepV2Block(FastRepBlock):
    '''
        RepBlock is a stage block with rep-style basic block
    '''

    # default silu
    def __init__(self, in_channels, out_channels, n=1, block=FastRepVGGBlockV2, act=True):
        super().__init__(in_channels, out_channels, n, block=block, act=act)


class FastRepV3Block(FastRepBlock):
    '''
        RepBlock is a stage block with rep-style basic block
    '''

    # default silu
    def __init__(self, in_channels, out_channels, n=1, block=FastRepVGGBlockV3, act=True):
        super().__init__(in_channels, out_channels, n, block=block, act=act)


class FastRepV4Block(nn.Module):
    '''
        RepBlock is a stage block with rep-style basic block
    '''

    def __init__(self, in_channels, out_channels, n=1, block=FastRepVGGBlockV4, act=True, shortcut=False):
        super().__init__()

        self.conv1 = block(in_channels, out_channels, act=act)
        self.block = nn.Sequential(
            *(block(out_channels, out_channels, act=act) for _ in range(n - 1))) if n > 1 else None

    def forward(self, x):
        x = self.conv1(x)
        if self.block is not None:
            x = self.block(x)
        return x


class FastRepDBlock(FastRepBlock):
    '''
        RepBlock is a stage block with rep-style basic block
    '''

    # default silu
    def __init__(self, in_channels, out_channels, n=1, w=1, block=FastRepVGGBlockD, act=True):
        super().__init__(in_channels, out_channels, n, block=block, act=act)
        self.conv1 = block(in_channels, out_channels, width1x1=w, act=act)
        self.block = nn.Sequential(
            *(block(out_channels, out_channels, width1x1=w, act=act) for _ in range(n - 1))) if n > 1 else None


class QARepBlock(nn.Module):
    '''
        RepBlock is a stage block with rep-style basic block
    '''

    # default silu
    def __init__(self, in_channels, out_channels, n=1, block=QARepVGGBlock):
        super().__init__()

        self.conv1 = block(in_channels, out_channels)
        self.block = nn.Sequential(
            *(block(out_channels, out_channels) for _ in range(n - 1))) if n > 1 else None

    def forward(self, x):
        x = self.conv1(x)
        if self.block is not None:
            x = self.block(x)
        return x


class SRepBottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):  # ch_in, ch_out, shortcut, kernels, groups, expand
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = SRepConv(c1, c_, k[0], 1)
        self.cv2 = SRepConv(c_, c2, k[1], 1, groups=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class C3fBlock(nn.Module):

    def __init__(self, dim, n, block=FastRepV3Block, k=3):
        super().__init__()
        self.blocks = block(dim, dim, n=n)
        self.conv_k = SRepConv(dim, dim, k, 1)

    def forward(self, x):
        return self.conv_k(self.blocks(x))

class C3sBlock(nn.Module):

    def __init__(self, dim, n, shortcut=False, block=FastRepV4Block, k=3):
        super().__init__()
        self.blocks = block(dim, dim, n=n)
        self.conv_k = SRepConv(dim, dim, k, 1)
        self.shortcut = shortcut

    def forward(self, x):
        return x + self.conv_k(self.blocks(x)) if self.shortcut else self.conv_k(self.blocks(x))


class C3tBlock(nn.Module):

    def __init__(self, dim, n, shortcut=False, block=FastRepV4Block, k=3):
        super().__init__()
        self.blocks = block(dim, dim, n=n)
        self.conv_k = Conv(dim, dim, k, 1)
        self.shortcut = shortcut

    def forward(self, x):
        return x + self.blocks(self.conv_k(x)) if self.shortcut else self.conv_k(self.blocks(x))

class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):  # ch_in, ch_out, shortcut, kernels, groups, expand
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class InvertBottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3, 1, 1), e=4):  # ch_in, ch_out, shortcut, kernels, groups, expand
        super().__init__()
        c_ = int(c1 * e)  # hidden channels
        self.cv1 = Conv(c1, c1, k[0], 1)
        self.cv2 = Conv(c1, c1, k[1], 1, g=g)
        self.cv3 = Conv(c1, c_, k[2], 1, g=g)
        self.cv4 = Conv(c_, c2, k[3], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv4(self.cv3(self.cv2(self.cv1(x)))) if self.add else self.cv4(self.cv3(self.cv2(self.cv1(x))))


class GhostBottleneck(nn.Module):
    # Ghost Bottleneck https://github.com/huawei-noah/ghostnet
    def __init__(self, c1, c2, k=3, s=1):  # ch_in, ch_out, kernel, stride
        super().__init__()
        c_ = c2 // 2
        self.conv = nn.Sequential(
            GhostConv(c1, c_, 1, 1),  # pw
            DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(),  # dw
            GhostConv(c_, c2, 1, 1, act=False))  # pw-linear
        self.shortcut = nn.Sequential(DWConv(c1, c1, k, s, act=False), Conv(c1, c2, 1, 1,
                                                                            act=False)) if s == 2 else nn.Identity()

    def forward(self, x):
        return self.conv(x) + self.shortcut(x)


class RTMDetBottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, k=(3, 5), e=0.5):  # ch_in, ch_out, shortcut, kernels, groups, expand
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = DWConv(c_, c2, k[1], 1)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class RTMDet2Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, k=(3, 5, 3), e=0.5):  # ch_in, ch_out, shortcut, kernels, groups, expand
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = DWConv(c_, c_, k[1], 1)
        self.cv3 = Conv(c_, c2, k[2], 1)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv3(self.cv2(self.cv1(x))) if self.add else self.cv3(self.cv2(self.cv1(x)))


class RepNeXtBottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, k=(3, {'k_dense':5, 'k_tiny': 3},  3), e=0.5):  # ch_in, ch_out, shortcut, kernels, groups, expand
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = RepNeXtBlock(c_, kernel_size_dense=k[1]['k_dense'], kernel_size_tiny=k[1]['k_tiny'], stride=1)
        self.cv3 = Conv(c_, c2, k[2], 1)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv3(self.cv2(self.cv1(x))) if self.add else self.cv3(self.cv2(self.cv1(x)))
