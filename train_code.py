import os
import copy
import xml.etree.ElementTree as ET
from sklearn.preprocessing import LabelEncoder
from torchvision.datasets.folder import has_file_allowed_extension, default_loader
import time
import random
import shutil
import warnings
import numpy as np
import torch
import torch as th
import torch.nn as nn
import torch.utils.data
import torchvision.utils as vutils
import torch.nn.functional as F
from torch.optim import Adam
from torch.nn import Parameter
from torchvision.utils import save_image
from torch.nn.functional import interpolate
from torchvision.transforms import transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from torch.nn import Conv2d, BCEWithLogitsLoss, DataParallel, AvgPool2d, ModuleList, LeakyReLU, ConvTranspose2d, Embedding
from PIL import Image
from tqdm import tqdm

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def get_transforms(size):
    base_transforms = transforms.Compose([transforms.Resize(size)])
    additional_transforms = transforms.Compose([transforms.RandomHorizontalFlip(p=0.5), transforms.RandomChoice([transforms.CenterCrop(size), transforms.RandomCrop(size)]), transforms.RandomApply([transforms.ColorJitter(brightness=0.2, contrast=(0.9, 1.2), saturation=0.3, hue=0.01)], p=0.5), transforms.ToTensor(), Rescale()])
    return (base_transforms, additional_transforms)
WEIGHTS_PATH = '../input/dog-face-generation-competition-kid-metric-input/classify_image_graph_def.pb'
model_params = {'Inception': {'name': 'Inception', 'imsize': 64, 'output_layer': 'Pretrained_Net/pool_3:0', 'input_layer': 'Pretrained_Net/ExpandDims:0', 'output_shape': 2048, 'cosine_distance_eps': 0.1}}

def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)

class SpectralNorm(nn.Module):

    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + '_u')
        v = getattr(self.module, self.name + '_v')
        w = getattr(self.module, self.name + '_bar')
        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(torch.mv(torch.t(w.view(height, -1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(height, -1).data, v.data))
        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + '_u')
            v = getattr(self.module, self.name + '_v')
            w = getattr(self.module, self.name + '_bar')
            return True
        except AttributeError:
            return False

    def _make_params(self):
        w = getattr(self.module, self.name)
        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]
        u = Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = Parameter(w.data)
        del self.module._parameters[self.name]
        self.module.register_parameter(self.name + '_u', u)
        self.module.register_parameter(self.name + '_v', v)
        self.module.register_parameter(self.name + '_bar', w_bar)

    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)

class _equalized_conv2d(th.nn.Module):
    """ conv2d with the concept of equalized learning rate
        Args:
            :param c_in: input channels
            :param c_out:  output channels
            :param k_size: kernel size (h, w) should be a tuple or a single integer
            :param stride: stride for conv
            :param pad: padding
            :param bias: whether to use bias or not
    """

    def __init__(self, c_in, c_out, k_size, stride=1, pad=0, bias=True):
        """ constructor for the class """
        from torch.nn.modules.utils import _pair
        from numpy import sqrt, prod
        super(_equalized_conv2d, self).__init__()
        self.weight = th.nn.Parameter(th.nn.init.normal_(th.empty(c_out, c_in, *_pair(k_size))))
        self.use_bias = bias
        self.stride = stride
        self.pad = pad
        if self.use_bias:
            self.bias = th.nn.Parameter(th.FloatTensor(c_out).fill_(0))
        fan_in = prod(_pair(k_size)) * c_in
        self.scale = sqrt(2) / sqrt(fan_in)

    def forward(self, x):
        """
        forward pass of the network
        :param x: input
        :return: y => output
        """
        from torch.nn.functional import conv2d
        return conv2d(input=x, weight=self.weight * self.scale, bias=self.bias if self.use_bias else None, stride=self.stride, padding=self.pad)

    def extra_repr(self):
        return ', '.join(map(str, self.weight.shape))

class _equalized_deconv2d(th.nn.Module):
    """ Transpose convolution using the equalized learning rate
        Args:
            :param c_in: input channels
            :param c_out: output channels
            :param k_size: kernel size
            :param stride: stride for convolution transpose
            :param pad: padding
            :param bias: whether to use bias or not
    """

    def __init__(self, c_in, c_out, k_size, stride=1, pad=0, bias=True):
        """ constructor for the class """
        from torch.nn.modules.utils import _pair
        from numpy import sqrt
        super(_equalized_deconv2d, self).__init__()
        self.weight = th.nn.Parameter(th.nn.init.normal_(th.empty(c_in, c_out, *_pair(k_size))))
        self.use_bias = bias
        self.stride = stride
        self.pad = pad
        if self.use_bias:
            self.bias = th.nn.Parameter(th.FloatTensor(c_out).fill_(0))
        fan_in = c_in
        self.scale = sqrt(2) / sqrt(fan_in)

    def forward(self, x):
        """
        forward pass of the layer
        :param x: input
        :return: y => output
        """
        from torch.nn.functional import conv_transpose2d
        return conv_transpose2d(input=x, weight=self.weight * self.scale, bias=self.bias if self.use_bias else None, stride=self.stride, padding=self.pad)

    def extra_repr(self):
        return ', '.join(map(str, self.weight.shape))

class _equalized_linear(th.nn.Module):
    """ Linear layer using equalized learning rate
        Args:
            :param c_in: number of input channels
            :param c_out: number of output channels
            :param bias: whether to use bias with the linear layer
    """

    def __init__(self, c_in, c_out, bias=True):
        """
        Linear layer modified for equalized learning rate
        """
        from numpy import sqrt
        super(_equalized_linear, self).__init__()
        self.weight = th.nn.Parameter(th.nn.init.normal_(th.empty(c_out, c_in)))
        self.use_bias = bias
        if self.use_bias:
            self.bias = th.nn.Parameter(th.FloatTensor(c_out).fill_(0))
        fan_in = c_in
        self.scale = sqrt(2) / sqrt(fan_in)

    def forward(self, x):
        """
        forward pass of the layer
        :param x: input
        :return: y => output
        """
        from torch.nn.functional import linear
        return linear(x, self.weight * self.scale, self.bias if self.use_bias else None)

class PixelwiseNorm(th.nn.Module):

    def __init__(self):
        super(PixelwiseNorm, self).__init__()

    def forward(self, x, alpha=1e-08):
        """
        forward pass of the module
        :param x: input activations volume
        :param alpha: small number for numerical stability
        :return: y => pixel normalized activations
        """
        y = x.pow(2.0).mean(dim=1, keepdim=True).add(alpha).sqrt()
        y = x / y
        return y

class MinibatchStdDev(th.nn.Module):
    """
    Minibatch standard deviation layer for the discriminator
    """

    def __init__(self):
        """
        derived class constructor
        """
        super(MinibatchStdDev, self).__init__()

    def forward(self, x, alpha=1e-08):
        """
        forward pass of the layer
        :param x: input activation volume
        :param alpha: small number for numerical stability
        :return: y => x appended with standard deviation constant map
        """
        batch_size, _, height, width = x.shape
        y = x - x.mean(dim=0, keepdim=True)
        y = th.sqrt(y.pow(2.0).mean(dim=0, keepdim=False) + alpha)
        y = y.mean().view(1, 1, 1, 1)
        y = y.repeat(batch_size, 1, height, width)
        y = th.cat([x, y], 1)
        return y

class GenInitialBlock(th.nn.Module):

    def __init__(self, in_channels, use_eql, use_spec_norm=False):
        super(GenInitialBlock, self).__init__()
        if use_eql:
            self.conv_1 = _equalized_deconv2d(in_channels, in_channels, (4, 4), bias=True)
            self.conv_2 = _equalized_conv2d(in_channels, in_channels, (3, 3), pad=1, bias=True)
        else:
            self.conv_1 = ConvTranspose2d(in_channels, in_channels, (4, 4), bias=True)
            self.conv_2 = Conv2d(in_channels, in_channels, (3, 3), padding=1, bias=True)
        if use_spec_norm:
            self.conv_1 = SpectralNorm(self.conv_1)
            self.conv_2 = SpectralNorm(self.conv_2)
        self.pixNorm = PixelwiseNorm()
        self.lrelu = LeakyReLU(0.2)

    def forward(self, x):
        y = th.unsqueeze(th.unsqueeze(x, -1), -1)
        y = self.lrelu(self.conv_1(y))
        y = self.lrelu(self.conv_2(y))
        y = self.pixNorm(y)
        return y

class GenGeneralConvBlock(th.nn.Module):

    def __init__(self, in_channels, out_channels, use_eql, use_spec_norm=False):
        super(GenGeneralConvBlock, self).__init__()
        self.upsample = lambda x: interpolate(x, scale_factor=2)
        if use_eql:
            self.conv_1 = _equalized_conv2d(in_channels, out_channels, (3, 3), pad=1, bias=True)
            self.conv_2 = _equalized_conv2d(out_channels, out_channels, (3, 3), pad=1, bias=True)
        else:
            self.conv_1 = Conv2d(in_channels, out_channels, (3, 3), padding=1, bias=True)
            self.conv_2 = Conv2d(out_channels, out_channels, (3, 3), padding=1, bias=True)
        if use_spec_norm:
            self.conv_1 = SpectralNorm(self.conv_1)
            self.conv_2 = SpectralNorm(self.conv_2)
        self.pixNorm = PixelwiseNorm()
        self.lrelu = LeakyReLU(0.2)

    def forward(self, x):
        """
        forward pass of the block
        :param x: input
        :return: y => output
        """
        y = self.upsample(x)
        y = self.pixNorm(self.lrelu(self.conv_1(y)))
        y = self.pixNorm(self.lrelu(self.conv_2(y)))
        return y

class DisGeneralConvBlock(th.nn.Module):
    """ General block in the discriminator  """

    def __init__(self, in_channels, out_channels, use_eql, use_spec_norm=False):
        """
        constructor of the class
        :param in_channels: number of input channels
        :param out_channels: number of output channels
        :param use_eql: whether to use equalized learning rate
        """
        super(DisGeneralConvBlock, self).__init__()
        if use_eql:
            self.conv_1 = _equalized_conv2d(in_channels, in_channels, (3, 3), pad=1, bias=True)
            self.conv_2 = _equalized_conv2d(in_channels, out_channels, (3, 3), pad=1, bias=True)
        else:
            self.conv_1 = Conv2d(in_channels, in_channels, (3, 3), padding=1, bias=True)
            self.conv_2 = Conv2d(in_channels, out_channels, (3, 3), padding=1, bias=True)
        if use_spec_norm:
            self.conv_1 = SpectralNorm(self.conv_1)
            self.conv_2 = SpectralNorm(self.conv_2)
        self.downSampler = AvgPool2d(2)
        self.lrelu = LeakyReLU(0.2)

    def forward(self, x):
        y = self.lrelu(self.conv_1(x))
        y = self.lrelu(self.conv_2(y))
        y = self.downSampler(y)
        return y

class ConDisFinalBlock(th.nn.Module):

    def __init__(self, in_channels, num_classes, use_eql):
        super(ConDisFinalBlock, self).__init__()
        self.batch_discriminator = MinibatchStdDev()
        if use_eql:
            self.conv_1 = _equalized_conv2d(in_channels + 1, in_channels, (3, 3), pad=1, bias=True)
            self.conv_2 = _equalized_conv2d(in_channels, in_channels, (4, 4), bias=True)
            self.conv_3 = _equalized_conv2d(in_channels, 1, (1, 1), bias=True)
        else:
            self.conv_1 = Conv2d(in_channels + 1, in_channels, (3, 3), padding=1, bias=True)
            self.conv_2 = Conv2d(in_channels, in_channels, (4, 4), bias=True)
            self.conv_3 = Conv2d(in_channels, 1, (1, 1), bias=True)
        self.label_embedder = Embedding(num_classes, in_channels)
        self.lrelu = LeakyReLU(0.2)
        nb_ft = 128
        self.ft_matching_dense = nn.Linear(2 * in_channels, nb_ft)

    def forward(self, x, labels, return_ft=False):
        """
        forward pass of the FinalBlock
        :param x: input
        :param labels: samples' labels for conditional discrimination
                       Note that these are pure integer labels [Batch_size x 1]
        :return: y => output
        """
        batch_size = x.size()[0]
        y = self.batch_discriminator(x)
        y = self.lrelu(self.conv_1(y))
        y = self.lrelu(self.conv_2(y))
        y_ = y.view((batch_size, -1))
        
        # FIX FOR DATAPARALLEL: Embedding with max_norm modifies in-place. 
        # We must clone before doing a view/reshape to prevent the BroadcastBackward crash on 2 GPUs.
        embedded_labels = self.label_embedder(labels.cuda())
        labels = embedded_labels.clone().view((batch_size, -1))
        
        if return_ft:
            self.ft_matching_dense(torch.cat((y_, labels), 1))
        projection_scores = (y_ * labels).sum(dim=-1)
        y = self.lrelu(self.conv_3(y))
        final_score = y.view(-1) + projection_scores
        return final_score

class GaussianNoise(nn.Module):

    def __init__(self, sigma=0.1):
        super().__init__()
        self.sigma = sigma
        self.noise = torch.tensor(0).cuda()

    def forward(self, x):
        if self.training:
            noise = self.noise.repeat(*x.size()).float().normal_() * self.sigma
            return x + noise
        return x

class Generator(nn.Module):

    def __init__(self, depth=5, latent_size=128, use_eql=True, use_spec_norm=False):
        super(Generator, self).__init__()
        assert latent_size != 0 and latent_size & latent_size - 1 == 0, 'latent size not a power of 2'
        if depth >= 4:
            assert latent_size >= np.power(2, depth - 4), 'latent size will diminish to zero'
        self.use_eql = use_eql
        self.use_spec_norm = use_spec_norm
        self.depth = depth
        self.latent_size = latent_size
        self.initial_block = GenInitialBlock(self.latent_size, use_eql=self.use_eql, use_spec_norm=False)
        self.layers = ModuleList([])
        if self.use_eql:
            self.toRGB = lambda in_channels: _equalized_conv2d(in_channels, 3, (1, 1), bias=True)
        else:
            self.toRGB = lambda in_channels: Conv2d(in_channels, 3, (1, 1), bias=True)
        self.rgb_converters = ModuleList([self.toRGB(self.latent_size)])
        for i in range(self.depth - 1):
            if i <= 2:
                layer = GenGeneralConvBlock(self.latent_size, self.latent_size, use_eql=self.use_eql, use_spec_norm=use_spec_norm)
                rgb = self.toRGB(self.latent_size)
            else:
                in_size = int(self.latent_size // np.power(2, i - 3))
                out_size = int(self.latent_size // np.power(2, i - 2))
                layer = nn.Sequential(GenGeneralConvBlock(in_size, out_size, use_eql=self.use_eql, use_spec_norm=use_spec_norm))
                rgb = self.toRGB(out_size)
            self.layers.append(layer)
            self.rgb_converters.append(rgb)
        self.temporaryUpsampler = lambda x: interpolate(x, scale_factor=2)
        self.tanh = nn.Tanh()

    def forward(self, x, depth, alpha):
        assert depth < self.depth, 'Requested output depth cannot be produced'
        y = self.initial_block(x)
        if depth > 0:
            for block in self.layers[:depth - 1]:
                y = block(y)
            residual = self.rgb_converters[depth - 1](self.temporaryUpsampler(y))
            straight = self.rgb_converters[depth](self.layers[depth - 1](y))
            out = alpha * straight + (1 - alpha) * residual
        else:
            out = self.rgb_converters[0](y)
        return self.tanh(out)

class ConditionalDiscriminator(nn.Module):

    def __init__(self, num_classes, height=7, feature_size=512, use_eql=True, use_spec_norm=False):
        super(ConditionalDiscriminator, self).__init__()
        assert feature_size != 0 and feature_size & feature_size - 1 == 0, 'latent size not a power of 2'
        if height >= 4:
            assert feature_size >= np.power(2, height - 4), 'feature size cannot be produced'
        self.use_eql = use_eql
        self.use_spec_norm = use_spec_norm
        self.height = height
        self.feature_size = feature_size
        self.num_classes = num_classes
        self.noise = GaussianNoise(sigma=0.2)
        self.final_block = ConDisFinalBlock(self.feature_size, self.num_classes, use_eql=self.use_eql)
        self.layers = ModuleList([])
        if self.use_eql:
            self.fromRGB = lambda out_channels: _equalized_conv2d(3, out_channels, (1, 1), bias=True)
        else:
            self.fromRGB = lambda out_channels: Conv2d(3, out_channels, (1, 1), bias=True)
        rgb = self.fromRGB(self.feature_size)
        if use_spec_norm:
            rgb = SpectralNorm(rgb)
        self.rgb_to_features = ModuleList([rgb])
        for i in range(self.height - 1):
            if i > 2:
                in_size = int(self.feature_size // np.power(2, i - 2))
                out_size = int(self.feature_size // np.power(2, i - 3))
                layer = nn.Sequential(DisGeneralConvBlock(in_size, out_size, use_eql=self.use_eql, use_spec_norm=use_spec_norm))
                rgb = self.fromRGB(in_size)
            else:
                layer = nn.Sequential(DisGeneralConvBlock(self.feature_size, self.feature_size, use_eql=self.use_eql, use_spec_norm=use_spec_norm))
                rgb = self.fromRGB(self.feature_size)
            if use_spec_norm:
                rgb = SpectralNorm(rgb)
            self.layers.append(layer)
            self.rgb_to_features.append(rgb)
        self.temporaryDownsampler = AvgPool2d(2)

    def forward(self, x, labels, height, alpha, return_ft=False):
        assert height < self.height, 'Requested output depth cannot be produced'
        if height > 0:
            residual = self.rgb_to_features[height - 1](self.temporaryDownsampler(x))
            straight = self.layers[height - 1](self.rgb_to_features[height](x))
            y = alpha * straight + (1 - alpha) * residual
            for block in reversed(self.layers[:height - 1]):
                y = block(y)
        else:
            y = self.rgb_to_features[0](x)
        out = self.final_block(y, labels, return_ft=return_ft)
        return out

class ConditionalGANLoss:
    """ Base class for all conditional losses """

    def __init__(self, dis):
        self.dis = dis

    def dis_loss(self, real_samps, fake_samps, labels, height, alpha):
        raise NotImplementedError('dis_loss method has not been implemented')

    def gen_loss(self, real_samps, fake_samps, labels, height, alpha):
        raise NotImplementedError('gen_loss method has not been implemented')

class StandardLoss(ConditionalGANLoss):

    def __init__(self, dis):
        super().__init__(dis)
        self.criterion = BCEWithLogitsLoss(reduction='sum')

    def dis_loss(self, real_samps, fake_samps, labels, height, alpha):
        assert real_samps.device == fake_samps.device, 'Different devices'
        preds_real = self.dis(real_samps, labels, height, alpha)
        preds_fake = self.dis(fake_samps, labels, height, alpha)
        labels_real = torch.from_numpy(np.random.uniform(0.5, 0.99, real_samps.size()[0])).float().cuda()
        labels_fake = torch.from_numpy(np.random.uniform(0, 0.25, fake_samps.size()[0])).float().cuda()
        real_loss = self.criterion(preds_real.view(-1), labels_real)
        fake_loss = self.criterion(preds_fake.view(-1), labels_fake)
        return real_loss + fake_loss

    def gen_loss(self, _, fake_samps, labels, height, alpha):
        preds_fake = self.dis(fake_samps, labels, height, alpha)
        labels_real = torch.from_numpy(np.random.uniform(0.5, 0.99, fake_samps.size()[0])).float().cuda()
        return self.criterion(preds_fake.view(-1), labels_real)

class Hinge(ConditionalGANLoss):

    def __init__(self, dis):
        super().__init__(dis)

    def dis_loss(self, real_samps, fake_samps, labels, height, alpha):
        r_preds = self.dis(real_samps, labels, height, alpha)
        f_preds = self.dis(fake_samps, labels, height, alpha)
        loss = torch.mean(th.nn.ReLU()(1 - r_preds)) + torch.mean(th.nn.ReLU()(1 + f_preds))
        return loss

    def gen_loss(self, _, fake_samps, labels, height, alpha):
        return -torch.mean(self.dis(fake_samps, labels, height, alpha))

def update_average(model_old, model_new, beta):

    def toggle_grad(model, requires_grad):
        for p in model.parameters():
            p.requires_grad_(requires_grad)
    toggle_grad(model_old, False)
    toggle_grad(model_new, False)
    param_dict_new = dict(model_new.named_parameters())
    for param_name, param_old in model_old.named_parameters():
        param_new = param_dict_new[param_name]
        assert param_old is not param_new
        param_old.copy_(beta * param_old + (1.0 - beta) * param_new)
    toggle_grad(model_old, True)
    toggle_grad(model_new, True)

def save_model_weights(model, filename, verbose=1):
    if verbose:
        print(f'-> Saving weights to {filename}')
    torch.save(model.state_dict(), filename)

def load_model_weights(model, filename, verbose=1):
    if verbose:
        print(f'-> Loading weights from {filename}')
    model.load_state_dict(torch.load(filename))
    return model

class ConditionalProGAN:

    def __init__(self, num_classes=120, depth=7, latent_size=128, embed_dim=64, lr_g=0.001, lr_d=0.001, n_critic=1, use_eql=True, use_spec_norm=False, loss=StandardLoss, use_ema=True, ema_decay=0.999):
        self.gen = Generator(depth=depth, latent_size=latent_size, use_eql=use_eql, use_spec_norm=False).cuda()
        self.dis = ConditionalDiscriminator(num_classes, height=depth, feature_size=latent_size, use_eql=use_eql, use_spec_norm=use_spec_norm).cuda()
        self.gen = DataParallel(self.gen)
        self.dis = DataParallel(self.dis)
        self.latent_size = latent_size
        self.num_classes = num_classes
        self.depth = depth
        self.use_ema = use_ema
        self.ema_decay = ema_decay
        self.n_critic = n_critic
        self.use_eql = use_eql
        self.drift = 0.001
        self.lr_g = lr_g
        self.lr_d = lr_d
        self.gen_optim = Adam(self.gen.parameters(), lr=self.lr_g, betas=(0.5, 0.99), eps=1e-08)
        self.dis_optim = Adam(self.dis.parameters(), lr=self.lr_d, betas=(0.5, 0.99), eps=1e-08)
        try:
            self.loss = loss(self.dis)
        except:
            self.loss = loss(self.dis, drift=self.drift, use_gp=True)
        if self.use_ema:
            self.gen_shadow = copy.deepcopy(self.gen)
            self.ema_updater = update_average
            self.ema_updater(self.gen_shadow, self.gen, beta=0)

    def __progressive_downsampling(self, real_batch, depth, alpha):
        """
        private helper for downsampling the original images in order to facilitate the
        progressive growing of the layers.
        :param real_batch: batch of real samples
        :param depth: depth at which training is going on
        :param alpha: current value of the fader alpha
        :return: real_samples => modified real batch of samples
        """
        down_sample_factor = int(np.power(2, self.depth - depth - 1))
        prior_downsample_factor = max(int(np.power(2, self.depth - depth)), 0)
        ds_real_samples = AvgPool2d(down_sample_factor)(real_batch)
        if depth > 0:
            prior_ds_real_samples = interpolate(AvgPool2d(prior_downsample_factor)(real_batch), scale_factor=2)
        else:
            prior_ds_real_samples = ds_real_samples
        real_samples = alpha * ds_real_samples + (1 - alpha) * prior_ds_real_samples
        return real_samples

    def optimize_discriminator(self, noise, real_batch, labels, depth, alpha):
        real_samples = self.__progressive_downsampling(real_batch, depth, alpha)
        loss_val = 0
        for _ in range(self.n_critic):
            fake_samples = self.gen(noise, depth, alpha).detach()
            loss = self.loss.dis_loss(real_samples, fake_samples, labels, depth, alpha)
            self.dis_optim.zero_grad()
            loss.backward()
            self.dis_optim.step()
            loss_val += loss.item()
        return loss_val / self.n_critic

    def optimize_generator(self, noise, real_batch, labels, depth, alpha):
        real_samples = self.__progressive_downsampling(real_batch, depth, alpha)
        fake_samples = self.gen(noise, depth, alpha)
        loss = self.loss.gen_loss(real_samples, fake_samples, labels, depth, alpha)
        self.gen_optim.zero_grad()
        loss.backward()
        self.gen_optim.step()
        if self.use_ema:
            self.ema_updater(self.gen_shadow, self.gen, self.ema_decay)
        return loss.item()

    def one_hot_encode(self, labels):
        if not hasattr(self, 'label_oh_encoder'):
            self.label_oh_encoder = th.nn.Embedding(self.num_classes, self.num_classes)
            self.label_oh_encoder.weight.data = th.eye(self.num_classes)
        return self.label_oh_encoder(labels.view(-1))

    @staticmethod
    def scale(imgs):

        def norm(img, inf, sup):
            img.clamp_(min=inf, max=sup)
            img.add_(-inf).div_(sup - inf + 1e-05)
        for img in imgs:
            norm(img, float(img.min()), float(img.max()))

    @staticmethod
    def truncated_normal(size, threshold=1):
        values = truncnorm.rvs(-threshold, threshold, size=size)
        return values

    def generate(self, depth=None, alpha=1, noise=None, races=None, n=64, n_plot=0):
        if depth is None:
            depth = self.depth - 1
        if noise is None:
            noise = th.randn(n, self.latent_size - self.num_classes).cuda()
        if races is None:
            races = torch.from_numpy(np.random.choice(range(self.num_classes), size=n)).long()
        label_information = self.one_hot_encode(races).cuda()
        gan_input = th.cat((label_information, noise), dim=-1)
        if self.use_ema:
            generated_images = self.gen_shadow(gan_input, depth, alpha).detach().cpu()
        else:
            generated_images = self.gen(gan_input, depth, alpha).detach().cpu()
        generated_images.add_(1).div_(2)
        images = generated_images.clone().numpy().transpose(0, 2, 3, 1)
        if n_plot >= 5:
            plt.figure(figsize=(15, 3 * n_plot // 5))
            for i in range(n_plot):
                plt.subplot(n_plot // 5, 5, i + 1)
                plt.imshow(images[i])
                plt.axis('off')
                plt.title(dataset.classes[races.cpu().numpy()[i]])
            plt.show()
        return generated_images

    def generate_score(self, depth=None, alpha=1, noise=None, races=None, n=64, n_plot=0):
        if depth is None:
            depth = self.depth - 1
        if noise is None:
            noise = th.randn(n, self.latent_size - self.num_classes).cuda()
        if races is None:
            races = torch.from_numpy(np.random.choice(range(self.num_classes), size=n)).long()
        label_information = self.one_hot_encode(races).cuda()
        gan_input = th.cat((label_information, noise), dim=-1)
        if self.use_ema:
            generated_images = self.gen_shadow(gan_input, depth, alpha).detach().cpu()
        else:
            generated_images = self.gen(gan_input, depth, alpha).detach().cpu()
        generated_images.add_(1).div_(2)
        images = generated_images.clone().numpy().transpose(0, 2, 3, 1)
        scores = nn.Sigmoid()(self.dis(generated_images, races, depth, alpha)).cpu().detach().numpy()
        return (images, generated_images, scores, races.cpu().numpy())

    def plot_race(self, race_idx, depth=4, alpha=1, n_plot=5, n=128):
        races = np.concatenate((np.array([race_idx] * n_plot), np.random.choice(range(self.num_classes), size=n - n_plot)))
        races = torch.from_numpy(races).long()
        self.generate(depth, alpha=alpha, races=races, n=n, n_plot=n_plot)

    def compute_mifid(self, alpha=1, folder='../tmp_images', n_images=10000, im_batch_size=100):
        if os.path.exists(folder):
            shutil.rmtree(folder, ignore_errors=True)
        os.mkdir(folder)
        for i_b in range(0, n_images, im_batch_size):
            gen_images = self.generate(n=im_batch_size)
            for i_img in range(gen_images.size(0)):
                save_image(gen_images[i_img, :, :, :], os.path.join(folder, f'img_{i_b + i_img}.png'))
        if len(os.listdir('../tmp_images')) != n_images:
            print(len(os.listdir('../tmp_images')))
        mifid = compute_mifid(folder, DATA_PATH, WEIGHTS_PATH, model_params)
        shutil.rmtree(folder, ignore_errors=True)
        return mifid

    def train(self, dataset, epochs, batch_sizes, fade_in_percentage, ema_decays, start_depth=0, verbose=1):
        assert self.depth == len(batch_sizes), 'batch_sizes not compatible with depth'
        infos = {'resolution': [], 'discriminator_loss': [], 'generator_loss': []}
        self.gen.train()
        self.dis.train()
        if self.use_ema:
            self.gen_shadow.train()
        fixed_noise = torch.randn(128, self.latent_size - self.num_classes).cuda()
        fixed_races = torch.from_numpy(np.random.choice(range(self.num_classes), size=128)).long()
        for current_depth in range(start_depth, self.depth):
            current_res = np.power(2, current_depth + 2)
            print('\n   -> Current resolution: %d x %d \n' % (current_res, current_res))
            data = torch.utils.data.DataLoader(dataset, batch_size=batch_sizes[current_depth], num_workers=4, shuffle=True)
            self.ema_decay = ema_decays[current_depth]
            ticker = 1
            for epoch in range(1, epochs[current_depth] + 1):
                start_time = time.time()
                d_loss = 0
                g_loss = 0
                fader_point = fade_in_percentage[current_depth] // 100 * epochs[current_depth] * len(iter(data))
                step = 0
                if current_res == 64 and epoch % 50 == 0:
                    self.ema_decay = 0.9 + self.ema_decay / 10
                for i, batch in enumerate(data, 1):
                    alpha = ticker / fader_point if ticker <= fader_point else 1
                    images, labels = batch
                    images = images.cuda()
                    labels = labels.view(-1, 1)
                    label_information = self.one_hot_encode(labels).cuda()
                    latent_vector = th.randn(images.shape[0], self.latent_size - self.num_classes).cuda()
                    gan_input = th.cat((label_information, latent_vector), dim=-1)
                    dis_loss = self.optimize_discriminator(gan_input, images, labels, current_depth, alpha)
                    d_loss += dis_loss / len(data)
                    gen_loss = self.optimize_generator(gan_input, images, labels, current_depth, alpha)
                    g_loss += gen_loss / len(data)
                    ticker += 1
                    step += 1
                infos['discriminator_loss'].append(d_loss)
                infos['generator_loss'].append(g_loss)
                infos['resolution'].append(current_res)
                if epoch % verbose == 0:
                    elapsed_time = time.time() - start_time
                    print(f'Epoch {epoch}/{epochs[current_depth]}     lr_g={self.lr_g:.1e}     lr_d={self.lr_d:.1e}     ema_decay={self.ema_decay:.4f}', end='     ')
                    print(f'disc_loss={d_loss:.3f}     gen_loss={g_loss:.3f}     t={elapsed_time:.0f}s')
                if epoch % (verbose * 5) == 0:
                    os.makedirs('progan_checkpoints', exist_ok=True)
                    os.makedirs('progan_output', exist_ok=True)
                    checkpoint_path = f'progan_checkpoints/conditional_progan_res{current_res}_epoch{epoch}.pth'
                    torch.save({'gen_state_dict': self.gen.state_dict(), 'dis_state_dict': self.dis.state_dict(), 'gen_shadow_state_dict': self.gen_shadow.state_dict() if self.use_ema else None, 'depth': current_depth, 'alpha': alpha}, checkpoint_path)
                    print(f'Saved checkpoint: {checkpoint_path}')
                    imgs = self.generate(current_depth, alpha=alpha, noise=fixed_noise, races=fixed_races, n=batch_sizes[current_depth], n_plot=0)
                    gen_imgs = imgs[1]
                    vutils.save_image(gen_imgs * 0.5 + 0.5, f'progan_output/conditional_res{current_res}_epoch{epoch}.png', nrow=8)
                if time.time() - KERNEL_START_TIME > 32000:
                    print('Time limit reached, interrupting training.')
                    break
        self.gen.eval()
        self.dis.eval()
        if self.use_ema:
            self.gen_shadow.eval()
        return infos
import argparse
import glob
from PIL import Image

class Rescale:

    def __init__(self):
        self.a = 2
        self.b = -1

    def __call__(self, tensor):
        return tensor.mul(self.a).add(self.b)

    def __repr__(self):
        return self.__class__.__name__ + '(x{}, +{})'.format(self.a, self.b)

def get_transforms(size):
    base_transforms = transforms.Compose([transforms.Resize(size)])
    additional_transforms = transforms.Compose([transforms.RandomHorizontalFlip(p=0.5), transforms.RandomChoice([transforms.CenterCrop(size), transforms.RandomCrop(size)]), transforms.RandomApply([transforms.ColorJitter(brightness=0.2, contrast=(0.9, 1.2), saturation=0.3, hue=0.01)], p=0.5), transforms.ToTensor(), Rescale()])
    return (base_transforms, additional_transforms)

class DogeDataset(Dataset):

    def __init__(self, folder, base_transforms, additional_transforms):
        self.folder = folder
        self.classes = [dirname[10:] for dirname in os.listdir(ANNOTATION_PATH)]
        self.base_transforms = base_transforms
        self.additional_transforms = additional_transforms
        self.imgs, self.labels = self.load_subfolders_images(folder)
        le = LabelEncoder().fit(self.classes)
        self.y = torch.from_numpy(le.transform(self.labels)).long()
        self.classes = le.inverse_transform(range(len(self.classes)))

    def __getitem__(self, index):
        return (self.additional_transforms(self.imgs[index]), self.y[index])

    def __len__(self):
        return len(self.imgs)

    @staticmethod
    def is_valid_file(x):
        img_extensions = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')
        return has_file_allowed_extension(x, img_extensions)

    @staticmethod
    def get_bbox(o):
        bndbox = o.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)
        return (xmin, ymin, xmax, ymax)

    @staticmethod
    def larger_bbox(bbox, ximg, yimg, a=10):
        xmin, ymin, xmax, ymax = bbox
        xmin = max(xmin - a, 0)
        ymin = max(ymin - a, 0)
        xmax = min(xmax + a, ximg)
        ymax = min(ymax + a, yimg)
        return (xmin, ymin, xmax, ymax)

    def load_subfolders_images(self, root):
        imgs = []
        paths = []
        labels = []
        for root, _, fnames in sorted(os.walk(root)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                if self.is_valid_file(path):
                    paths.append(path)
        for path in paths:
            img = default_loader(path)
            annotation_basename = os.path.splitext(os.path.basename(path))[0]
            annotation_dirname = next((dirname for dirname in os.listdir(ANNOTATION_PATH) if dirname.startswith(annotation_basename.split('_')[0])))
            annotation_filename = os.path.join(ANNOTATION_PATH, annotation_dirname, annotation_basename)
            label = annotation_dirname[10:]
            tree = ET.parse(annotation_filename)
            root = tree.getroot()
            objects = root.findall('object')
            for o in objects:
                bbox = self.get_bbox(o)
                bbox = self.larger_bbox(bbox, img.size[0], img.size[1])
                object_img = self.base_transforms(img.crop(bbox))
                imgs.append(object_img)
                labels.append(label)
        return (imgs, labels)
import zipfile

DATA_PATH = '/kaggle/working/all-dogs/all-dogs'
ANNOTATION_PATH = '/kaggle/working/Annotation/Annotation'

if __name__ == '__main__':
    KERNEL_START_TIME = time.time()
    
    # Auto-extract Kaggle Zips
    if not os.path.exists('/kaggle/working/all-dogs'):
        print("Extracting dogs...")
        with zipfile.ZipFile('/kaggle/input/generative-dog-images/all-dogs.zip', 'r') as z:
            z.extractall('/kaggle/working/all-dogs')
    if not os.path.exists('/kaggle/working/Annotation'):
        print("Extracting annotations...")
        with zipfile.ZipFile('/kaggle/input/generative-dog-images/Annotation.zip', 'r') as z:
            z.extractall('/kaggle/working/Annotation')
    seed_everything(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    base_transforms, additional_transforms = get_transforms(64)
    dataset = DogeDataset(DATA_PATH, base_transforms, additional_transforms)
    
    nb_classes = len(dataset.classes)
    print(f'Number of dogs : {len(dataset)}')
    print(f'Number of classes : {nb_classes}')
    depth = 5
    latent_size = 256
    loss = Hinge
    lr_d = 0.006
    lr_g = 0.006
    pro_gan = ConditionalProGAN(num_classes=nb_classes, depth=depth, latent_size=latent_size, loss=loss, lr_d=lr_d, lr_g=lr_g, use_ema=True, use_eql=True, use_spec_norm=False)
    num_epochs = [5, 10, 20, 40, 100]
    fade_ins = [50, 20, 20, 10, 5]
    batch_sizes = [64] * 5
    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        print(f"Let's use {num_gpus} GPUs!")
        batch_sizes = [b * num_gpus for b in batch_sizes]
    ema_decays = [0.9, 0.9, 0.99, 0.99, 0.99]
    if len(dataset) > 0:
        print('Starting training!')
        infos = pro_gan.train(dataset=dataset, epochs=num_epochs, fade_in_percentage=fade_ins, batch_sizes=batch_sizes, ema_decays=ema_decays, verbose=1)
        print("Training complete! Checkpoints are saved in 'progan_checkpoints' and samples in 'progan_output'")
    else:
        print('Dataset not found! Please check DATA_PATH.')