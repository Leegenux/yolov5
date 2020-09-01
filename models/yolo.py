import argparse
import math
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.common import Conv, Bottleneck, SPP, DWConv, Focus, BottleneckCSP, Concat
from models.experimental import MixConv2d, CrossConv, C3
from utils.general import check_anchor_order, make_divisible, check_file
from utils.torch_utils import (
    time_synchronized, fuse_conv_and_bn, model_info, scale_img, initialize_weights, select_device, IOULoss)


class FCOSDetect(nn.Module):
    def __init__(self, nc=80, in_feat_channels=[], shared_params=False):
        super(FCOSDetect, self).__init__()

        self.num_classes = nc  # number of classes
        self.num_levels = len(in_feat_channels)  # check if necessary
        self._shared_params = shared_params

        # set up the detection heads
        assert self._shared_params == False
        if self._shared_params == False:
            self.cls_logits = nn.ModuleList()
            self.bbox_pred = nn.ModuleList()
            self.ctrness = nn.ModuleList()
            for ch_in in in_feat_channels:
                self.cls_logits.append(nn.Conv2d(
                    ch_in, self.num_classes,
                    kernel_size=3, stride=1,
                    padding=1
                ))
                self.bbox_pred.append(nn.Conv2d(
                    ch_in, 4, kernel_size=3,
                    stride=1, padding=1
                ))
                self.ctrness.append(nn.Conv2d(
                    ch_in, 1, kernel_size=3,
                    stride=1, padding=1
                ))
        else:
            # make sure that ch_ins are same
            assert len(set(in_feat_channels)
                       ) == 1, "Please make sure that in features have the same number of channels!"
            self.cls_logits = nn.Conv2d(
                in_feat_channels[0], self.num_classes,
                kernel_size=3, stride=1,
                padding=1
            )
            self.bbox_pred = nn.Conv2d(
                in_feat_channels[0], 4, kernel_size=3,
                stride=1, padding=1
            )
            self.ctrness = nn.Conv2d(
                in_feat_channels[0], 1, kernel_size=3,
                stride=1, padding=1
            )

    def forward(self, x):  # input is a list of features from different levels
        logits = []
        bbox_reg = []
        ctrness = []

        # TODO implement Scale for bbox_reg
        if self._shared_params == False:
            for l, feature in enumerate(x):
                logits.append(self.cls_logits[l](feature).float())
                bbox_reg.append(F.relu(self.bbox_pred[l](feature)).float())
                ctrness.append(self.ctrness[l](feature).float())

        else:
            for feature in x:
                logits.append(self.cls_logits(feature).float())
                bbox_reg.append(F.relu(self.bbox_pred(feature)).float())
                ctrness.append(self.ctrness(feature).float())

        return logits, bbox_reg, ctrness

    def is_shared_params(self):
        return self._shared_params


class Detect(nn.Module):
    def __init__(self, nc=80, anchors=(), ch=()):  # detection layer
        super(Detect, self).__init__()
        self.stride = None  # strides computed during build
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer('anchors', a)  # shape(nl,na,2)
        self.register_buffer('anchor_grid', a.clone().view(
            self.nl, 1, -1, 1, 1, 2))  # shape(nl,1,na,1,1,2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1)
                               for x in ch)  # output conv
        self.export = False  # onnx export

    def forward(self, x):
        # x = x.copy()  # for profiling
        z = []  # inference output
        self.training |= self.export
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(
                0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)

                y = x[i].sigmoid()
                y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 +
                               self.grid[i].to(x[i].device)) * self.stride[i]  # xy
                y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * \
                              self.anchor_grid[i]  # wh
                z.append(y.view(bs, -1, self.no))

        return x if self.training else (torch.cat(z, 1), x)

    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()


class Model(nn.Module):
    # model, input channels, number of classes
    def __init__(self, cfg='yolov5s.yaml', ch=3, nc=None):
        super(Model, self).__init__()
        if isinstance(cfg, dict):
            self.yaml = cfg  # model dict
        else:  # is *.yaml
            import yaml  # for torch hub
            self.yaml_file = Path(cfg).name
            with open(cfg) as f:
                self.yaml = yaml.load(f, Loader=yaml.FullLoader)  # model dict

        # Define model
        if nc and nc != self.yaml['nc']:
            print('Overriding %s nc=%g with nc=%g' %
                  (cfg, self.yaml['nc'], nc))
            self.yaml['nc'] = nc  # override yaml value
        self.nc = self.yaml['nc']
        self.model, self.save = parse_model(
            deepcopy(self.yaml), ch=[ch])  # model, savelist, ch_out
        # print([x.shape for x in self.forward(torch.zeros(1, ch, 64, 64))])

        # Build strides, anchors
        m = self.model[-1]
        self.is_fcos = isinstance(m, FCOSDetect)
        ## forward with a fake image and get the strides
        s = 128  # 2x min stride
        if self.is_fcos:
            m.stride = torch.tensor([s / x.shape[-1] for x in self.forward(torch.zeros(1, ch, s, s))[0]])
            # sizes of interest
            INF = 100000000
            soi = []
            prev_size = -1
            for s in self.yaml['soi']:
                soi.append([prev_size, s])
                prev_size = s
            soi.append([prev_size, INF])
            self.sizes_of_interest = soi
            # center sample setting
            self.center_sample = (self.yaml['center_sample'] == 1)
            # focal loss hyper parameters
            self.focal_loss_alpha = self.yaml['focal_loss_alpha']
            self.focal_loss_gamma = self.yaml['focal_loss_gamma']
            # IOULoss
            self.loc_loss_func = IOULoss(self.yaml['iou'])
        else:
            m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s))])  # forward
        self.stride = m.stride
        ## anchors
        if not self.is_fcos:  # non-FCOS Detect requires anchors
            m.anchors /= m.stride.view(-1, 1, 1)
            check_anchor_order(m)
            self._initialize_biases()  # for non-fcos only
            # print('Strides: %s' % m.stride.tolist())

        # Init weights, biases
        initialize_weights(self)
        self.info()
        print('')

    def forward(self, x, augment=False, profile=False):
        if augment:
            img_size = x.shape[-2:]  # height, width
            s = [1, 0.83, 0.67]  # scales
            f = [None, 3, None]  # flips (2-ud, 3-lr)
            y = []  # outputs
            for si, fi in zip(s, f):
                xi = scale_img(x.flip(fi) if fi else x, si)
                yi = self.forward_once(xi)[0]  # forward
                # cv2.imwrite('img%g.jpg' % s, 255 * xi[0].numpy().transpose((1, 2, 0))[:, :, ::-1])  # save
                yi[..., :4] /= si  # de-scale
                if fi == 2:
                    yi[..., 1] = img_size[0] - yi[..., 1]  # de-flip ud
                elif fi == 3:
                    yi[..., 0] = img_size[1] - yi[..., 0]  # de-flip lr
                y.append(yi)
            return torch.cat(y, 1), None  # augmented inference, train
        else:
            # single-scale inference, train
            return self.forward_once(x, profile)

    def forward_once(self, x, profile=False):
        y, dt = [], []  # outputs
        for m in self.model:
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [
                    x if j == -1 else y[j] for j in m.f]  # from earlier layers

            if profile:
                try:
                    import thop
                    o = thop.profile(m, inputs=(x,), verbose=False)[
                            0] / 1E9 * 2  # FLOPS
                except:
                    o = 0
                t = time_synchronized()
                for _ in range(10):
                    _ = m(x)
                dt.append((time_synchronized() - t) * 100)
                print('%10.1f%10.0f%10.1fms %-40s' % (o, m.np, dt[-1], m.type))

            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output

        if profile:
            print('%.1fms total' % sum(dt))
        return x

    # initialize biases into Detect(), cf is class frequency
    def _initialize_biases(self, cf=None):
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        m = self.model[-1]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            # obj (8 objects per 640 image)
            b[:, 4] += math.log(8 / (640 / s) ** 2)
            b[:, 5:] += math.log(0.6 / (m.nc - 0.99)
                                 ) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def _print_biases(self):
        m = self.model[-1]  # Detect() module
        for mi in m.m:  # from
            b = mi.bias.detach().view(m.na, -1).T  # conv.bias(255) to (3,85)
            print(('%6g Conv2d.bias:' + '%10.3g' * 6) %
                  (mi.weight.shape[1], *b[:5].mean(1).tolist(), b[5:].mean()))

    # def _print_weights(self):
    #     for m in self.model.modules():
    #         if type(m) is Bottleneck:
    #             print('%10.3g' % (m.w.detach().sigmoid() * 2))  # shortcut weights

    def fuse(self):  # fuse model Conv2d() + BatchNorm2d() layers
        print('Fusing layers... ', end='')
        for m in self.model.modules():
            if type(m) is Conv:
                m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatability
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                m.bn = None  # remove batchnorm
                m.forward = m.fuseforward  # update forward
        self.info()
        return self

    def info(self):  # print model information
        model_info(self)

    def compute_locations(self, im_width, im_height, device):
        locations = []
        for s in self.stride:
            shift_x = torch.arange(0, im_width, step=s, dtype=torch.float32, device=device)
            shift_y = torch.arange(0, im_height, step=s, dtype=torch.float32, device=device)
            shift_y, shift_x = torch.meshgrid(shift_y, shift_x)
            shift_x = shift_x.reshape(-1)
            shift_y = shift_y.reshape(-1)
            locations_per_level = torch.stack((shift_x, shift_y), dim=1) + s // 2
            locations.append(locations_per_level)
        return locations

    def get_sample_region(self, boxes, num_loc_list, loc_xs, loc_ys, bitmasks=None):
        strides = self.stride
        radius = self.yaml.get('radius', 1)  # defaults to 1

        if bitmasks is not None:
            _, h, w = bitmasks.size()

            ys = torch.arange(0, h, dtype=torch.float32, device=bitmasks.device)
            xs = torch.arange(0, w, dtype=torch.float32, device=bitmasks.device)

            m00 = bitmasks.sum(dim=-1).sum(dim=-1).clamp(min=1e-6)
            m10 = (bitmasks * xs).sum(dim=-1).sum(dim=-1)
            m01 = (bitmasks * ys[:, None]).sum(dim=-1).sum(dim=-1)
            center_x = m10 / m00
            center_y = m01 / m00
        else:
            center_x = boxes[..., [0, 2]].sum(dim=-1) * 0.5
            center_y = boxes[..., [1, 3]].sum(dim=-1) * 0.5

        num_gts = boxes.shape[0]
        K = len(loc_xs)
        boxes = boxes[None].expand(K, num_gts, 4)  # expand to fit each location
        center_x = center_x[None].expand(K, num_gts)  # 针对每个box 获取中心的坐标
        center_y = center_y[None].expand(K, num_gts)
        center_gt = boxes.new_zeros(boxes.shape)  # b
        # no gt
        if center_x.numel() == 0 or center_x[..., 0].sum() == 0:
            return loc_xs.new_zeros(loc_xs.shape, dtype=torch.uint8)
        beg = 0
        for level, num_loc in enumerate(num_loc_list):  # 根据radius，限制每个box的大小
            end = beg + num_loc
            stride = strides[level] * radius
            xmin = center_x[beg:end] - stride  # 针对每个像素，x的范围（此为最小值）
            ymin = center_y[beg:end] - stride
            xmax = center_x[beg:end] + stride
            ymax = center_y[beg:end] + stride
            # limit sample region in gt
            center_gt[beg:end, :, 0] = torch.where(xmin > boxes[beg:end, :, 0], xmin, boxes[beg:end, :, 0])
            center_gt[beg:end, :, 1] = torch.where(ymin > boxes[beg:end, :, 1], ymin, boxes[beg:end, :, 1])
            center_gt[beg:end, :, 2] = torch.where(xmax > boxes[beg:end, :, 2], boxes[beg:end, :, 2], xmax)
            center_gt[beg:end, :, 3] = torch.where(ymax > boxes[beg:end, :, 3], boxes[beg:end, :, 3], ymax)
            beg = end
        left = loc_xs[:, None] - center_gt[..., 0]  # 作比较，其中较大的在bbox里面
        right = center_gt[..., 2] - loc_xs[:, None]
        top = loc_ys[:, None] - center_gt[..., 1]
        bottom = center_gt[..., 3] - loc_ys[:, None]
        center_bbox = torch.stack((left, top, right, bottom), -1)
        inside_gt_bbox_mask = center_bbox.min(-1)[0] > 0
        return inside_gt_bbox_mask

    def _transpose(self, training_targets, num_loc_list):
        '''
        This function is used to transpose image first training targets to level first ones
        :return: level first training targets
        '''
        for im_i in range(len(training_targets)):
            training_targets[im_i] = torch.split(
                training_targets[im_i], num_loc_list, dim=0
            )

        targets_level_first = []
        for targets_per_level in zip(*training_targets):
            targets_level_first.append(
                torch.cat(targets_per_level, dim=0)
            )
        return targets_level_first


def parse_model(d, ch):  # model_dict, input_channels(3)
    print('\n%3s%18s%3s%10s  %-40s%-30s' %
          ('', 'from', 'n', 'params', 'module', 'arguments'))
    # no `anchors`, no `no`
    nc, gd, gw = d['nc'], d['depth_multiple'], d['width_multiple']
    no = None  # set a default value
    if d.get('anchors'):
        anchors = d['anchors']
        na = (len(anchors[0]) // 2) if isinstance(anchors,
                                                  list) else anchors  # number of anchors
        no = na * (nc + 5)  # number of outputs = anchors * (classes + 5)

    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    # from, number, module, args
    for i, (f, n, m, args) in enumerate(d['backbone'] + (d.get('fcos_head') or d.get('head'))):
        m = eval(m) if isinstance(m, str) else m  # eval strings
        for j, a in enumerate(args):
            try:
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings
            except:
                pass

        n = max(round(n * gd), 1) if n > 1 else n  # depth gain
        if m in [nn.Conv2d, Conv, Bottleneck, SPP, DWConv, MixConv2d, Focus, CrossConv, BottleneckCSP, C3]:
            c1, c2 = ch[f], args[0]

            # Normal
            # if i > 0 and args[0] != no:  # channel expansion factor
            #     ex = 1.75  # exponential (default 2.0)
            #     e = math.log(c2 / ch[1]) / math.log(2)
            #     c2 = int(ch[1] * ex ** e)
            # if m != Focus:

            c2 = make_divisible(c2 * gw, 8) if c2 != no else c2

            # Experimental
            # if i > 0 and args[0] != no:  # channel expansion factor
            #     ex = 1 + gw  # exponential (default 2.0)
            #     ch1 = 32  # ch[1]
            #     e = math.log(c2 / ch1) / math.log(2)  # level 1-n
            #     c2 = int(ch1 * ex ** e)
            # if m != Focus:
            #     c2 = make_divisible(c2, 8) if c2 != no else c2

            args = [c1, c2, *args[1:]]
            if m in [BottleneckCSP, C3]:
                args.insert(2, n)
                n = 1
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum([ch[-1 if x == -1 else x + 1] for x in f])
        elif m is FCOSDetect:
            # set in_feat_channels for FCOSHead
            args.append([ch[x + 1] for x in f])
        elif m is Detect or m is FCOSDetect:
            args.append([ch[x + 1] for x in f])
            if isinstance(args[1], int):  # set the actual number of anchors
                args[1] = [list(range(args[1] * 2))] * len(f)
        else:
            c2 = ch[f]

        m_ = nn.Sequential(*[m(*args) for _ in range(n)]
                           ) if n > 1 else m(*args)  # module
        t = str(m)[8:-2].replace('__main__.', '')  # module type
        np = sum([x.numel() for x in m_.parameters()])  # number params
        # attach index, 'from' index, type, number params
        m_.i, m_.f, m_.type, m_.np = i, f, t, np
        print('%3s%18s%3s%10.0f  %-40s%-30s' % (i, f, n, np, t, args))  # print
        save.extend(x % i for x in ([f] if isinstance(
            f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str,
                        default='yolov5s.yaml', help='model.yaml')
    parser.add_argument('--device', default='',
                        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    opt = parser.parse_args()
    opt.cfg = check_file(opt.cfg)  # check file
    device = select_device(opt.device)

    # Create model
    model = Model(opt.cfg).to(device)
    model.train()

    # Profile
    # img = torch.rand(8 if torch.cuda.is_available() else 1, 3, 640, 640).to(device)
    # y = model(img, profile=True)

    # ONNX export
    # model.model[-1].export = True
    # torch.onnx.export(model, img, opt.cfg.replace('.yaml', '.onnx'), verbose=True, opset_version=11)

    # Tensorboard
    # from torch.utils.tensorboard import SummaryWriter
    # tb_writer = SummaryWriter()
    # print("Run 'tensorboard --logdir=models/runs' to view tensorboard at http://localhost:6006/")
    # tb_writer.add_graph(model.model, img)  # add model to tensorboard
    # tb_writer.add_image('test', img[0], dataformats='CWH')  # add model to tensorboard
