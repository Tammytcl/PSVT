import einops

from mmseg.models.backbones.swin_transformer_v2 import SwinTransformerV2, BasicLayer
# from mmseg.models.backbones.my_svit_v2 import SwinTransformerV2,BasicLayer
import torch.nn as nn
import torch
import numpy as np
from collections import OrderedDict
from einops import rearrange
import copy
from mmseg.registry import MODELS


def to_img_shape(x):
    B, L, C = x.shape
    H = int(np.sqrt(L))
    W = H
    return x.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()


def to_swin_shape(x):
    B, C, H, W = x.shape
    return x.view(B, C, H * W).permute(0, 2, 1).contiguous()


def change_shape(x):
    # B L C -> B 4*L C//2
    B, L, C = x.shape
    H = int(np.sqrt(L))
    W = H
    x = x.view(B, H, W, C)
    # batch*3136*96 -> batch*224*224*96
    x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c',
                  p1=2, p2=2,
                  c=C // 4)
    x = x.view(B, -1, C // 4)
    return x


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1,act=nn.GELU,groups=1,bias=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size, padding=padding,bias=bias,groups=groups)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = act()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Head4x(nn.Module):
    def __init__(self, dim):
        super().__init__()
        dim//=3
        self.deconv1=nn.ConvTranspose2d(in_channels=3*dim,out_channels=dim,kernel_size=4,stride=4)
        self.norm=nn.BatchNorm2d(dim)
        self.act=nn.GELU()
        #self.conv1=nn.Conv2d(in_channels=dim,out_channels=dim,kernel_size=1)
        
        # self.deconv2=nn.ConvTranspose2d(in_channels=2*dim,out_channels=dim,kernel_size=4,stride=4)

        # self.up=nn.Upsample(mode='bilinear',scale_factor=2,align_corners=True)
        # self.conv1=ConvBlock(in_channels=3*dim,out_channels=2*dim)
        # self.conv2=ConvBlock(in_channels=2*dim,out_channels=dim)

    def forward(self, x):
        # print(1,x.shape)
        x=self.deconv1(x)
        x=self.norm(x)
        x=self.act(x)
        # print(2,x.shape)
        # x = self.up(x)
        # x=self.conv1(x)
        # x=self.up(x)
        # x=self.conv2(x)
        return x


class PatchExpand(nn.Module):
    def __init__(self, dim, dim_scale=2,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.expand = nn.Linear(dim, 2 * dim)
        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x):
        """
        x: B, L, C
        """

        x = self.expand(x)
        B, L, C = x.shape
        H = int(np.sqrt(L))
        W = H
        x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=2, p2=2,
                      c=C // 4)
        # print(x.shape)
        x = x.view(B, -1, C // 4)
        x = self.norm(x)
        return x


class FinalPatchExpand_X4(nn.Module):
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.expand1 = nn.Linear(dim, 4 * dim, bias=False)
        self.norm1 = norm_layer(dim)
        self.expand2 = nn.Linear(dim, 4 * dim, bias=False)
        self.norm2 = norm_layer(dim)

    def forward(self, x):
        """
        x: B, L, C
        """
        x = self.expand1(x)
        x = change_shape(x)
        x = self.norm1(x)
        x = self.expand2(x)
        x = change_shape(x)
        x = self.norm2(x)

        return x


class PatchUp(nn.Module):
    def __init__(self, dim):
        super(PatchUp, self).__init__()
        # dim==768
        self.dim = dim
        # seq1

        self.up1=nn.Upsample(scale_factor=2,align_corners=True)
        self.conv1=ConvBlock(in_channels=dim,out_channels=dim//2)
        self.conv2 = ConvBlock(in_channels=dim, out_channels=dim // 2)
        # self.up1 = nn.ConvTranspose2d(in_channels=dim, out_channels=dim // 2,
        #                               kernel_size=4, stride=2, padding=1)
        # self.conv2 = ConvBlock(in_channels=dim, out_channels=dim // 2,
        #                        kernel_size=1, padding=0)

        # seq2
        self.linear1 = nn.Linear(in_features=dim, out_features=dim * 2)
        self.norm1 = nn.LayerNorm(dim // 2)
        self.linear2 = nn.Linear(in_features=dim, out_features=dim // 2)
        self.norm2 = nn.LayerNorm(dim // 2)

        self.norm = nn.LayerNorm(dim // 2)

    def forward(self, x, encoder_output):
        # x1 = x
        # x1 = to_img_shape(x1)
        # x1 = self.up1(x1)
        # x1=self.conv1(x1)
        # temp_encoder = to_img_shape(encoder_output)
        # x1 = torch.cat([temp_encoder, x1], dim=1)
        # x1 = self.conv2(x1)
        # x1 = to_swin_shape(x1)

        x2 = x
        x2 = self.linear1(x2)
        x2 = change_shape(x2)
        x2 = torch.cat([encoder_output, x2], dim=-1)
        x2 = self.linear2(x2)
        x2 = self.norm2(x2)

        return self.norm(x2)
class UpBlock(nn.Module):
    def __init__(self,dim,mode='bilinear',scale_factor=2):
        super(UpBlock, self).__init__()
        #self.up=nn.Upsample(scale_factor=scale_factor,mode=mode,align_corners=True)
        #self.conv=ConvBlock(in_channels=dim,out_channels=dim//2)
        #self.act=nn.ReLU()
        self.deconv1 = nn.ConvTranspose2d(in_channels=dim, out_channels=dim//2,
                                          kernel_size=2, stride=2)
        self.norm = nn.BatchNorm2d(dim//2)
        self.act = nn.GELU()
    def forward(self,input):
        #output=self.up(input)
        #output=self.conv(output)
        #output=self.act(output)
        x = self.deconv1(input)
        x = self.norm(x)
        output= self.act(x)
        return output
class MultiDecoder(nn.Module):
    def __init__(self, max_dim, min_dim):
        super(MultiDecoder, self).__init__()
        self.depths = int(np.log2(max_dim / min_dim))
        self.decoders = nn.ModuleList([nn.Sequential() for i in range(self.depths)])

        temp_dim = max_dim
        for i in range(self.depths):
            for j in range(self.depths):
                if j > i:
                    break
                self.decoders[j].add_module(str(i), UpBlock(dim=temp_dim))
            temp_dim //= 2

    def forward(self, inputs):
        outputs = []
        for i in range(self.depths):
            output = self.decoders[i](inputs[i])
            outputs.append(output)
        outputs.append(inputs[-1])

        output = torch.cat(outputs, dim=1)
        return output
class BasicLayerUp(nn.Module):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None,
                 use_checkpoint=False,
                 pretrained_window_size=0):
        super().__init__()

        # conv
        self.up=UpBlock(dim=2*dim)
        self.conv=ConvBlock(in_channels=2*dim,out_channels=dim)


        # self.up = PatchUp(dim=dim * 2)

        self.layers = BasicLayer(
            dim=dim,
            input_resolution=input_resolution,
            depth=depth,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop=drop,
            attn_drop=attn_drop,
            drop_path=drop_path,
            norm_layer=norm_layer,
            downsample=None,
            use_checkpoint=False,
            pretrained_window_size=0)

    def forward(self, input, encoder_output):
        input=to_img_shape(input)
        encoder_output=to_img_shape(encoder_output)

        output=self.up(input)
        output=torch.cat([encoder_output,output],dim=1)
        output=self.conv(output)
        output = to_swin_shape(output)

        # output = self.up(input, encoder_output)

        output_shortcut, output = self.layers(output)

        return output


@MODELS.register_module(force=True)
class PSVT(nn.Module):
    def __init__(self,
                 img_size=224,
                 patch_size=4,
                 in_chans=3,
                 num_classes=1000,
                 embed_dim=96,
                 depths=[2, 2, 6, 2],
                 num_heads=[3, 6, 12, 24],
                 window_size=[7,7,7,7],
                 mlp_ratio=4.,
                 qkv_bias=True,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm,
                 ape=False,
                 patch_norm=True,
                 use_checkpoint=False,
                 pretrained_window_sizes=[0, 0, 0, 0],
                 pretrain_path=None,
                 **kwargs):
        super().__init__()
        self.encoders = SwinTransformerV2(img_size=img_size,
                                          window_size=window_size,
                                          drop_path_rate=drop_path_rate,
                                          ape=ape)

        temp_size = img_size // patch_size
        temp_dim = embed_dim

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate,
                                                sum(depths))]  # stochastic depth decay rule

        self.decoders = nn.ModuleList()
        for i in range(len(depths) - 1):
            layer_up = BasicLayerUp(
                dim=int(temp_dim),
                input_resolution=(int(temp_size), int(temp_size)),
                depth=depths[i],
                num_heads=num_heads[i],
                window_size=window_size[i],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                norm_layer=norm_layer,
                downsample=None,
                use_checkpoint=False,
                pretrained_window_size=0
            )
            self.decoders.append(layer_up)

            temp_size /= 2
            temp_dim *= 2
        # multi scale
        self.multi_decoder = MultiDecoder(max_dim=temp_dim//2, min_dim=embed_dim)
        multi_channels=(len(depths)-1)*embed_dim
        self.dwconv=ConvBlock(in_channels=multi_channels,out_channels=multi_channels,groups=multi_channels)

        self.head4x=Head4x(dim=(len(depths)-1)*embed_dim)

        if pretrain_path is not None:
            self.load_pretrain_weight(pretrain_path)

    def forward(self, inputs):
        encoder_outputs = self.encoders(inputs)
        outputs = []
        # outputs.append(to_img_shape(encoder_outputs[-1]))
        output = encoder_outputs.pop(-1)

        # range=2,1,0
        for i in range(len(encoder_outputs) - 1, -1, -1):
            # print(encoder_outputs[i].shape)
            output = self.decoders[i](output, encoder_outputs[i])
            outputs.append(to_img_shape(output))


        output=self.multi_decoder(outputs)
        # multi scale
        output=self.dwconv(output)

        # print(outputs[-1].shape)
        output=self.head4x(output)

        return [output]

    def load_pretrain_weight(self, pretrain_path):
        encoders_weight = torch.load(pretrain_path)['model']
        decoders_weight = OrderedDict()

        for k, v in encoders_weight.items():
            # print(k)
            if 'layers' in k:
                ks = k.split('.')
                ks[0], ks[1] = ks[1], ks[0]
                new_k = '.'.join(ks)
                # print(k,new_k)
                decoders_weight[new_k] = v
        self.encoders.load_state_dict(encoders_weight, strict=False)
        msg = self.decoders.load_state_dict(decoders_weight, strict=False)
        print(msg)


if __name__ == '__main__':
    import time
    from fvcore.nn import FlopCountAnalysis
    from fvcore.nn import flop_count_table
    import numpy as np

    seed = 4217
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    img_size = 2

    # To evaluate GFLOPs, pleaset set `rms_norm=False` and `fused_add_norm=False`
    model = PSVT(img_size=256,window_size=[8,8,8,8],depths=[2, 2, 6, 2]).cuda()
    x=torch.rand((1, 3, 256, 256)).cuda()
    # model=nn.BatchNorm2d(3).cuda()
    # x=torch.from_numpy(np.array([[[[1,2],
    #                       [3,4]],
    #                      [[5,6],
    #                       [7,8]],
    #                      [[9,10],
    #                       [11,12]]],
    #                      [[[1,2],
    #                       [3,4]],
    #                      [[5,6],
    #                       [7,8]],
    #                      [[9,10],
    #                       [11,12]]]])).float().cuda()
    # print(model(x).view(2, 3, img_size * img_size).permute(0, 2, 1).contiguous())
    # model=nn.LayerNorm(3).cuda()
    # x=x.view(2, 3, img_size * img_size).permute(0, 2, 1).contiguous()
    # print(x)
    # print(model(x))
    flops = FlopCountAnalysis(model, x.cuda())
    s = time.time()
    print(flop_count_table(flops,max_depth=1))
    print(time.time()-s)
    # x = torch.randn((4, 3, 256, 256))
    # net = MySwinUnetV2(img_size=256,window_size=[8,8,8,8],depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24])
    # # print(net)
    # outputs = net(x)
    # for output in outputs:
    #     print(output.shape)
