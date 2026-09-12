from diffusers import ModelMixin, ConfigMixin
from diffusers.configuration_utils import register_to_config
from diffusers.utils.torch_utils import randn_tensor
from torch import nn
from torch.nn import functional as F
import torch


class VAE(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(
        self,
        base_dim = 96,
        z_dim = 16,
        dim_mult = [1, 2, 4, 4],
        latents_mean = [],
        latents_std = [],
    ):
        super().__init__()

        self.encoder = Encoder(base_dim, z_dim * 2, dim_mult)
        self.decoder = Decoder(base_dim, z_dim, dim_mult)

        self.quant_conv = CausalConv3d(z_dim * 2, z_dim * 2, kernel_size=1)
        self.post_quant_conv = CausalConv3d(z_dim, z_dim, kernel_size=1)

    def encode(self, x):
        _, _, num_frame, height, width = x.shape
        
        self.clear_cache()

        iter_ = 1 + (num_frame - 1) // 4
        for i in range(iter_):
            self._enc_conv_idx = [0]
            if i == 0:
                out = self.encoder(x[:, :, :1, :, :],self._enc_feat_map,self._enc_conv_idx)
            else:
                out_ = self.encoder(x[:, :, 1 + 4 * (i - 1) : 1 + 4 * i, :, :],self._enc_feat_map,self._enc_conv_idx,)
                out = torch.cat([out, out_], 2)

        enc = self.quant_conv(out)
        self.clear_cache()

        posterior = DiagonalGaussianDistribution(enc)

        return posterior

    def decode(self,x):
        _, _, num_frame, height, width = x.shape

        self.clear_cache()

        x = self.post_quant_conv(x)
        for i in range(num_frame):
            self._conv_idx = [0]
            if i == 0:
                out = self.decoder(x[:, :, i : i + 1, :, :], self._feat_map, self._conv_idx)
            else:
                out_ = self.decoder(x[:, :, i : i + 1, :, :], self._feat_map, self._conv_idx)
                out = torch.cat([out, out_], 2)
        
        out = torch.clamp(out, min=-1.0, max=1.0)

        self.clear_cache()

        return VAEDecoderOutput(out)


    def clear_cache(self):
        def _count_conv3d(model):
            count = 0
            for m in model.modules():
                if isinstance(m, CausalConv3d):
                    count += 1
            return count

        self._conv_num = _count_conv3d(self.decoder)
        self._conv_idx = [0]
        self._feat_map = [None] * self._conv_num

        self._enc_conv_num = _count_conv3d(self.encoder)
        self._enc_conv_idx = [0]
        self._enc_feat_map = [None] * self._enc_conv_num

    def forward(self, x):
        x = self.encode(x)
        z = x.sample()
        x = self.decode(x)
        return x


class VAEDecoderOutput():
    def __init__(self,sample):
        self.sample = sample


class Encoder(nn.Module):
    def __init__(self, dim=96, z_dim=16*2, dim_mult=[1, 2, 4, 4]):
        super().__init__()
        
        self.nonlinearity = nn.SiLU()

        # [96, 96, 192, 384, 384]
        dims = [dim * u for u in [1] + dim_mult]

        self.conv_in = CausalConv3d(3, dims[0], 3, padding=1)
        self.down_blocks = nn.ModuleList([
            ResNet(dims[0], dims[1]),
            ResNet(dims[1], dims[1]),
            DownSample2D(dims[1]),

            ResNet(dims[1], dims[2]),
            ResNet(dims[2], dims[2]),
            DownSample3D(dims[2]),

            ResNet(dims[2], dims[3]),
            ResNet(dims[3], dims[3]),
            DownSample3D(dims[3]),

            ResNet(dims[3], dims[4]),
            ResNet(dims[4], dims[4])
        ])

        self.mid_block = MidBlock(dims[-1])

        self.norm_out = RMSNorm(dims[-1])
        self.conv_out = CausalConv3d(dims[-1], z_dim, 3, padding=1)

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        if feat_cache is not None:
            idx = feat_idx[0]

            # i=0：(B,3,1,H,W)
            # i>0：(B,3,2,H,W)
            cache_x = x[:, :, -2:, :, :].clone()

            # i=0：(B,3,1,H,W) None -> (B,96,1,H,W)
            # i>0：(B,3,4,H,W) (B,3,2,H,W) -> (B,96,4,H,W)
            x = self.conv_in(x, feat_cache[idx])
            
            # i=0：(B,3,1,H,W)
            # i>0：(B,3,2,H,W)
            feat_cache[idx] = cache_x

            feat_idx[0] += 1
        else:
            # i=0：(B,3,1,H,W)
            # i>0：(B,3,4,H,W)
            x = self.conv_in(x)

        # i=0
        # (B,96,1,1024,1024) -> (B,96,1,512,512)
        # (B,96,1,512,512) -> (B,192,1,256,256)
        # (B,192,1,256,256) -> (B,384,1,128,128)
        # (B,384,1,128,128) -> (B,384,1,128,128)
        # i>0
        # (B,96,4,1024,1024) -> (B,96,4,512,512)
        # (B,96,4,512,512) -> (B,192,2,256,256)
        # (B,192,2,256,256) -> (B,384,1,128,128)
        # (B,384,1,128,128) -> (B,384,1,128,128)
        for layer in self.down_blocks:
            if feat_cache is not None:
                x = layer(x,feat_cache,feat_idx)
            else:
                x = layer(x)

        # (B,384,1,128,128) -> (B,384,1,128,128)
        x = self.mid_block(x, feat_cache, feat_idx)

        # (B,384,1,128,128)
        x = self.norm_out(x)
        # (B,384,1,128,128)
        x = self.nonlinearity(x)

        if feat_cache is not None:
            idx = feat_idx[0]

            # (B,C,1,H,W)
            cache_x = x[:, :, -1:, :, :].clone()

            # (B,C,1,H,W) (B,C,1,H,W) -> (B,C,2,H,W)
            cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device),cache_x],dim=2)

            # i=0：(B,384,1,128,128) None -> (B,16x2,1,128,128)
            # i>0：(B,384,1,128,128) (B,384,2,128,128) -> (B,16x2,1,128,128)
            x = self.conv_out(x, feat_cache[idx])

            # (B,384,2,128,128)
            feat_cache[idx] = cache_x

            feat_idx[0] += 1
        else:
            # (B,384,1,128,128) -> (B,16x2,1,128,128)
            x = self.conv_out(x)

        # (B,16x2,1,128,128)
        return x


class CausalConv3d(nn.Conv3d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride = 1,
        padding = 0,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

        # (left,right,top,bottom,front,back)
        self._padding = (
            self.padding[2], 
            self.padding[2], 
            self.padding[1], 
            self.padding[1], 
            2 * self.padding[0], 
            0
        )
        self.padding = (0, 0, 0)

    def forward(self, x, cache_x=None):
        padding = list(self._padding)

        # 使用上一个编码循环4帧图片的最后2帧图片拼接当前循环4帧图片
        # 如果上一个编码循环不够2帧，就是用上上个循环1帧跟上一个帧的1帧合并成两帧，再跟当前4帧进行拼接。
        # 当前4帧图片，使用上个循环2帧图片填充到6帧，使用3x3x3卷积核，最终6-3+1=4帧
        if cache_x is not None and self._padding[4] > 0:
            cache_x = cache_x.to(x.device)
            x = torch.cat([cache_x, x], dim=2)

            # i=1：只有i=0循环的一帧的缓存，这时候padding[4]-1=1，只填充一个维度
            padding[4] -= cache_x.shape[2]

        x = F.pad(x, padding)
        return super().forward(x)


class ResNet(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
    ):
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim

        self.nonlinearity = nn.SiLU()
        self.dropout = nn.Dropout()

        self.norm1 = RMSNorm(in_dim)
        self.conv1 = CausalConv3d(in_dim, out_dim, 3, padding=1)

        self.norm2 = RMSNorm(out_dim)
        self.conv2 = CausalConv3d(out_dim, out_dim, 3, padding=1)
        
        self.conv_shortcut = CausalConv3d(in_dim, out_dim, 1) if in_dim != out_dim else nn.Identity()

    def forward(self, x, feat_cache, feat_idx):
        # (B,out_dim,T,H,W)
        h = self.conv_shortcut(x)
        
        # (B,in_dim,T,H,W)
        x = self.norm1(x)
        # (B,in_dim,T,H,W)
        x = self.nonlinearity(x)

        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -2:, :, :].clone()

            # 靠后面的下采样和上采样，时间序列维度只有1帧，使用上个循环的1帧和当前1帧拼接成2帧
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2)

            # (B,in_dim,T,H,W) -> (B,out_dim,T,H,W)
            x = self.conv1(x, feat_cache[idx])

            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            # (B,in_dim,T,H,W) -> (B,out_dim,T,H,W)
            x = self.conv1(x)

        # (B,out_dim,T,H,W)
        x = self.norm2(x)
        # (B,out_dim,T,H,W)
        x = self.nonlinearity(x)
        # (B,out_dim,T,H,W)
        x = self.dropout(x)

        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -2:, :, :].clone()

            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2)

            # (B,out_dim,T,H,W) -> (B,out_dim,T,H,W)
            x = self.conv2(x, feat_cache[idx])

            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            # (B,out_dim,T,H,W) -> (B,out_dim,T,H,W)
            x = self.conv2(x)

        # (B,out_dim,T,H,W)
        return x + h


class RMSNorm(nn.Module):
    def __init__(self, dim, vedio=True):
        super().__init__()
        broadcastable_dims = (1, 1, 1) if vedio else (1, 1)

        # (C,1,1,1)
        shape = (dim, *broadcastable_dims) 

        self.scale = dim**0.5

        # (C,1,1,1)
        self.gamma = nn.Parameter(torch.ones(shape))

    def forward(self, x):
        # (B,C,T,H,W) * (C,1,1,1)
        return F.normalize(x, dim=1) * self.scale * self.gamma


class DownSample2D(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.resample = nn.Sequential(
            # left,right,top,bottom
            nn.ZeroPad2d((0, 1, 0, 1)), 
            nn.Conv2d(dim, dim, 3, stride=(2, 2))
        )

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        b, c, t, h, w = x.size()
        
        # (B,C,T,H,W) -> (BxT,C,H,W)
        x = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        # (BxT,C,H,W) -> (BxT,C,H/2,W/2)
        x = self.resample(x)
        # (BxT,C,H/2,W/2) -> (B,C,T,H/2,W/2)
        x = x.view(b, t, c, h, w).permute(0, 2, 1, 3, 4)
        
        return x


class DownSample3D(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.resample = nn.Sequential(
            # left,right,top,bottom
            nn.ZeroPad2d((0, 1, 0, 1)), 
            nn.Conv2d(dim, dim, 3, stride=(2, 2))
        )

        self.time_conv = CausalConv3d(dim, dim, (3, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0))


    def forward(self, x, feat_cache=None, feat_idx=[0]):
        b, c, t, h, w = x.size()
        
        # (B,C,T,H,W) -> (BxT,C,H,W)
        x = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        # (BxT,C,H,W) -> (BxT,C,H/2,W/2)
        x = self.resample(x)
        # (BxT,C,H/2,W/2) -> (B,C,T,H/2,W/2)
        x = x.view(b, t, c, h, w).permute(0, 2, 1, 3, 4)

        if feat_cache is not None:
            idx = feat_idx[0]
            if feat_cache[idx] is None:
                # i=0：(B,C,1,H/2,W/2)
                feat_cache[idx] = x.clone()
                feat_idx[0] += 1
            else:
                # i>0：(B,C,1,H/2,W/2)
                cache_x = x[:, :, -1:, :, :].clone()

                # (B,C,1,H/2,W/2) (B,C,4,H/2,W/2) => (B,C,5,H/2,W/2)
                x = torch.cat([feat_cache[idx][:, :, -1:, :, :], x], dim=2)

                # (B,C,5,H/2,W/2) -> (B,C,2,H/2,W/2)
                # (5-3)/2+1 = 2
                x = self.time_conv(x)

                # (B,C,1,H/2,W/2)
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
        
        return x


class MidBlock(nn.Module):
    def __init__(self,dim):
        super().__init__()

        self.resnets = nn.ModuleList([
            ResNet(dim, dim),
            ResNet(dim, dim),
        ])

        self.attentions = nn.ModuleList([
            Attention(dim),
        ])

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        # (B,C,T,H,W)
        x = self.resnets[0](x, feat_cache, feat_idx)
        # (B,C,T,H,W)
        x = self.attentions[0](x)
        # (B,C,T,H,W)
        x = self.resnets[1](x, feat_cache, feat_idx)

        return x

        
class Attention(nn.Module):
    def __init__(self,dim):
        super().__init__()

        self.norm = RMSNorm(dim, vedio=False)
        self.to_qkv = nn.Conv2d(dim, dim * 3, kernel_size=1)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1)

    def forward(self,x):
        # (B,C,T,H,W)
        residual = x

        batch_size, channels, time, height, width = x.size()

        # (B,C,T,H,W) -> (B,T,C,H,W) -> (BxT,C,H,W)
        x = x.permute(0, 2, 1, 3, 4).reshape(batch_size * time, channels, height, width)
        # (BxT,C,H,W)
        x = self.norm(x)

        # (BxT,C,H,W) -> (BxT,3xC,H,W)
        qkv = self.to_qkv(x)
        # (BxT,3xC,H,W) -> (BxT,1,3xC,HxW)
        qkv = qkv.reshape(batch_size * time, 1, channels * 3, -1)
        # (BxT,1,3xC,HxW) -> (BxT,1,HxW,3xC)
        qkv = qkv.permute(0, 1, 3, 2).contiguous()
        # (BxT,1,HxW,C)(BxT,1,HxW,C) (BxT,1,HxW,C) 
        q, k, v = qkv.chunk(3, dim=-1)

        # (BxT,1,HxW,C)
        x = F.scaled_dot_product_attention(q, k, v)
        # (BxT,1,HxW,C) -> (BxT,HxW,C) -> (BxT,C,HxW) -> (BxT,C,H,W)
        x = x.squeeze(1).permute(0, 2, 1).reshape(batch_size * time, channels, height, width)

        # (BxT,C,H,W) -> (BxT,C,H,W)
        x = self.proj(x)

        # (BxT,C,H,W) -> (B,T,C,H,W)
        x = x.view(batch_size, time, channels, height, width)

        # (B,T,C,H,W) -> (B,C,T,H,W)
        x = x.permute(0, 2, 1, 3, 4)

        # (B,C,T,H,W)
        return x + residual


class Decoder(nn.Module):
    def __init__(
        self,
        dim=96,
        z_dim=16,
        dim_mult=[1, 2, 4, 4],
    ):
        super().__init__()

        self.nonlinearity = nn.SiLU()

        # [384, 384, 384, 192, 96]
        dims = [dim * u for u in [dim_mult[-1]] + dim_mult[::-1]]

        self.conv_in = CausalConv3d(z_dim, dims[0], 3, padding=1)
        self.mid_block = MidBlock(dims[0])

        self.up_blocks = nn.ModuleList([
            UpBlock(dims[0], dims[1], mode="sample3d"),
            UpBlock(dims[1]//2, dims[2], mode="sample3d"),
            UpBlock(dims[2]//2, dims[3], mode="sample2d"),
            UpBlock(dims[3]//2, dims[4]),
        ])

        self.norm_out = RMSNorm(dims[-1])
        self.conv_out = CausalConv3d(dims[-1], 3, 3, padding=1)

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        if feat_cache is not None:
            idx = feat_idx[0]
            # (B,16,1,128,128)
            cache_x = x[:, :, -2:, :, :].clone()

            # i>0：使用上个循环的1帧和当前1帧拼接成2帧
            # (B,16,1,128,128) (B,16,1,128,128) => (B,16,2,128,128)
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device),cache_x],dim=2)

            # (B,16,1,128,128) (B,16,2,128,128) => (B,384,1,128,128)
            x = self.conv_in(x, feat_cache[idx])

            # (B,16,2,128,128)
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            # (B,16,1,128,128) => # (B,384,1,128,128)
            x = self.conv_in(x)

        # (B,384,1,128,128)
        x = self.mid_block(x, feat_cache, feat_idx)

        # (B,384,1,128,128) -> (B,384/2,2,256,256)
        # (B,384/2,2,256,256) -> (B,384/2,4,512,512)
        # (B,384/2,4,512,512) -> (B,192/2,4,1024,1024)
        # (B,192/2,4,1024,1024) -> (B,96,4,1024,1024)
        for up_block in self.up_blocks:
            x = up_block(x, feat_cache, feat_idx)

        # (B,96/2,4,1024,1024)
        x = self.norm_out(x)
        # (B,96/2,4,1024,1024)
        x = self.nonlinearity(x)

        if feat_cache is not None:
            idx = feat_idx[0]
            # (B,96/2,2,1024,1024)
            cache_x = x[:, :, -2:, :, :].clone()
            
            # (B,96/2,4,1024,1024) -> (B,3,4,1024,1024)
            x = self.conv_out(x, feat_cache[idx])

            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            # (B,96/2,4,1024,1024) -> (B,3,4,1024,1024)
            x = self.conv_out(x)

        # (B,3,4,1024,1024)
        return x


class UpBlock(nn.Module):
    def __init__(self, in_dim, out_dim, mode=None):
        super().__init__()

        self.resnets = nn.ModuleList([
            ResNet(in_dim, out_dim),
            ResNet(out_dim, out_dim),
            ResNet(out_dim, out_dim),
        ])

        self.upsamplers = None

        if (mode == 'sample3d'):
            self.upsamplers = nn.ModuleList([
                UpSample3D(out_dim),
            ])
        elif(mode == 'sample2d'):
            self.upsamplers = nn.ModuleList([
                UpSample2D(out_dim),
            ])
        else:
            self.upsamplers = None

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        x = self.resnets[0](x, feat_cache, feat_idx)
        x = self.resnets[1](x, feat_cache, feat_idx)
        x = self.resnets[2](x, feat_cache, feat_idx)

        if self.upsamplers is not None:
            x = self.upsamplers[0](x, feat_cache, feat_idx)

        return x


class UpSample2D(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.resample = nn.Sequential(
            nn.Upsample(scale_factor=(2.0, 2.0), mode="nearest-exact"),
            nn.Conv2d(dim, dim // 2, 3, padding=1),
        )

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        b, c, t, h, w = x.size()
        
        # (B,C,T,H,W) -> (B,T,C,H,W) -> (BxT,C,H,W)
        x = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        # (BxT,C,H,W) -> (BxT,C,2xH,2xW) -> (BxT,C/2,2xH,2xW)
        x = self.resample(x)
        # (BxT,C/2,2xH,2xW) -> (B,T,C/2,2xH,2xW) -> (B,C/2,T,2xH,2xW)
        x = x.view(b, t, x.size(1), x.size(2), x.size(3)).permute(0, 2, 1, 3, 4)

        # (B,C/2,T,2xH,2xW)
        return x


class UpSample3D(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.resample = nn.Sequential(
            nn.Upsample(scale_factor=(2.0, 2.0), mode="nearest-exact"),
            nn.Conv2d(dim, dim // 2, 3, padding=1),
        )

        self.time_conv = CausalConv3d(dim, dim * 2, (3, 1, 1), padding=(1, 0, 0))

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        b, c, t, h, w = x.size()
        if feat_cache is not None:
            idx = feat_idx[0]
            if feat_cache[idx] is None:
                # i=0
                feat_cache[idx] = "Rep"
                feat_idx[0] += 1
            else:
                # i>0
                # (B,C,2,W,H)
                cache_x = x[:, :, -2:, :, :].clone()

                # i>1
                # 靠后面的下采样和上采样，时间序列维度只有1帧，使用上个循环的1帧和当前1帧拼接成2帧
                # (B,C,1,H,W) (B,C,1,H,W) => (B,C,2,H,W)
                if cache_x.shape[2] < 2 and feat_cache[idx] is not None and feat_cache[idx] != "Rep":
                    cache_x = torch.cat([
                        feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), 
                        cache_x
                    ], dim=2)
                
                if feat_cache[idx] == "Rep":
                    # i=1
                    # (B,C,T,H,W) => (B,2xC,T,H,W)
                    x = self.time_conv(x)
                else:
                    # i>1
                    # (B,C,T,H,W) => (B,2xC,T,H,W)
                    x = self.time_conv(x, feat_cache[idx])

                # (B,C,2,H,W)
                feat_cache[idx] = cache_x
                feat_idx[0] += 1

                # (B,2xC,T,H,W) -> (B,2,C,T,H,W)
                x = x.reshape(b, 2, c, t, h, w)
                # (B,1,C,T,H,W) (B,1,C,T,H,W) -> (B,1,C,2xT,H,W)
                x = torch.stack((x[:, 0, :, :, :, :], x[:, 1, :, :, :, :]), 3)
                # (B,1,C,2xT,H,W) -> (B,C,2xT,H,W)
                x = x.reshape(b, c, t * 2, h, w)

        # 2xT
        t = x.shape[2]

        # (B,C,2xT,W,H) -> (B,2xT,C,W,H) -> (Bx2xT,C,W,H)
        x = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        
        # (Bx2xT,C,W,H) -> (Bx2xT,C,2xW,2xH) -> (Bx2xT,C/2,2xW,2xH)
        x = self.resample(x)
        
        # (Bx2xT,C/2,2xH,2xW) -> (B,2xT,C/2,2xH,2xW) -> (B,C/2,2xT,2xH,2xW)
        x = x.view(b, t, x.size(1), x.size(2), x.size(3)).permute(0, 2, 1, 3, 4)

        return x


class DiagonalGaussianDistribution():
    def __init__(self, input_tensor, deterministic=False):
        self.mean, self.logvar = torch.chunk(input_tensor, 2, dim=1)

        # 对数方差
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        # 标准差
        self.std = torch.exp(0.5 * self.logvar)
        # 方差
        self.var = torch.exp(self.logvar)
        self.deterministic = deterministic

        if self.deterministic:
            self.var = self.std = torch.zeros_like(
                self.mean, device=self.mean.device, dtype=self.mean.dtype
            )

    def sample(self, generator=None):
        if self.deterministic:
            return self.mean

        eps = randn_tensor(
            self.mean.shape,
            generator=generator,
            device=self.mean.device,
            dtype=self.mean.dtype
        )
        return self.mean + self.std * eps

    def kl(self):
        if self.deterministic:
            return torch.Tensor([0.0])

        return 0.5 * torch.sum(
            torch.pow(self.mean, 2) + self.var - 1.0 - self.logvar,
            dim=[1, 2, 3],
        )

    def mean(self):
        return self.mean