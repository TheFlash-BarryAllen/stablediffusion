from diffusers import ModelMixin, ConfigMixin
from diffusers.configuration_utils import register_to_config
from torch import nn
from torch.nn import functional as F
import torch
import math

# 主模型
class DiffusionModel(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(
        self,
        in_channels=4,  # 输入通道数
        out_channels=4,  # 输出通道数
        timesteps_embedding_dim=320*4,  # 时间步嵌入维度 1280
        block_out_channels=[320, 640, 1280, 1280],  # 四个下采样和上采样层的输入特征通道数
        group_norm_num=32,  # 分组归一化的组数
    ):
        super().__init__()

        self.time_embedding = TimestepsEmbedding(block_out_channels[0], timesteps_embedding_dim) 

        # （4，64，64）->（320，64，64）  先通过一个卷积层把输入的4通道图像变成320通道特征图
        self.conv_in = nn.Conv2d(in_channels, block_out_channels[0], kernel_size=3, padding=1) 

        # 下采样模块，包含四个下采样层
        self.down_blocks = nn.ModuleList([
            # 前三个下采样层是有transformer模块，最后一个下采样层没有transformer模块
            # （320，64，64）->（320，32，32）
            AttnDownBlock(block_out_channels[0], block_out_channels[0], timesteps_embedding_dim),
            # （320，32，32）->（640，16，16）
            AttnDownBlock(block_out_channels[0], block_out_channels[1], timesteps_embedding_dim),
            # （640，16，16）->（1280，8，8）
            AttnDownBlock(block_out_channels[1], block_out_channels[2], timesteps_embedding_dim),
            # （1280，8，8）->（1280，8，8）
            DownBlock(block_out_channels[2], block_out_channels[3], timesteps_embedding_dim, add_downsample=False),
        ])

        self.mid_block = MidBlock(block_out_channels[-1], timesteps_embedding_dim)  # 中间模块

        # 上采样模块，包含四个上采样层
        self.up_blocks = nn.ModuleList([
            # 第一个上采样层没有transformer模块，后三个上采样层有transformer模块
            # 每个层第一个参数是输入特征通道数，第二个参数是下采样层对应的特征通道数，第三个参数是输出特征通道数
            # （1280，8，8）->（1280，16，16）
            UpBlock(block_out_channels[2], block_out_channels[3], block_out_channels[3], timesteps_embedding_dim),
            # （1280，16，16）->（1280，32，32）
            AttnUpBlock(block_out_channels[1], block_out_channels[2], block_out_channels[3],timesteps_embedding_dim),
            # （1280，32，32）->（640，64，64）
            AttnUpBlock(block_out_channels[0], block_out_channels[1], block_out_channels[2],timesteps_embedding_dim),
            # （640，64，64）->（320，64，64）
            AttnUpBlock(block_out_channels[0], block_out_channels[0], block_out_channels[1],timesteps_embedding_dim),
        ])

        self.conv_norm_out = nn.GroupNorm(group_norm_num, block_out_channels[0])  # 分组归一化
        self.conv_act_out = nn.SiLU()  # 激活函数
        self.conv_out = nn.Conv2d(block_out_channels[0], out_channels, kernel_size=3, padding=1)  # 输出卷积层，将特征图变回4通道图像

    def forward(self, x_i, timesteps, promt_embeds):
        # (B)-> (B, 1280)
        timesteps_emdeds = self.time_embedding(timesteps)  # 将时间步数转换为嵌入向量

        # (B, 4, 64, 64) -> (B, 320, 64, 64)
        sample = self.conv_in(x_i)  # 将输入图像通过卷积层变为特征图

        # 下采样+收集跳跃连接
        down_block_res_samples = []  # 用于存储下采样模块每个层的输出特征图，供上采样模块使用
        for block in self.down_blocks:

            # 下采样模块的前向传播；    sample：当前层的"主输出"，会继续往下传。
            # res_samples：当前层内部每一小步产生的所有中间特征（因为一个 DownBlock 内部可能有好几层 ResNet+Attention），这些特征要留着给上采样用。
            res_samples, sample = block(sample, timesteps_emdeds, promt_embeds)  

            # 用 down_block_res_samples 存着每个层中每一步产生的中间特征，供上采样模块使用
            down_block_res_samples.append(res_samples)

        # 把主输出 sample 传给中间模块，得到中间模块的输出特征图
        sample = self.mid_block(sample, timesteps_emdeds, promt_embeds) # 中间模块的前向传播


        # 上采样+跳跃连接
        for block in self.up_blocks:

            # pop() 是从列表尾部弹出（后进先出），从down_block_res_samples取出对应的下采样层的输出特征
            res_samples = down_block_res_samples.pop()

            # 上采样模块的前向传播
            sample = block(sample, res_samples, timesteps_emdeds, promt_embeds)


        sample = self.conv_norm_out(sample)  # 分组归一化

        sample = self.conv_act_out(sample)  # 激活函数

        sample = self.conv_out(sample)  # 输出卷积层，将特征图变回4通道图像

        return DiffusionModelOutput(sample=sample)  # 返回模型输出


# 时间编码层
class TimestepsEmbedding(nn.Module):
    def __init__(self, in_channel_num, timesteps_embedding_dim):
        super().__init__()

        self.in_channel_num = in_channel_num

        self.linear_1 = nn.Linear(in_channel_num, timesteps_embedding_dim)  # 线性层1
        self.act_1 = nn.SiLU()  # 激活函数
        self.linear_2 = nn.Linear(timesteps_embedding_dim, timesteps_embedding_dim)  # 线性层2

    def forward(self, timesteps):
        # (B, 1) -> (B, 320)
        pos_encoded = self.pos_encoding(timesteps)  # 将时间步数转换为位置编码向量

        # (B, 320) -> (B, 1280)
        timesteps_embed = self.linear_1(pos_encoded)  # 线性变换
        timesteps_embed = self.act_1(timesteps_embed)  # 激活函数

        # (B, 1280) -> (B, 1280)
        timesteps_embed = self.linear_2(timesteps_embed)  # 线性

        return timesteps_embed

    def pos_encoding(self, timesteps):
        # 最终要生成 320 维的编码，一半用 sin，一半用 cos，所以各占 160 维
        half_dim = self.in_channel_num // 2

        col = torch.arange(half_dim, device=timesteps.device) # 生成[0, 1, ..., 159]，这是频率索引
        col = torch.exp(-math.log(10000) * col / (half_dim))  # 计算频率因子，得到[1.0, 0.794, 0.630, ..., 0.0001]，这是每个频率的缩放因子
        col = col[None, :]      # 把形状从 (160,) 变成 (1, 160)，为了后面广播

        # 假设 timesteps = [981]，形状 (1,)。timesteps[:, None] 变成 (1, 1)，方便和 col 相乘
        row = timesteps[:, None].float()    

        # 广播相乘：(B, 1) * (1, 160) = (B, 160)
        emb = row * col
        # 对每个元素分别取 sin 和 cos。拼接后形状 (B, 320)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

        return emb

# 带transformer的下采样块，Resnet+Attention+Resnet+Attention+DownSample
class AttnDownBlock(nn.Module):
    def __init__(self, in_channel_num, out_channel_num, timesteps_embedding_dim, add_downsample = True):
        super().__init__()

        self.resnets = nn.ModuleList([
            Resnet(in_channel_num, out_channel_num, timesteps_embedding_dim),
            Resnet(out_channel_num, out_channel_num, timesteps_embedding_dim)
        ])

        self.attentions = nn.ModuleList([
            Transformer(out_channel_num),
            Transformer(out_channel_num)
        ])

        self.downsamplers = None
        if add_downsample:
            self.downsamplers = nn.ModuleList([
                DownSample(out_channel_num, out_channel_num)
            ])

    def forward(self, hidden_states, timesteps_embeds, encoder_hidden_states):
        # 输入：   hidden_states：当前特征图，形状 (B, C, H, W)。
        #         timesteps_embeds：时间嵌入，形状 (B, 1280)。
        #         encoder_hidden_states：文本嵌入（prompt），形状 (B, 77, 768)（CLIP 输出）
        
        # 一个元组，第一个元素是输入的原始 hidden_states（这是跳跃连接的关键）
        output_res_states = (hidden_states,)

        # 第一个 Resnet+Attention
        # (B, 320, 64, 64) + 时间嵌入 ->  (B, 320, 64, 64)
        hidden_states = self.resnets[0](hidden_states, timesteps_embeds)
        # (B, 320, 64, 64) + 文本嵌入 ->  (B, 320, 64, 64)
        hidden_states = self.attentions[0](hidden_states, encoder_hidden_states)
        # 把 hidden_states 追加到 output_res_states 里
        output_res_states += (hidden_states,)

        # 第二个 Resnet+Attention
        # (B, 320, 64, 64) + 时间嵌入 ->  (B, 320, 64, 64)
        hidden_states = self.resnets[1](hidden_states, timesteps_embeds)
        # (B, 320, 64, 64) + 文本嵌入 ->  (B, 320, 64, 64)
        hidden_states = self.attentions[1](hidden_states, encoder_hidden_states)
        # 把 hidden_states 追加到 output_res_states 里
        output_res_states += (hidden_states,)   # 现在output_res_states = (原始输入, 第1轮结果, 第2轮结果)

        if self.downsamplers is not None:
            # 下采样，输出特征图的尺寸减半, (B, 320, 64, 64) -> (B, 320, 32, 32)
            hidden_states = self.downsamplers[0](hidden_states)

        # output_res_states：一个元组（跳跃连接用）;   hidden_states：最终输出（传给下一层）
        return output_res_states, hidden_states  # 返回当前层的输出特征图和下采样后的特征图

# 不带transformer的下采样块，Resnet+Resnet
class DownBlock(nn.Module):
    def __init__(self, in_channel_num, out_channel_num, timesteps_embedding_dim, add_downsample = True):
        super().__init__()

        self.resnets = nn.ModuleList([
            Resnet(in_channel_num, out_channel_num, timesteps_embedding_dim),
            Resnet(out_channel_num, out_channel_num, timesteps_embedding_dim)
        ])

        self.downsamplers = None
        if add_downsample:
            self.downsamplers = nn.ModuleList([
                DownSample(out_channel_num, out_channel_num)
            ])

    def forward(self, hidden_states, timesteps_embeds, encoder_hidden_states):
        # 输入：   hidden_states：当前特征图，形状 (B, C, H, W)。
        #         timesteps_embeds：时间嵌入，形状 (B, 1280)。
        #         encoder_hidden_states：文本嵌入（prompt），形状 (B, 77, 768)（CLIP 输出）
        
        # 一个元组，第一个元素是输入的原始 hidden_states（这是跳跃连接的关键）
        output_res_states = (hidden_states,)

        # 第一个 Resnet
        # (B, 320, 64, 64) + 时间嵌入 ->  (B, 320, 64, 64)
        hidden_states = self.resnets[0](hidden_states, timesteps_embeds)
        # 把 hidden_states 追加到 output_res_states 里
        output_res_states += (hidden_states,)   # 现在output_res_states = (原始输入, 第1轮结果)

        # 第二个 Resnet
        # (B, 320, 64, 64) + 时间嵌入 ->  (B, 320, 64, 64)
        hidden_states = self.resnets[1](hidden_states, timesteps_embeds)
        # 把 hidden_states 追加到 output_res_states 里
        output_res_states += (hidden_states,)   # 现在output_res_states = (原始输入, 第1轮结果，第2轮结果)

        if self.downsamplers is not None:
            # 下采样，输出特征图的尺寸减半
            hidden_states = self.downsamplers[0](hidden_states)

        return output_res_states, hidden_states  # 返回当前层的输出特征图和下采样后的特征图

class Resnet(nn.Module):
    def __init__(self, in_channel_num, out_channel_num, timesteps_embedding_dim, group_norm_num=32):
        super().__init__()

        self.norm1 = nn.GroupNorm(group_norm_num, in_channel_num)  # 分组归一化
        self.act1 = nn.SiLU()  # 激活函数
        self.conv1 = nn.Conv2d(in_channel_num, out_channel_num, kernel_size=3, stride=1, padding=1)  # 卷积层

        self.time_emb_proj = nn.Linear(timesteps_embedding_dim, out_channel_num)  # 时间嵌入线性变换

        self.norm2 = nn.GroupNorm(group_norm_num, out_channel_num)  # 分组归一化
        self.act2 = nn.SiLU()  # 激活函数
        self.conv2 = nn.Conv2d(out_channel_num, out_channel_num, kernel_size=3, stride=1, padding=1)  # 卷积层


        if in_channel_num != out_channel_num:
            self.conv_shortcut = nn.Conv2d(in_channel_num, out_channel_num, kernel_size=1, stride=1, padding=0)  # 残差连接卷积层

    def forward(self, hidden_states, timesteps_embeds):
        # 输入：   hidden_states：当前特征图，形状 (B, C, H, W)。
        #         timesteps_embeds：时间嵌入，形状 (B, 1280)。

        # 残差 
        residual = hidden_states  

        # 分组归一化    (B, in_channel_num, H, W)
        hidden_states = self.norm1(hidden_states)
        # 激活函数      (B, in_channel_num, H, W)
        hidden_states = self.act1(hidden_states)  
        # 卷积层 (B, in_channel_num, H, W) -> (B, out_channel_num, H, W)
        hidden_states = self.conv1(hidden_states)  # (B, 320, 32, 32)-> (B, 640, 32, 32)

        # 时间嵌入线性变换，线性层把时间嵌入从 1280 维投影到 out_channel_num 维
        timesteps_embeds = self.time_emb_proj(timesteps_embeds)  # (B, 1280) -> (B, out_channel_num)
        # (B, out_channel_num) -> (B, out_channel_num, 1, 1)
        timesteps_embeds = timesteps_embeds[:, :, None, None]  

        # 特征 加上时间嵌入 (B, out_channel_num, H, W)
        hidden_states = hidden_states + timesteps_embeds  

        # 分组归一化    (B, out_channel_num, H, W)
        hidden_states = self.norm2(hidden_states) 
        # 激活函数      (B, out_channel_num, H, W)
        hidden_states = self.act2(hidden_states) 
        # 卷积  (B, out_channel_num, H, W)-> (B,out_channel_num,H,W)
        hidden_states = self.conv2(hidden_states)

        if hasattr(self, 'conv_shortcut') and self.conv_shortcut is not None:

            residual = self.conv_shortcut(residual)  # 残差连接卷积层

        hidden_states = hidden_states + residual    # 残差连接

        return hidden_states

class Transformer(nn.Module):
    def __init__(self, in_channel_num):
        super().__init__()

        # 组归一化
        self.norm = nn.GroupNorm(32, in_channel_num)

        # 1×1 卷积，做一次通道维度的线性变换
        self.proj_in = nn.Conv2d(in_channel_num, in_channel_num, kernel_size=1, stride=1, padding=0)

        # Self-Attn → Cross-Attn → FFN
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(in_channel_num)
        ])

        # 把注意力处理后的特征再变换一次，最后输出
        self.proj_out = nn.Conv2d(in_channel_num, in_channel_num, kernel_size=1, stride=1, padding=0)

    def forward(self, hidden_states, encoder_hidden_states):
        # 把 B、C、H、W 存下来，后面 reshape 用
        B, C, H, W = hidden_states.shape

        # 保存残差
        residual = hidden_states

        # 归一化  (B,C,H,W) -> (B,C,H,W)
        hidden_states = self.norm(hidden_states)

        # 1×1 卷积，做一次通道维度的线性变换    (B,C,H,W) -> (B,C,H,W)
        hidden_states = self.proj_in(hidden_states)

        # (B,C,H,W) -> (B,H,W，C) -> (B,H*W，C) 
        hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(B, H*W, C)

        # 输入 hidden_states：(B, H*W, C)。
        # 输入 encoder_hidden_states：(B, seq_text, context_dim)
        for block in self.transformer_blocks:
            hidden_states = block(hidden_states, encoder_hidden_states)

        # (B, H*W, C) -> (B, H, W, C) -> (B, C, H, W) 
        hidden_states = hidden_states.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()

        # 1×1 卷积
        hidden_states = self.proj_out(hidden_states)

        return hidden_states + residual

class TransformerBlock(nn.Module):
    def __init__(self, in_channel_num):
        super().__init__()

        # 层归一化和自注意力
        self.norm1 = nn.LayerNorm(in_channel_num)
        self.attn1 = Attension(in_channel_num)

        # 层归一化和交叉注意力
        self.norm2 = nn.LayerNorm(in_channel_num)
        self.attn2 = Attension(in_channel_num, cross_attention_dim = 768)

        # 层归一化和前馈网络
        self.norm3 = nn.LayerNorm(in_channel_num)
        self.ff = FeedForward(in_channel_num)

    def forward(self, hidden_states, encoder_hidden_states):
        # ========== 第 1 段：Self-Attention ==========
        # 保存残差
        residual = hidden_states
        # 层归一化  (B, H*W, C)
        norm_hidden_states = self.norm1(hidden_states)
        # 自注意力
        attn_output = self.attn1(norm_hidden_states)
        # 残差连接
        hidden_states = residual + attn_output

        # ========== 第 2 段：Cross-Attention ========== 
        # 保存残差
        residual = hidden_states
        # 层归一化  (B, H*W, C)
        norm_hidden_states = self.norm2(hidden_states)
        # 交叉注意力    (B, H*W, C)
        attn_output = self.attn2(norm_hidden_states, encoder_hidden_states)
        # 残差连接
        hidden_states = residual + attn_output

        # ========== 第 3 段：FeedForward ==========
        # 保存残差
        residual = hidden_states       
        # 层归一化
        norm_hidden_states = self.norm3(hidden_states)
        # 前馈网络
        ff_output = self.ff(norm_hidden_states)
        # 残差连接
        hidden_states = residual + ff_output

        return hidden_states

class Attension(nn.Module):
    def __init__(self, in_channel_num, cross_attention_dim=None):
        super().__init__()

        # 注意力头数
        self.heads = 8  

        # 输入特征维度除以头数，得到每个头对应的特征维度
        self.head_dim = in_channel_num // self.heads

        # 如果没有cross_attention, 则把in_channel_num赋值给cross_attention
        if cross_attention_dim is None:
            cross_attention_dim = in_channel_num

        # 三个线性层
        self.to_q = nn.Linear(in_channel_num, in_channel_num, bias=False)
        self.to_k = nn.Linear(cross_attention_dim, in_channel_num, bias=False)
        self.to_v = nn.Linear(cross_attention_dim, in_channel_num, bias=False)

        # 输出
        self.to_out = nn.ModuleList([
            nn.Linear(in_channel_num, in_channel_num)
        ])


    def forward(self, hidden_states, encoder_hidden_states=None):

        # 提取B, H_W, C
        B, H_W, C = hidden_states.shape

        # 生成q值   (B,HxW,C) -> (B,HxW,C)
        query = self.to_q(hidden_states)

        # 如果没有encoder_hidden_states, 则把hidden_states赋值给encoder_hidden_states
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        # 生成k值       (B,HxW,C) -> (B,HxW,C) or (B,seq_len,C) -> (B,seq_len,C)
        key = self.to_k(encoder_hidden_states)
        # 生成v值
        value = self.to_v(encoder_hidden_states)

        # 拆成多头
        # (B,HxW,C) -> （B，HxW, heads，head_dim）-> (B, heads, HxW，head_dim)
        query = query.view(B, -1, self.heads, self.head_dim).transpose(1, 2)

        # (B,HxW,C) -> (B,heads,HxW,head_dim) or (B,seq_len,C) -> (B,heads,seq_len,head_dim)
        key = key.view(B, -1, self.heads, self.head_dim).transpose(1, 2)
        value = value.view(B, -1, self.heads, self.head_dim).transpose(1, 2)

        # (B,heads,HxW,head_dim)
        hidden_states = F.scaled_dot_product_attention(query, key, value)

        # (B,heads,HxW,head_dim) -> (B,HxW,heads,head_dim) -> (B,HxW,C)
        hidden_states = hidden_states.transpose(1,2).reshape(B, -1, self.heads * self.head_dim)

        # (B,HxW,C) -> (B,HxW,C)
        hidden_states = self.to_out[0](hidden_states)

        return hidden_states  

class FeedForward(nn.Module):
    def __init__(self, in_channel_num):
        super().__init__()

        self.net = nn.ModuleList([
            GEGLU(in_channel_num, in_channel_num*4),
            nn.Dropout(0.0),
            nn.Linear(in_channel_num*4, in_channel_num)
        ])

    def forward(self, hidden_states):
        hidden_states = self.net[0](hidden_states)

        hidden_states = self.net[1](hidden_states)

        hidden_states = self.net[2](hidden_states)

        return hidden_states

class GEGLU(nn.Module):
    def __init__(self, in_channel_num, out_channel_num):
        super().__init__()

        self.proj = nn.Linear(in_channel_num, out_channel_num*2)

    def forward(self, hidden_states):
        hidden_states = self.proj(hidden_states)

        hidden_states, gate = hidden_states.chunk(2, dim = -1)

        hidden_states = hidden_states * F.gelu(gate)

        return hidden_states

class DownSample(nn.Module):
    def __init__(self, in_channel_num, out_channel_num):
        super().__init__()

        self.conv = nn.Conv2d(in_channel_num, out_channel_num, kernel_size=3, padding=1, stride=2)

    def forward(self, hidden_states):

        hidden_states = self.conv(hidden_states)

        return hidden_states

class MidBlock(nn.Module):
    def __init__(self, in_channel, timesteps_embed_dim):
        super().__init__()

        self.resnets = nn.ModuleList([
            Resnet(in_channel, in_channel, timesteps_embed_dim),
            Resnet(in_channel, in_channel, timesteps_embed_dim)
        ])

        self.attentions = nn.ModuleList([
            Transformer(in_channel)
        ])

    def forward(self, hidden_states, timesteps_embed, encoder_hidden_states):
        hidden_states  = self.resnets[0](hidden_states, timesteps_embed)
        hidden_states  = self.attentions[0](hidden_states, encoder_hidden_states)
        hidden_states  = self.resnets[1](hidden_states, timesteps_embed)

        return hidden_states

class UpBlock(nn.Module):
    def __init__(
        self, 
        in_channel_num, 
        out_channel_num, 
        res_encoder_channel_num, 
        timesteps_embedding_dim,
        add_upsample = True
    ):
        super().__init__()

        self.resnets = nn.ModuleList([
            Resnet(res_encoder_channel_num + out_channel_num, out_channel_num, timesteps_embedding_dim),
            Resnet(out_channel_num + out_channel_num, out_channel_num, timesteps_embedding_dim),
            Resnet(in_channel_num + out_channel_num, out_channel_num, timesteps_embedding_dim)
        ])

        self.upsamplers = None
        if add_upsample:
            self.upsamplers = nn.ModuleList([
                UpSample(out_channel_num, out_channel_num)
            ])

    def forward(self, hidden_states, encoder_res_hidden_states, timesteps_embeds, encoder_hidden_states):
        # 输入：   hidden_states：当前特征图，形状 (B, C, H, W)。
        #         timesteps_embeds：时间嵌入，形状 (B, 1280)。
        #         encoder_hidden_states：文本嵌入（prompt），形状 (B, 77, 768)（CLIP 输出）
        
        
        hidden_states = torch.cat([hidden_states, encoder_res_hidden_states[2]], dim=1)

        hidden_states = self.resnets[0](hidden_states, timesteps_embeds)

        hidden_states = torch.cat([hidden_states, encoder_res_hidden_states[1]], dim=1)

        hidden_states = self.resnets[1](hidden_states, timesteps_embeds)

        hidden_states = torch.cat([hidden_states, encoder_res_hidden_states[0]], dim=1)

        hidden_states = self.resnets[2](hidden_states, timesteps_embeds)
        

        if self.upsamplers is not None:
            # 上采样，输出特征图的尺寸翻倍
            hidden_states = self.upsamplers[0](hidden_states)

        return hidden_states

class AttnUpBlock(nn.Module):
    def __init__(
        self, 
        in_channel_num, 
        out_channel_num, 
        res_encoder_channel_num, 
        timesteps_embedding_dim,
        add_upsample = True
    ):
        super().__init__()

        self.resnets = nn.ModuleList([
            Resnet(res_encoder_channel_num + out_channel_num, out_channel_num, timesteps_embedding_dim),
            Resnet(out_channel_num + out_channel_num, out_channel_num, timesteps_embedding_dim),
            Resnet(in_channel_num + out_channel_num, out_channel_num, timesteps_embedding_dim)
        ])

        self.attentions = nn.ModuleList([
            Transformer(out_channel_num),
            Transformer(out_channel_num),
            Transformer(out_channel_num)
        ])

        self.upsamplers = None
        if add_upsample:
            self.upsamplers = nn.ModuleList([
                UpSample(out_channel_num, out_channel_num)
            ])

    def forward(self, hidden_states, encoder_res_hidden_states, timesteps_embeds, encoder_hidden_states):
        # 输入：   hidden_states：当前特征图，形状 (B, C, H, W)。
        #         timesteps_embeds：时间嵌入，形状 (B, 1280)。
        #         encoder_hidden_states：文本嵌入（prompt），形状 (B, 77, 768)（CLIP 输出）
        
        
        hidden_states = torch.cat([hidden_states, encoder_res_hidden_states[2]], dim=1)

        hidden_states = self.resnets[0](hidden_states, timesteps_embeds)

        hidden_states = self.attentions[0](hidden_states, encoder_hidden_states)

        hidden_states = torch.cat([hidden_states, encoder_res_hidden_states[1]], dim=1)

        hidden_states = self.resnets[1](hidden_states, timesteps_embeds)

        hidden_states = self.attentions[1](hidden_states, encoder_hidden_states)

        hidden_states = torch.cat([hidden_states, encoder_res_hidden_states[0]], dim=1)

        hidden_states = self.resnets[2](hidden_states, timesteps_embeds)

        hidden_states = self.attentions[2](hidden_states, encoder_hidden_states)

        if self.upsamplers is not None:
            # 上采样，输出特征图的尺寸翻倍
            hidden_states = self.upsamplers[0](hidden_states)

        return hidden_states

class UpSample(nn.Module):
    def __init__(self, in_channel_num, out_channel_num):
        super().__init__()

        self.conv = nn.Conv2d(in_channel_num, out_channel_num, kernel_size=3, padding=1, stride=1)

    def forward(self, hidden_states):

        hidden_states = F.interpolate(hidden_states, scale_factor=2, mode="nearest")

        hidden_states = self.conv(hidden_states)

        return hidden_states

