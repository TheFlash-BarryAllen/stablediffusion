import torch
from diffusers import ConfigMixin, SchedulerMixin
from diffusers.configuration_utils import register_to_config

class Scheduler(SchedulerMixin, ConfigMixin):
    @register_to_config
    def __init__(
        self,
        num_train_timesteps = 1000, # 训练时总步数
        beta_start = 0.00085,       # β起始值
        beta_end = 0.012            # β结束值
    ):
        # 参数初始化
        self.num_train_timesteps = num_train_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end

        # 对 √β 做线性插值，再平方回来得到 β
        self.betas = torch.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps, dtype=torch.float32) ** 2
        # 计算 α = 1 - β
        self.alphas = 1.0 - self.betas  
        # cumprod方法求 α累乘，里面有 1000 个 α累乘 的值，分别对应每个时间步的 α累乘                     
        self.alphas_line = torch.cumprod(self.alphas, dim=0) 

        self.final_alpha_line = torch.tensor(1.0)   # α的边界，t=0时α=1

        # 通过arange生成时间序列，然后通过flip反转时间序列，生成[999, 998, 997, ..., 0]
        self.timesteps = torch.flip(torch.arange(0, num_train_timesteps, dtype=torch.int64), dims=[0])  


    # 设置 推理 步数
    def set_timesteps(self, num_inference_steps):   
        # num_inference_steps = 50  即1000步变成50步，每一步要跨20个索引

        # 50
        self.num_inference_steps = num_inference_steps

        # 20  
        step_ratio = 1000 // num_inference_steps

        # [0,20,40,...,980]
        timesteps = (torch.arange(0, num_inference_steps) * step_ratio).round()

        # [980,960,...,20,0]
        timesteps = torch.flip(timesteps, dims=[0]).to(torch.int64)

        # [981, 961, ..., 21, 1]
        timesteps = timesteps + 1

        # [981, 961, ..., 21, 21, 1]
        self.timesteps = torch.concat([
            timesteps[:-1],     # 去掉最后一个元素（即去掉 1）
            timesteps[-2:-1],   # 取倒数第二个元素（即原来的 21）
            timesteps[-1:]      # 取最后一个元素（即原来的 1）
        ])
    

    # 训练阶段的前向加噪方法
    def add_noise(self, x_0, noise, timesteps):     
        # 训练的时候，一个批次是32张图像，因此 timesteps 是一个32维的向量，里面的每个元素都是一个整数，表示每张图像对应的时间步数
        
        # 将 alphas_line 转换到与 x_0 相同的设备和数据类型
        self.alphas_line = self.alphas_line.to(device=x_0.device)
        self.alphas_line = self.alphas_line.to(dtype=x_0.dtype)
        timesteps = timesteps.to(x_0.device)

        # （batch，）       
        # self.alphas_line 是长度为 1000 的数组；gather方法根据 timesteps 里存的 32 个索引号，从 alphas_line 里把对应的 32 个 α累乘 值掏出来，再开方得到32个 α累乘 的平方根
        sqrt_alpha_line_i = torch.sqrt(self.alphas_line.gather(0, timesteps))
        # 开方得到32个 (1-α累乘) 的平方根
        sqrt_one_minus_alpha_line_i = torch.sqrt(1 - self.alphas_line.gather(0, timesteps))  

        # (batch,)  -> (batch, 1, 1, 1)
        # reshape 成 (32, 1, 1, 1) 后，利用 PyTorch 的广播机制（Broadcasting），这 32 个系数会自动拉伸，分别去乘 32 张图片里的每一个像素（3x512x512 个像素全乘同一个系数）
        sqrt_alpha_line_i = sqrt_alpha_line_i.reshape(-1, 1, 1, 1)  # 32个 α累乘 的平方根
        sqrt_one_minus_alpha_line_i = sqrt_one_minus_alpha_line_i.reshape(-1, 1, 1, 1)  # 32个 (1-α累乘) 的平方根

        # (batch, chanel, height, width)    批量计算32张图像的第 i 步的加噪结果
        x_i = sqrt_alpha_line_i * x_0 + sqrt_one_minus_alpha_line_i * noise  # 训练阶段的加噪公式

        return x_i


    # 推理阶段的去噪方法
    def step(self, model_output, timesteps, x_i):    # 推理只生成一张图片，这里的 timesteps 是一个标量
        # (1,)
        prev_timestep = timesteps - (1000 // self.num_inference_steps)  # 计算上一个时间步的索引

        # (1,)
        alphas_line_i = self.alphas_line[timesteps]  # 当前时间步的 累乘 α

        # (1,)
        alphas_line_i_n = self.alphas_line[prev_timestep] if prev_timestep >= 0 else self.final_alpha_line  # 上一个时间步的 累乘 α

        # (1,)      去噪公式参数
        a = torch.sqrt(alphas_line_i_n / alphas_line_i)
        b = torch.sqrt(1 - alphas_line_i_n)
        c = torch.sqrt(alphas_line_i_n * (1 - alphas_line_i) / alphas_line_i)

        # (1, chanel, height, width)
        x_i_n = a * x_i + (b - c) * model_output  # 推理阶段的去噪公式

        return SchedulerStepOutput(x_i_n)

    def __len__(self):
        return self.num_inference_steps


class SchedulerStepOutput:
    def __init__(self, prev_sample):
        self.prev_sample = prev_sample




