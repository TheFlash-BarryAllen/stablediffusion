from diffusers import DiffusionPipeline
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2Tokenizer
import torch
from torchvision.utils import save_image
from transformer import Transformer
from scheduler import Scheduler
from vae import VAE

class Pipeline(DiffusionPipeline):
    def __init__(self, vae, text_encoder, tokenizer, transformer, scheduler):
        super().__init__()

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            transformer=transformer,
            scheduler=scheduler,
        )

        self.prompt_template_encode = "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
        
        self.prompt_template_encode_start_idx = 34

    def _extract_masked_hidden(self, hidden_states, mask):
        # (B,1024+34)
        bool_mask = mask.bool()

        # (B,1)
        valid_lengths = bool_mask.sum(dim=1)

        # (total_seq_len,3584)
        selected = hidden_states[bool_mask]
        # [(valid_seq_len,3584),...,]
        split_result = torch.split(selected, valid_lengths.tolist(), dim=0)

        # [(valid_seq_len,3584),...,]
        return split_result

    def encode_prompt(self, prompt,device=None):
        template = self.prompt_template_encode
        drop_idx = self.prompt_template_encode_start_idx
        prompt = template.format(prompt)

        # (B,1024+34)
        txt_tokens = self.tokenizer(
            prompt, 
            max_length=1024 + drop_idx, 
            padding=True, 
            truncation=True, 
            return_tensors="pt"
        ).to(device)

        encoder_hidden_states = self.text_encoder(
            input_ids=txt_tokens.input_ids,
            attention_mask=txt_tokens.attention_mask,
            output_hidden_states=True,
        )

        # (B,1024+34,3584)
        hidden_states = encoder_hidden_states.hidden_states[-1]

        # [(valid_seq_len,3584),...,]
        split_hidden_states = self._extract_masked_hidden(hidden_states, txt_tokens.attention_mask)
        
        # [(valid_seq_len - 34,3584),...,]
        split_hidden_states = [e[drop_idx:] for e in split_hidden_states]
        
        # [(valid_seq_len - 34),...,]
        attn_mask_list = [torch.ones(e.size(0), dtype=torch.long, device=e.device) for e in split_hidden_states]
        
        # valid_seq_len - 34
        max_seq_len = max([e.size(0) for e in split_hidden_states])

        # (B,max_seq_len,3584)
        prompt_embeds = torch.stack(
            [torch.cat([u, u.new_zeros(max_seq_len - u.size(0), u.size(1))]) for u in split_hidden_states]
        )

        # (B,max_seq_len)
        encoder_attention_mask = torch.stack(
            [torch.cat([u, u.new_zeros(max_seq_len - u.size(0))]) for u in attn_mask_list]
        )

        return prompt_embeds, encoder_attention_mask

    def __call__(self, prompt,generator=None,device=None):
        device = "cuda"

        prompt_embeds, prompt_embeds_mask = self.encode_prompt(
            prompt=prompt,
            device=device,
        )
        txt_seq_lens = prompt_embeds_mask.sum(dim=1).tolist()

        # (1,16,128,128) -> (1,16,64,2,64,2) -> (1,64x64,16,2,2) -> (1,4096,64) 
        latent_shape = (1, 4096, 64)
        latents = torch.randn(latent_shape,generator=generator,dtype=prompt_embeds.dtype).to(device)
        # (1,1,(1, 64, 64))
        img_shapes = [[(1, 64, 64)]]

        self.scheduler.set_timesteps(20, device=device, mu=0.69)
        timesteps = self.scheduler.timesteps

        for timestep in timesteps:
            print("timestep:",timestep)
            with torch.no_grad():
                timestep_tensor = torch.tensor([timestep],dtype=prompt_embeds.dtype).to(device)

                noise_pred = self.transformer(
                    hidden_states=latents,
                    timestep=timestep_tensor / 1000,
                    guidance=None,
                    encoder_hidden_states=prompt_embeds,
                    img_shapes=img_shapes,
                    txt_seq_lens=txt_seq_lens,
                ).sample

                latents = self.scheduler.step(noise_pred, timestep, latents)[0]

        # (1,4096,64) -> (1,64,64,16,2,2) -> (1,16,64,2,64,2)
        latents = latents.view(1, 64, 64, 16, 2, 2).permute(0, 3, 1, 4, 2, 5)
        # (1,16,64,2,64,2) -> (1,16,1,128,128)
        latents = latents.reshape(1, 16, 1, 128, 128)

        # (1,16,1,1,1)
        latents_mean = torch.tensor(self.vae.config.latents_mean).view(1, self.vae.config.z_dim, 1, 1, 1).to(
            latents.device, latents.dtype
        )
        # (1,16,1,1,1)
        latents_std = torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
            latents.device, latents.dtype
        )
        # # (1,16,1,128,128)
        latents = latents * latents_std + latents_mean

        image = self.vae.decode(latents).sample[:, :, 0]
        image = (image / 2 + 0.5).clamp(0, 1)

        return image

if __name__ == "__main__":
    device = "cuda"

    vae = VAE.from_pretrained(
        "Qwen/Qwen-Image",
        subfolder="vae",
        torch_dtype=torch.bfloat16,
    )
    
    text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen-Image",
        subfolder="text_encoder",
        dtype=torch.bfloat16,
    )

    tokenizer = Qwen2Tokenizer.from_pretrained(
        "Qwen/Qwen-Image",
        subfolder="tokenizer",
    )

    transformer = Transformer.from_pretrained(
        "Qwen/Qwen-Image",
        subfolder="transformer",
        torch_dtype=torch.bfloat16,
    )

    scheduler = Scheduler()

    pipeline = Pipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        transformer=transformer,
        scheduler=scheduler,
    )
    pipeline.to(device)

    generator = torch.Generator().manual_seed(20)
    prompt = "a photo of a cat"
    output = pipeline(prompt,generator=generator,device=device)
    save_image(output[0], "output.png")