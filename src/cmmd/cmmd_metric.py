from transformers import CLIPVisionModelWithProjection
import torch
from torch import nn
from torchvision import transforms

from cmmd import utils as cu


class CMMD(nn.Module):
    """
    Compute CMMD metric to compare sets of real and generated images.
    Based on: https://github.com/sayakpaul/cmmd-pytorch
    """
    def __init__(self, device, range_converter=cu.range_2_1, reset_real_embedding=True, clip_mode_name="openai/clip-vit-large-patch14-336", scale=1000):
        """
        :param device: CPU or GPU device.
        :param range_converter: a function to convert image to the input range [0, 1]
        :param reset_real_embedding: Whether to also reset the real embeddings. Since in many cases the real dataset
         does not change, the features can be cached them to avoid recomputing them which is costly.
         Set this to ``False`` if your dataset does not change.
        :param clip_mode_name: CLIP model to use.
        """
        super().__init__()
        self._device = device
        self._range_converter = range_converter
        self._reset_real_embedding = reset_real_embedding
        self._scale = scale
        self._normalize = transforms.Compose([transforms.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711))])

        self._clip_model = CLIPVisionModelWithProjection.from_pretrained(clip_mode_name,
                                                                         torch_dtype=torch.float16,
                                                                         attn_implementation="sdpa"
                                                                         ).to(self._device)
        self._clip_model.eval()
        self._clip_model.requires_grad_(False)

        self.register_buffer("real_images", torch.tensor([], device=device))
        self.register_buffer("generated_images", torch.tensor([], device=device))
        self.register_buffer("real_embs", torch.tensor([], device='cpu'))
        self.register_buffer("generated_embs", torch.tensor([], device='cpu'))

        self._input_image_size = (336, 336)
        self._min_input_batch_size = 16
        self._sigma = 10

    @staticmethod
    def _resize_bicubic(images, size):
        return torch.nn.functional.interpolate(images, size=size, mode="bicubic", antialias=True)

    @torch.no_grad()
    def _get_clip_embeddings(self, images):
        """Computes CLIP embeddings for the given images.
        Args:
            images: An image tensor of shape (batch_size, 3, height, width). Values are
            in range [-1, 1].
        Returns:
          Embedding array of shape (batch_size, embedding_width).
        """
        images = images.to(self._device)
        images = self._resize_bicubic(images, size=self._input_image_size)
        if self._range_converter:
            images = self._range_converter(images)
        images = self._normalize(images)

        image_embs = self._clip_model(pixel_values=images).image_embeds.cpu()
        image_embs /= torch.linalg.norm(image_embs, axis=-1, keepdims=True)
        del images

        return image_embs

    def num_real(self):
        """ Number of real embeddings. """
        return self.real_embs.size(0)

    def reset(self):
        """ Reset accumulated embeddings. """
        if self._reset_real_embedding:
            self.real_embs = torch.tensor([], device='cpu')
        self.generated_embs = torch.tensor([], device='cpu')

    @torch.no_grad()
    def update(self, images, kind):
        """
        Accumulate embeddings.
        :param images: tensor of shape (batch_size, 3, height, width). Values are in range [-1, 1].
        :param kind: str, 'real' or 'fake'.
        :return:
        """
        if images.ndim != 4:
            raise ValueError(f"Expected 4D Real tensor, got {images.ndim}D tensor")

        if images.size(0) == 0:
            return
        images = images.to(self._device)
        if kind == 'real':
            self.real_images = torch.cat((self.real_images, images))
            # Compute embeddings in fixed batches to avoid latency issues.
            if self.real_images.size(0) >= self._min_input_batch_size:
                real_emb = self._get_clip_embeddings(self.real_images)
                self.real_embs = torch.cat((self.real_embs, real_emb))
                del real_emb
                self.real_images = torch.tensor([], device=self._device)
        if kind == 'fake':
            self.generated_images = torch.cat((self.generated_images, images))
            # Compute embeddings in fixed batches to avoid latency issues.
            if self.generated_images.size(0) >= self._min_input_batch_size:
                gen_emb = self._get_clip_embeddings(self.generated_images)
                self.generated_embs = torch.cat((self.generated_embs, gen_emb))
                del gen_emb
                self.generated_images = torch.tensor([], device=self._device)

    @torch.no_grad()
    def _update_embed_for_last_images(self):
        """
        Accumulate embeddings for the last images.
        """
        if self.real_images.size(0) > 0:
            real_emb = self._get_clip_embeddings(self.real_images)
            self.real_embs = torch.cat((self.real_embs, real_emb))
            del real_emb
            self.real_images = torch.tensor([], device=self._device)
        if self.generated_images.size(0) > 0:
            gen_emb = self._get_clip_embeddings(self.generated_images)
            self.generated_embs = torch.cat((self.generated_embs, gen_emb))
            del gen_emb
            self.generated_images = torch.tensor([], device=self._device)

    @torch.no_grad()
    def compute(self):
        """ Compute the MMD distance between two sets of CLIP embeddings.
        Args:
            x: The first set of embeddings of shape (n, embedding_dim).
            y: The second set of embeddings of shape (n, embedding_dim).

        Returns:
            The CMMD score for the two sets of embeddings.
        """
        # Accumulate embeddings for the last leftover images.
        self._update_embed_for_last_images()

        x = self.real_embs
        y = self.generated_embs

        x_sqnorms = torch.diag(torch.matmul(x, x.T))
        y_sqnorms = torch.diag(torch.matmul(y, y.T))

        gamma = 1 / (2 * self._sigma ** 2)
        k_xx = torch.mean(
            torch.exp(
                -gamma * (-2 * torch.matmul(x, x.T) + torch.unsqueeze(x_sqnorms, 1) + torch.unsqueeze(x_sqnorms, 0)))
        )
        k_xy = torch.mean(
            torch.exp(
                -gamma * (-2 * torch.matmul(x, y.T) + torch.unsqueeze(x_sqnorms, 1) + torch.unsqueeze(y_sqnorms, 0)))
        )
        k_yy = torch.mean(
            torch.exp(
                -gamma * (-2 * torch.matmul(y, y.T) + torch.unsqueeze(y_sqnorms, 1) + torch.unsqueeze(y_sqnorms, 0)))
        )
        del x, y, x_sqnorms, y_sqnorms

        return self._scale * (k_xx + k_yy - 2 * k_xy)
