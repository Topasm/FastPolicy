import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
# model/modules/modules.py 에 SpatialSoftmax가 있다고 가정
from model.modules.modules import SpatialSoftmax

# Import configuration from dedicated config file
from model.predictor.config import BidirectionalARTransformerConfig


class ImageEncoder(nn.Module):
    def __init__(self, config: BidirectionalARTransformerConfig):
        super().__init__()
        self.config = config
        self.do_crop = True
        self.center_crop = transforms.CenterCrop(
            (config.image_size, config.image_size))
        self.maybe_random_crop = transforms.RandomCrop(
            (config.image_size, config.image_size)) if config.crop_is_random else self.center_crop

        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])

        with torch.no_grad():
            dummy_input = torch.zeros(
                1, config.image_channels, config.image_size, config.image_size)
            feature_map_shape = self.backbone(dummy_input).shape[1:]

        self.pool = SpatialSoftmax(feature_map_shape, num_kp=32)
        self.out = nn.Linear(32 * 2, config.image_latent_dim)
        self.layer_norm = nn.LayerNorm(config.image_latent_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        images = self.maybe_random_crop(
            images) if self.training else self.center_crop(images)
        features = self.backbone(images)
        keypoints = self.pool(features)
        features_flat = torch.flatten(keypoints, start_dim=1)
        return self.layer_norm(self.out(features_flat))


class ImageDecoder(nn.Module):
    def __init__(self, config: BidirectionalARTransformerConfig):
        super().__init__()
        self.config = config
        self.initial_linear = nn.Sequential(
            nn.Linear(config.image_latent_dim, 512 * 3 * 3), nn.ReLU())
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 512, 4, 2, 1), nn.BatchNorm2d(
                512), nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1), nn.BatchNorm2d(
                256), nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.BatchNorm2d(
                128), nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.BatchNorm2d(
                64), nn.ReLU(True),
            nn.ConvTranspose2d(64, config.image_channels, 4, 2, 1), nn.Sigmoid())

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        x = self.initial_linear(latents).view(-1, 512, 3, 3)
        return self.decoder(x)
