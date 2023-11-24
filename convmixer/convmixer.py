import torch.nn as nn
from einops import rearrange

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x

# def ConvMixer(dim, depth, kernel_size=9, patch_size=7, n_classes=1000):
#     return nn.Sequential(
#         nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size),
#         nn.GELU(),
#         nn.BatchNorm2d(dim),
#         *[nn.Sequential(
#                 Residual(nn.Sequential(
#                     nn.Conv2d(dim, dim, kernel_size, groups=dim, padding="same"),
#                     nn.GELU(),
#                     nn.BatchNorm2d(dim)
#                 )),
#                 nn.Conv2d(dim, dim, kernel_size=1),
#                 nn.GELU(),
#                 nn.BatchNorm2d(dim)
#         ) for i in range(depth)],
#         nn.AdaptiveAvgPool2d((1,1)),
#         nn.Flatten(),
#         nn.Linear(dim, n_classes)
#     )


class ConvMixer(nn.Module):
    def __init__(self, dim, depth, kernel_size=9, patch_size=7, num_classes=1000, in_chans=3, activation=nn.GELU, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = dim
        self.head = nn.Linear(dim, num_classes) if num_classes > 0 else nn.Identity()
        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, dim, kernel_size=patch_size, stride=patch_size),
            activation(),
            nn.BatchNorm2d(dim)
        )
        self.blocks = nn.Sequential(
            *[nn.Sequential(
                    Residual(nn.Sequential(
                        nn.Conv2d(dim, dim, kernel_size, groups=dim, padding="same"),
                        activation(),
                        nn.BatchNorm2d(dim)
                    )),
                    nn.Conv2d(dim, dim, kernel_size=1),
                    activation(),
                    nn.BatchNorm2d(dim)
            ) for i in range(depth)]
        )
        self.pooling = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
          
    def forward_features(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.pooling(x)
        return x
    
    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)

        return x


class ConvMixer2(nn.Module):
    def __init__(self, dim, depth, kernel_size=9, patch_size=7, num_classes=1000, in_chans=3, activation=nn.GELU, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = dim
        self.head = nn.Linear(dim, num_classes) if num_classes > 0 else nn.Identity()
        # self.stem = nn.Sequential(
        #     nn.Conv2d(in_chans, dim, kernel_size=patch_size, stride=patch_size),
        #     activation(),
        #     nn.BatchNorm2d(dim)
        # )
        self.blocks = nn.Sequential(
            *[nn.Sequential(
                    Residual(nn.Sequential(
                        nn.Conv2d(dim, dim, kernel_size, groups=dim, padding="same"),
                        activation(),
                        nn.BatchNorm2d(dim)
                    )),
                    nn.Conv2d(dim, dim, kernel_size=1),
                    activation(),
                    nn.BatchNorm2d(dim)
            ) for i in range(depth)]
        )
        self.pooling = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
          
    def forward_features(self, x):
        # x = self.stem(x)
        # without patch embedding
        x = rearrange(x, 'b c (h p1) (w p2) -> b (p1 p2 c) h w', p1=4, p2=4)
        x = self.blocks(x)
        x = self.pooling(x)
        return x
    
    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)

        return x
