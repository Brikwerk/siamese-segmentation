import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F


class CoSegNet(nn.Module):
    def __init__(self, input_channels=3, output_channels=2):
        super(CoSegNet, self).__init__()

        self.input_channels = input_channels # 3 = RGB
        self.output_channels = output_channels # 2 = Foreground + Background

        # Encoder
        # Using pretrained VGG16 as the backbone
        self.encoder = models.vgg16_bn(pretrained=True).features

        # Decoder
        self.decoder3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.ConvTranspose2d(512, 512, kernel_size=3, padding=1),
            nn.ConvTranspose2d(512, 512, kernel_size=3, padding=1),
            nn.ConvTranspose2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        self.decoder6 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.ConvTranspose2d(512, 512, kernel_size=3, padding=1),
            nn.ConvTranspose2d(512, 512, kernel_size=3, padding=1),
            nn.ConvTranspose2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        self.decoder9 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.ConvTranspose2d(512, 256, kernel_size=3, padding=1),
            nn.ConvTranspose2d(256, 256, kernel_size=3, padding=1),
            nn.ConvTranspose2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.decoder11 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.ConvTranspose2d(256, 128, kernel_size=3, padding=1),
            nn.ConvTranspose2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.decoder13 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.ConvTranspose2d(128, 64, kernel_size=3, padding=1),
            nn.ConvTranspose2d(64, 1, kernel_size=3, padding=1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        # Metric Net
        self.metricnet = SiameseMetricNet()

        # Decision Net
        self.decisionnet = SiameseDecisionNet()


    def forward(self, imageA, imageB):
        # Getting VGG features for the first stage of the encoding
        featuresA = self.encoder(imageA)
        featuresB = self.encoder(imageB)

        # Decoding
        # 3 layers
        featuresA = self.decoder3(featuresA)
        featuresB = self.decoder3(featuresB)
        # 6 layers
        featuresA = self.decoder6(featuresA)
        featuresB = self.decoder6(featuresB)
        # 9 layers
        featuresA = self.decoder9(featuresA)
        featuresB = self.decoder9(featuresB)

        # Siamese Metric Net
        metric_featureA = self.metricnet(featuresA)
        metric_featureB = self.metricnet(featuresB)
        
        # Siamese Decision Net
        # Concatenating the two vectors to make a single prediction vector
        decision_vector = torch.cat((metric_featureA, metric_featureB), dim=0)
        decision = self.decisionnet(decision_vector)

        # 11 layers
        featuresA = self.decoder11(featuresA)
        featuresB = self.decoder11(featuresB)
        # 13 layers
        featuresA = self.decoder13(featuresA)
        featuresB = self.decoder13(featuresB)

        return featuresA, featuresB, metric_featureA, metric_featureB, decision


class SiameseMetricNet(nn.Module):
    def __init__(self):
        super(SiameseMetricNet, self).__init__()
        self.metricnet = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )


    def forward(self, feature):
        # Global Average Pooling (GAP)
        gap_feature_vector = F.avg_pool2d(feature, kernel_size=feature.size()[2:])
        gap_feature_vector = gap_feature_vector.squeeze()

        return self.metricnet(gap_feature_vector)


class SiameseDecisionNet(nn.Module):
    def __init__(self):
        super(SiameseDecisionNet, self).__init__()
        self.decisionnet = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, vector):
        return self.decisionnet(vector)
