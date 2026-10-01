import torch
import torch.nn as nn


class GazeModel(nn.Module):
    def __init__(self):
        super().__init__()

        # Extract visual features from the image
        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )

        # Image features + head bbox (4) + eye point (2)
        self.regressor = nn.Sequential(
            nn.Linear(256 + 6, 128),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, 2),
            nn.Sigmoid(),
        )

    def forward(self, image, head_bbox, eye_point):
        image_features = self.image_encoder(image)

        metadata = torch.cat([head_bbox, eye_point], dim=1)

        combined = torch.cat([image_features, metadata], dim=1)

        gaze_prediction = self.regressor(combined)

        return gaze_prediction


if __name__ == "__main__":
    model = GazeModel()

    # Dummy batch for checking model input/output dimensions
    images = torch.randn(4, 3, 224, 224)
    head_bbox = torch.rand(4, 4)
    eye_point = torch.rand(4, 2)

    predictions = model(images, head_bbox, eye_point)

    print("Prediction shape:", predictions.shape)
    print("Predictions:", predictions)