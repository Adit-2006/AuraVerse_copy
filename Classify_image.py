from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision import models


def ClassifyImages(image_path) -> str:

    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    img = Image.open(image_path).convert("RGB")

    img_t = transform(img).unsqueeze(0)

    with torch.no_grad():
        output = model(img_t)

    labels = models.ResNet50_Weights.DEFAULT.meta["categories"]

    _, predicted_index = output.max(1)

    predicted_label = labels[predicted_index]

    return predicted_label
