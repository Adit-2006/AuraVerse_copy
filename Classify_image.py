from PIL import Image                     # Used to open and handle image files
import torch                              # PyTorch main library (tensors, model execution)
import torchvision.transforms as transforms  # Preprocessing utilities
from torchvision import models            # Pretrained deep learning models


def ClassifyImages(image_path) -> str:
    """
    This function takes an image file path as input and returns the predicted label.
    """

    # 1. Load a pretrained neural network (ResNet-50) trained on ImageNet (1000 classes)
    # weights=models.ResNet50_Weights.DEFAULT loads the most recent recommended pretrained weights
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

    # Put the model in evaluation mode (turns off dropout, batchnorm training behavior)
    model.eval()

    # 2. Define a preprocessing pipeline for the input image
    # Images must be resized, converted to tensors, and normalized to match training conditions
    transform = transforms.Compose([
        transforms.Resize((224, 224)),   # ResNet expects input size 224x224
        transforms.ToTensor(),           # Convert image from PIL to PyTorch tensor
        transforms.Normalize(            # Normalize using ImageNet mean and std (training standard)
            mean=[0.485, 0.456, 0.406],  # Channel-wise means (R, G, B)
            std=[0.229, 0.224, 0.225]    # Channel-wise standard deviations
        )
    ])

    # 3. Load the image from disk and ensure it's in RGB mode
    img = Image.open(image_path).convert("RGB")

    # Apply preprocessing transformation and add batch dimension (model requires 4D: batch x channel x height x width)
    img_t = transform(img).unsqueeze(0)

    # 4. Perform prediction without calculating gradients (saves memory and computation)
    with torch.no_grad():
        output = model(img_t)   # Forward pass → returns raw class scores (logits)

    # 5. Load the list of class labels (ImageNet categories) from the model metadata
    labels = models.ResNet50_Weights.DEFAULT.meta["categories"]

    # 6. Get the index of the highest-scoring prediction (argmax)
    _, predicted_index = output.max(1)

    # Convert index to actual label string
    predicted_label = labels[predicted_index]

    return predicted_label
