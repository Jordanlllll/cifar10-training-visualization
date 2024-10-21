import torch
import torch.nn as nn
from torchviz import make_dot
from model import EfficientNetV2

def visualize_model_architecture(model, input_size=(1, 3, 32, 32)):
    x = torch.randn(input_size)
    y = model(x)
    graph = make_dot(y, params=dict(model.named_parameters()))
    graph.render("model_architecture", format="png")

if __name__ == "__main__":
    model = EfficientNetV2()
    visualize_model_architecture(model)
