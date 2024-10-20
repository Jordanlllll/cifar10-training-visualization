import torch
import torch.nn as nn
from torchview import draw_graph
from model import EfficientNetV2

def visualize_model_architecture(model, input_size=(1, 3, 32, 32)):
    x = torch.randn(input_size)
    graph = draw_graph(model, input_size)
    graph.visual_graph.render("model_architecture", format="png")

if __name__ == "__main__":
    model = EfficientNetV2()
    visualize_model_architecture(model)
