import os
import torch
import torch.nn as nn
from torchinfo import summary
from torchview import draw_graph
import matplotlib.pyplot as plt

def model_graph(network, expand_nested, graph_name, example = (1, 3, 640, 640)):
    """
    Visualize the network architecture and save it to an image file, then display it.

    Args:
        network: An object containing the model to visualize. Expected to have a .model attribute (e.g. YOLOv8 or custom wrapper).
        expand_nested: include internal children/modules in the graph
        example: dummy input tensor to trace the network graph, default: (1, 3, 640, 640) — typical for YOLO-type models.
    """    

    # Generate the graph representation of the model
    graph = draw_graph(
        network.model,
        input_size= example,
        expand_nested= expand_nested,
        roll=False,
        graph_name= graph_name,
    )

    # Render and save the visual graph to a PNG file
    # "cleanup=True" removes temporary files generated during the rendering process
    graph.visual_graph.render(graph_name, format="png", cleanup=True)
    saved_dir = graph_name + ".png"
    print("Model's graph is saved in:", saved_dir)

    # Load the saved image file
    img = plt.imread(saved_dir)
    # Display the image
    plt.imshow(img)
    # Hide axis ticks and borders for a cleaner view
    plt.axis('off')
    plt.show()
