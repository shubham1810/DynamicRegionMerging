import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def visualize_rag(img, labels, graph):
    """
    Function to visualize RAG over the actual image and
    the segmented image.
    """
    # Create the plot placeholder
    plt.figure(0, figsize=(20, 10))
    
    plt.subplot(1, 2, 1)
    # Plot the image first
    plt.imshow(img)

    # Iterate over all regions/nodes
    for ix in range(1, max(graph.edge_data)+1):
        curr = ix
        graph.edge_data
        for iy in graph.edge_data.get(ix, []):
            # Don't repeat lines for plots
            if iy > ix:
                region1_center = graph.nodes[ix][:, -2:].mean(0)
                region2_center = graph.nodes[iy][:, -2:].mean(0)

                # Line width related to the weight of the edge
                lw = graph.edges[(ix, iy)]

                x_coords = [region1_center[0], region2_center[0]]
                y_coords = [region1_center[1], region2_center[1]]
                plt.plot(y_coords, x_coords, 'k-o', linewidth=1*lw)
    
    # Same process for the segmented image. Shows better graphs
    plt.subplot(1, 2, 2)
    plt.imshow(labels)
    for ix in range(1, max(graph.edge_data)+1):
        curr = ix
        for iy in graph.edge_data.get(ix, []):
            if iy > ix:
                region1_center = graph.nodes[ix][:, -2:].mean(0)
                region2_center = graph.nodes[iy][:, -2:].mean(0)

                lw = graph.edges[(ix, iy)]

                # x_coords = [400-region1_center[0], 400-region2_center[0]]
                x_coords = [region1_center[0], region2_center[0]]
                y_coords = [region1_center[1], region2_center[1]]
                plt.plot(y_coords, x_coords, 'k-o', linewidth=1*lw)
    
    plt.show()
    
def quantize_image(image, labels):
    """
    Get a pseudo-quantized version of the image
    """
    new_img = np.zeros_like(image)
    for lx in np.unique(labels):
        mask = (labels == lx)[:, :, None]
        vals = mask*image
        col = vals.reshape((-1, 3)).sum(0) / mask.reshape((-1,)).sum(0)
        new_img += (mask*col).astype(np.uint8)

    return new_img