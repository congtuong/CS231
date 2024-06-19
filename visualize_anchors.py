import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def visualize_anchors(anchors, image_size):
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(np.ones((image_size, image_size, 3)))
    for anchor in anchors:
        w, h = anchor
        rect = Rectangle((0-w/2, 0-h/2), w, h, edgecolor='r', facecolor='none')

        ax.add_patch(rect)
        
    fig.savefig('anchors.png')

if __name__ == '__main__':
    with open('anchors.pkl', 'rb') as f:
        anchors = pickle.load(f)
    print(anchors)
    anchors = np.array(anchors)/32
    visualize_anchors(anchors, 1)