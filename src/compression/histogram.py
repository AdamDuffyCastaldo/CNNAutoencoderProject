import numpy as np
import matplotlib.pyplot as plt

def createnormalisedhistogram(imgarray):
    histogram, bin_edges = np.histogram(imgarray, bins = 256, range = (0, 256))
    histogram = histogram / histogram.sum()
    fig, ax = plt.subplots()
    ax.set_title("histogram of pixel values in image")
    ax.set_xlabel("grayscale value")
    ax.set_ylabel("pixel count")
    ax.plot(bin_edges[0:-1], histogram)
    return histogram 