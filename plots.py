import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np


## bar chart data skew

x = np.arange(2)
dataset = [242, 148]




fig, ax = plt.subplots()
plt.bar(x, dataset, color=('orange','red'))
plt.xticks(x, ('Bacterial Pneumonia', 'Viral Pneumonia'))
plt.show()