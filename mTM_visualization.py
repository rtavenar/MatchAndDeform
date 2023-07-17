import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import medfilt
import os

def load_a_dataset(data_path, domain_id):
    fname_base = f"miniTimeMatch_{domain_names.index(domain_id)}test.npy"
    X = np.load(os.path.join(data_path, fname_base))
    y = np.load(os.path.join(data_path, fname_base.replace(".npy", "_labels.npy")))
    return X, y

data_path = "./Dataset/miniTimeMatch"
#              corn              meadow   barley    rape
color_index = ['orange', 'plum', 'green', 'red', 'pink', 'lightgreen']
class_name = ['corn',
 'horsebeans',
 'meadow',
 'spring_barley',
 'winter_barley',
 'winter_rapeseed',
 'winter_triticale',
 'winter_wheat'
 ]

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Palatino"],
})

domain_names = ["FR1", "FR2", "DK1", "AT1"]
class_indices = [0, 1, 2, 3]
pairs = [
    ("DK1", "FR1"),
    # ("DK1", "FR2"),
    # ("DK1", "AT1"),
    # ("FR1", "DK1"),
    # ("FR1", "FR2")
]
epsilon = 30

fig, axes = plt.subplots(len(pairs), len(class_indices), figsize=(16, 3))

for ix in range(len(pairs)):
    source_series, source_labels = load_a_dataset(
        data_path=data_path, 
        domain_id=pairs[ix][0]
    )[:2]
        
    target_series, target_labels = load_a_dataset(
        data_path=data_path, 
        domain_id=pairs[ix][1]
    )[:2]
    
    for iy in class_indices:
        axes[iy].set_xticks([])
        axes[iy].set_yticks([])
        axes[iy].set_xlim([0, 61])
        axes[iy].set_ylim([0., 0.8])
        
        # if ix == 0:
        #     axes[ix, iy].set_title(f"{pairs[ix][0]} $\\rightarrow$ {pairs[ix][1]}", size=20)
        if ix == 0:
            axes[iy].set_title(class_name[iy].replace("_", " ").capitalize(), 
                                    size=30)
        
        for cur_series, cur_labels, cur_alpha, label in zip([source_series, target_series],
                                                     [source_labels, target_labels],
                                                     [1., 0.4],
                                                     ["Average source NDVI series", "Average target NDVI series"]):
            cur_series = medfilt(cur_series[cur_labels == iy], (1, 5, 1))
            denominator = cur_series[:, :, 6] + cur_series[:, :, 2]
            numerator = cur_series[:, :, 6] - cur_series[:, :, 2]
    
            
            average_ndvi = np.nanmean(numerator / denominator, axis=0)
            axes[iy].plot(average_ndvi, color=color_index[iy], linewidth=2, alpha=cur_alpha, label=label)

handles, labels = axes[-1].get_legend_handles_labels()
plt.tight_layout()
plt.figlegend(handles, labels, loc="upper center", bbox_to_anchor=(.5, 0.02), ncol=2, fontsize=30)
plt.savefig("ndvi_profiles.pdf", bbox_inches="tight")