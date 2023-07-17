import torch.nn as nn
import os
import numpy as np
import torch
import warnings

from dataset_loader import load_a_dataset
from General_model import Basic_CNN, CNN_DeepJDOT, CNNMAD
from MAD import MAD



# Load data
X_source, y_source, X_valid_source, y_valid_source, X_test_source, y_test_source = load_a_dataset(dataset_name="Dataset/TinyTimeMatch/TinyTimeMatch", domain_id=3)  # 3: Denmark
X_target, y_target, X_valid_target, y_valid_target, X_test_target, y_test_target = load_a_dataset(dataset_name="Dataset/TinyTimeMatch/TinyTimeMatch", domain_id=4)  # 4: Austria

n_classes = len(set(y_source.tolist()))
n_features = X_source.shape[-1]

feature_extractor = nn.Sequential(
            nn.Conv1d(in_channels=n_features, out_channels=128, kernel_size=8, stride=1, padding="same", bias=False),
            nn.BatchNorm1d(num_features=128, affine=False),
            nn.ReLU(),

            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=5, stride=1, padding="same", bias=False),
            nn.BatchNorm1d(num_features=256, affine=False),
            nn.ReLU(),

            nn.Conv1d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding="same", bias=False),
            nn.BatchNorm1d(num_features=128, affine=False),
            nn.ReLU())
classifier = nn.Sequential(nn.Linear(128, n_classes))


model_MAD_Deep = CNNMAD(batchsize=128, 
                 feature_extractor=feature_extractor, 
                 classifier=classifier,
                 alpha=0.1, 
                 beta=0.01, 
                 MAD_class=True, #To be set to True if we want C-MAD or to False if we want MAD
                 lr=1e-4,
                 X_target=X_target,
                 max_iterations=2)
model_MAD_Deep_avec_sauvegardes = CNNMAD(batchsize=128, 
                                         feature_extractor=feature_extractor, 
                                         classifier=classifier,
                                         alpha=0.1, 
                                         beta=0.01, 
                                         MAD_class=True, #To be set to True if we want C-MAD or to False if we want MAD
                                         lr=1e-4,
                                         X_target=X_target,
                                         save_DTW_matrices=True,
                                         save_OT_plan=True,
                                         max_iterations=2)
model_MAD = MAD()

for model in [model_MAD_Deep, model_MAD_Deep_avec_sauvegardes, model_MAD]:
    model.fit(X_source, y_source)
    model.fit(X_source, y_source, X_valid_source, y_valid_source)
    model.predict(X_test_target)
    print(model.evaluate(X_test_target, y_test_target))
    if model.save_DTW_matrices:
        print(len(model.history_DTW_matrices), model.history_DTW_matrices[0][0].shape)
    if model.save_OT_plan:
        print(len(model.history_OT_plan), model.history_OT_plan[0].shape)