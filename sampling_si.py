from scipy.stats import gaussian_kde
from scipy.spatial.distance import cdist

import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.stats import norm, gaussian_kde
from sklearn.model_selection import train_test_split
import pandas as pd

from data_preprocessing import * 

directory_path = '' #data_path
images, size_label, location_label, count_label = process_directory(directory_path)


sizes = []
x_coords = []
y_coords = []

for (size1, size2), (y1, x1, y2, x2) in zip(size_label, location_label):
    sizes.append(size1)
    x_coords.append(x1)
    y_coords.append(y1)
    if size2 != 0 or (x2 != 0 and y2 != 0):  
        sizes.append(size2)
        x_coords.append(x2)
        y_coords.append(y2)


data = {'size': sizes,'x_coord': x_coords,'y_coord': y_coords }
df = pd.DataFrame(data)
# If Normalize == True:
size_normalization_factor = args.image_size*args.image_size
coord_normalization_factor = args.image_size
df['size_normalized'] = df['size'] / size_normalization_factor
df['x_coord_normalized'] = df['x_coord'] / coord_normalization_factor
df['y_coord_normalized'] = df['y_coord'] / coord_normalization_factor
X = df[['y_coord_normalized', 'x_coord_normalized']].values
y = df['size_normalized'].values


kernel = C(1.0, (1e-4, 1e1)) * RBF(10, (1e-4, 1e1))
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=1e-2, normalize_y=False)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0, random_state=42)
gp.fit(X_train, y_train)


y_pred = gp.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)


# Sampling Location
kde = gaussian_kde(X.T) 

def sample_coords(n_samples):
    samples = kde.resample(n_samples).T
    return samples

# Sampling size with sampled location
def sample_size(new_coords, min_size=0, mae=mae):
    y_mean, y_std = gp.predict([new_coords], return_std=True)
    lower_bound = y_mean - mae
    upper_bound = y_mean + mae
    
    samples = np.random.uniform(lower_bound, upper_bound, 1)  
    samples = np.maximum(samples, min_size / size_normalization_factor)  
    return samples


def sample_from_distribution(n_samples):
    sampled_data = []
    coords_samples = sample_coords(n_samples)
    
    for coords in coords_samples:
        size_samples = sample_size(coords)
        
        x_denormalized = coords[1] * coord_normalization_factor
        y_denormalized = coords[0] * coord_normalization_factor
        size_denormalized = size_samples * size_normalization_factor
        
        sampled_data.append((size_denormalized[0],y_denormalized,  x_denormalized))
    return sampled_data


gaussian_aug_conds = []

sampled_data = sample_from_distribution()
for i in range(len(sampled_data)):
    gaussian_aug_conds.append([sampled_data[i][0], sampled_data[i][1], sampled_data[i][2], 1])

gaussian_aug_conds = np.array(gaussian_aug_conds)
gaussian_aug_conds = torch.tensor(gaussian_aug_conds, dtype=torch.float32)

# count >= 2
def sample_coords(n_samples):
    samples = kde.resample(n_samples).T
    while (samples <= 0.1).any() or (samples >= 0.90).any():
        samples = kde.resample(n_samples).T
    return samples

def distance(coord1, coord2):
    return np.sqrt((coord1[0] - coord2[0]) ** 2 + (coord1[1] - coord2[1]) ** 2)

def sample_from_distribution(n_samples):
    sampled_data = []
    coords_samples1 = sample_coords(n_samples)
    coords_samples2 = sample_coords(n_samples)
    for coords1, coords2 in zip(coords_samples1, coords_samples2):
        size_samples1 = sample_size(coords1)
        size_samples2 = sample_size(coords2)

        x_denormalized1 = coords1[1] * coord_normalization_factor
        y_denormalized1 = coords1[0] * coord_normalization_factor
        size_denormalized1 = size_samples1 * size_normalization_factor

        x_denormalized2 = coords2[1] * coord_normalization_factor
        y_denormalized2 = coords2[0] * coord_normalization_factor
        size_denormalized2 = size_samples2 * size_normalization_factor

        while distance((y_denormalized1, x_denormalized1), (y_denormalized2, x_denormalized2)) <= (np.sqrt(size_denormalized1[0]) + np.sqrt(size_denormalized2[0])):
            coords2 = sample_coords(1)[0]
            size_samples2 = sample_size(coords2)
            x_denormalized2 = coords2[1] * coord_normalization_factor
            y_denormalized2 = coords2[0] * coord_normalization_factor
            size_denormalized2 = size_samples2 * size_normalization_factor
        sampled_data.append((size_denormalized1[0], size_denormalized2[0], y_denormalized1, x_denormalized1, y_denormalized2, x_denormalized2))
    return sampled_data

gaussian_aug_conds2 = []

sampled_data = sample_from_distribution()
for i in range(len(sampled_data)):
    gaussian_aug_conds2.append([sampled_data[i][0], sampled_data[i][1], sampled_data[i][2], sampled_data[i][3],sampled_data[i][4]/args.image_size, sampled_data[i][5]/args.image_size, 0, 1])

gaussian_aug_conds2 = np.array(gaussian_aug_conds2)
gaussian_aug_conds2 = torch.tensor(gaussian_aug_conds2, dtype=torch.float32)
