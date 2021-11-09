import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report
# from kmeans import kmeans, kmeans_predict
# import torch
import os
from datetime import datetime
import pickle

def load_data(f: str):
    data = np.load(f)
    return data['arr_0'], data['arr_1']


# Generate id to reference files/plots/images
dt_id = datetime.now().isoformat()
print(f'ID: {dt_id}')

RESULT_DIR = f'results/kmeans/{dt_id}'
os.mkdir(RESULT_DIR)
print('Result dir:', RESULT_DIR)

print('Load data...')
X_train, y_train = load_data('/home/bethge/ahochlehnert48/results/imnet_train_latent_all.npz')
# ids = np.random.choice(range(X_train.shape[0]), 100000)

# scikit-learn
km = KMeans(n_clusters=np.unique(y_train).size, verbose=1, random_state=0)
pred_y = km.fit_predict(X_train)

# PyTorch
# X_train = torch.Tensor(X_train)
# cluster_ids_x, cluster_centers = kmeans(
#     X=X_train[ids], num_clusters=np.unique(y_train[ids]).size, distance='cosine', device=torch.device('cpu')
# )
print('Save clusters')
# torch.save(cluster_centers, f'{RESULT_DIR}/{dt_id}_cluster_centers.pt')
pickle.dump(km, open(f"{RESULT_DIR}/{dt_id}_kmeans.pkl", "wb"))
# clt = pickle.load(open("save.pkl", "rb"))


# print('Predict labels')
# pred_y = kmeans_predict(
#     X_train[ids], cluster_centers, distance='cosine', device=torch.device('cpu')
# )

# report = classification_report(y_train[ids], pred_y[ids])
report = classification_report(y_train, pred_y)
print(report)
with open(f'{RESULT_DIR}/{dt_id}_report.txt', 'w') as f:
    f.write(report)

