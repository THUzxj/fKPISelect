import os
import numpy as np
from sklearn.cluster import AgglomerativeClustering

from anomaly_detection.solvers import AnomalyTransformerSolver


def generate_feature_latents(config, solver):
    # solver = AnomalyTransformerSolver(vars(config))
    solver.load_model()
    model = solver.model

    latents = []
    for i in range(533):
        config.dataset = f"CTF-{i}"
        solver = AnomalyTransformerSolver(vars(config))
        solver.model = model
        feature_latent = solver.feature_latent()
        latents.append(feature_latent)
        print(i, feature_latent.shape,)

    latents = np.array(latents)
    print(latents)
    np.save("output/feature_latents.npy", latents)
    return latents

"""
Cluster latents with cosine similarity
"""
def cluster_on_feature_latents():
    latents = np.load("output/feature_latents.npy")
    print(latents.shape)
    clustering = AgglomerativeClustering(n_clusters=10, affinity='cosine', linkage='average').fit(latents)
    print(clustering.labels_)

    output_path = "output/features.png"
    plot_latents(latents, clustering.labels_, output_path)

    clustering_labels = np.array(clustering.labels_)    
    os.makedirs("output/clusters", exist_ok=True)    
    for i in range(10):
        f = open(os.path.join("output/clusters", f"cluster_{i}.txt"), "w")
        for number in np.argwhere(clustering_labels == i):
            f.write(f"{int(number)}\n")
