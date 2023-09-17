import numpy as np

def select_kpi(event_hit_data, threshold, output_path):
    selected_kpis = np.argwhere(np.sum(event_hit_data, axis=1) >= threshold)
    # print(selected_kpis)
    np.savetxt(output_path, selected_kpis, fmt="%d")
    return selected_kpis

def select_kpi_v2(event_hit_data, threshold):
    selected_kpis = np.argwhere(np.sum(event_hit_data, axis=1) >= threshold)
    return selected_kpis

def select_kpi_with_cluster(event_hit_data, kpi_threshold, cluster_threshold, cluster_label, output_path):
    selected_kpis = np.argwhere(np.sum(event_hit_data, axis=1) >= kpi_threshold).flatten()
    select_values, select_counts = np.unique(cluster_label[selected_kpis], return_counts=True)
    all_values, all_counts = np.unique(cluster_label, return_counts=True)

    selected_clusters = []
    for i in range(np.max(cluster_label)):
        if i in select_values:
            select_ratio = select_counts[np.argwhere(select_values == i)[0][0]] / all_counts[np.argwhere(all_values == i)[0][0]]
            print(i, select_ratio)
            if select_ratio > cluster_threshold:
                selected_clusters.append(i)
    print("selected clusters", selected_clusters)
    new_selected_kpis = np.union1d(np.argwhere(np.isin(cluster_label, selected_clusters)), selected_kpis)
    print(new_selected_kpis, len(new_selected_kpis) > len(selected_kpis))
    np.savetxt(output_path, new_selected_kpis, fmt="%d")
    return new_selected_kpis

def select_kpi_with_cluster_v2(event_hit_data, kpi_threshold, cluster_threshold, cluster_label, output_path):
    selected_kpis = np.argwhere(np.sum(event_hit_data, axis=1) >= kpi_threshold).flatten()
    select_values, select_counts = np.unique(cluster_label[selected_kpis], return_counts=True)
    all_values, all_counts = np.unique(cluster_label, return_counts=True)

    selected_clusters = []
    for i in range(np.max(cluster_label)):
        if i in select_values:
            select_ratio = select_counts[np.argwhere(select_values == i)[0][0]] / all_counts[np.argwhere(all_values == i)[0][0]]
            print(i, select_ratio)
            if select_ratio > cluster_threshold:
                selected_clusters.append(i)
    print("selected clusters", selected_clusters)
    new_selected_kpis = np.union1d(np.argwhere(np.isin(cluster_label, selected_clusters)), selected_kpis)
    return new_selected_kpis

def validate_select(event_hit_data, selected_kpis):
    return np.sum(event_hit_data[selected_kpis], axis=0)

def validate_select_cover(event_hit_data, selected_kpis):
    origin_num = np.sum(np.sum(event_hit_data, axis=0) > 0)
    selected_num = np.sum(np.sum(event_hit_data[selected_kpis], axis=0) > 0)
    return selected_num / origin_num
