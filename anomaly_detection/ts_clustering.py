import numpy as np
from sklearn.cluster import DBSCAN
import numpy as np
from sklearn.preprocessing import StandardScaler


def interpolation(df):
    df = df.resample('15s').ffill()
    df.fillna(method='ffill', inplace=True)
    df.fillna(method='bfill', inplace=True)
    return df


def standardize(tses):
    scaler = StandardScaler()
    scaler.fit(tses)
    return scaler.transform(tses)


def smoothing_extreme_values(values):
    """
    In general,the ratio of anomaly points in a time series is less than 5%[1].
    As such,simply remove the top5% data which deviate the most from the mean 
    value,and use linear interpolation to fill them.

    Args:
        values(np.ndarray) is a time series which has been preprosessed by linear 
        interpolation and standardization(to have zero mean and unit variance)

    Returns:
        np.ndarray: The smoothed `values`
    """

    values = np.asarray(values, np.float32)
    if len(values.shape) != 1:
        raise ValueError('`values` must be a 1-D array')
#    if (values.mean() != 0) or (values.std() != 1):
#        raise ValueError('`values` must be standardized to have zero mean and unit variance')

    # get the deviation of each point from zero mean
    values_deviation = np.abs(values)

    # the abnormal portion
    abnormal_portion = 0.05

    # replace the abnormal points with linear interpolation
    abnormal_max = np.max(values_deviation)
    abnormal_index = np.argwhere(
        values_deviation > abnormal_max * (1-abnormal_portion))
    abnormal = abnormal_index.reshape(len(abnormal_index))
    normal_index = np.argwhere(
        values_deviation <= abnormal_max * (1-abnormal_portion))
    normal = normal_index.reshape(len(normal_index))
    normal_values = values[normal]
    if len(abnormal) == 0:
        return values
    if len(normal) == 0:
        return values
    abnormal_values = np.interp(abnormal, normal, normal_values)
    values[abnormal] = abnormal_values

    return values


def extract_baseline(values, w):
    """
    A simple but effective method for removing noises if to apply moving 
    average with a small sliding window(`w`) on the KPI(`values`),separating 
    its curve into two parts:baseline and residuals.
    For a KPI,T,with a sliding window of length of `w`,stride = 1,for each 
    point x(t),the corresponding point on the baseline,denoted as x(t)*,is the 
    mean of vector (x(t-w+1),...,x(t)).
    Then the diffrence between x(t) and x(t)* is called a residuals.

    Args:
        values(np.ndarray): time series after preprocessing and smoothed

        w():

    Returns:
        tuple(np.ndarray,np.float32,np.float32):
            np.ndarray: the baseline of rawdata;
            np.float32: the mean of input values after moving average;
            np.float32: the std of input values after moving average.
        np.ndarray:the residuals between rawdata between baseline


    """
    # moving average to get the baseline
    baseline = np.convolve(values, np.ones((w,))/w, mode='valid')
    # get the residuals,the difference between raw series and baseline
    residuals = values[w-1:] - baseline

    return baseline, residuals


def sbd_ele(values1, values2):
    """
    Given two time seires `values1' and `values2`,cross-correlation slides 
    `values2' to `values1` to compute the inner-product for each shift `s`,the
    range of shift `s` ∈ [-len(values2) + 1, len(values1)-1]
    SBD is based of cross-correlation. SBD ranges from 0 to 2, where 0 means 
    two time series have exactly the same shape. A smaller SBD means higher 
    shape similarity.

    Args:
        values1(np.ndarray): time series 1
        values2(np.ndarray): time series 2

    Returns:
        np.float32: the SBD between `values1` and `values2`
    """
    # get the 2 norm
    l2_values1 = np.linalg.norm(values1)
    l2_values2 = np.linalg.norm(values2)
    # get the cross-correlation of each shift `s`
    cross_corre = np.convolve(values1, values2, mode='full')

    # return the SBD between `values1` and `values2`
    return max(0, (1 - np.max(cross_corre)/(l2_values1 * l2_values2)) if (l2_values1 * l2_values2) != 0 else 1)


def SBD(values_list, minPts=4):
    """
    Caculate the shape based distance(SBD) between any two time series for 
    similarity measure. SBD is used for DBSCAN for clustering.
    The main idea of DBSCAN is to find some cores in dense regions,and then 
    expand the cores by transitivity of similarity to form clusters.

    Args:
        List(np.ndarray): a list consists of all time series(np.ndarray),the 
            lengths of different time series could be different
        minPts(np.int32): The core `p` in DBSCAN is defined as an object that has
            at least `minPts` objects within a distance of ϵ from it(excluding
            `p`). The default value of `minPts` is 4.

    Returns:
        np.ndarray: for each time series, take the SBD between it and its 
        minPts-Nearest-Neighbor(KNN). The SBDs of all time series in `values`
        returned as an np.ndarray.
    """

    if len(values_list) < minPts:
        raise ValueError('`values_list` must contain more than %d time series'
                         % minPts)
    if len(values_list[0].shape) != 1:
        raise ValueError('`values` must be a 1-D array')
    if (type(minPts) is not int) or (minPts < 1):
        raise ValueError('`minPts` must be a positive integar')

    # Caculate the SBD between any two time time series
    sbd_matrix = np.zeros((len(values_list), len(values_list)))
    for i in range(len(values_list)):
        # print("calculating sbd", i)
        for j in range(i, len(values_list)):
            sbd_matrix[i][j] = sbd_ele(values_list[i], values_list[j])
            sbd_matrix[j][i] = sbd_matrix[i][j]

    # Return the minPts nearest SBD for each time series(excluding itself)
    ret_sbd = np.zeros(len(values_list))
    for i in range(len(values_list)):
        src_index = np.argsort(sbd_matrix[i])
        ret_sbd[i] = sbd_matrix[i][src_index][minPts]

    return sbd_matrix, ret_sbd


def density_radius(sbd_arr, len_thresh, max_radius, slope_thresh, slope_diff_thresh):
    """
    Given K-Nearest-Neighbor SBDs of each sample,calculate the density radius
    for DESCAN clustering.

    Args:
        `sbd_arr`: np.ndarray, array of the K-Nearest-Neighbor SBD of each 
            sample.
        `len_thresh`: np.int32, the length of traget SBDs for candidate radius
            search.
        `max_radius`: np.float32, candidate radius are no larger than 
            `max_radius`.
        `slope_thresh`: np.float32, the slopes on the left and right of 
            candidate point are no larger than `slope_thresh`
        `slope_diff_thresh`: np.float32, the diff between leftslope and right-
            slope of candidate point are no larger than `slope_diff_thresh`

    Returns:
        np.float32: the final density radius is the largest value of all 
            candidate radii.
    """
    src_index = np.argsort(sbd_arr)
    sbd_arr_sorted = sbd_arr[src_index][::-1]
    candidates_index = np.argwhere(sbd_arr_sorted <= max_radius)
    start = np.min(candidates_index)
    end = len(sbd_arr_sorted)

    def find_candidate_radius(sbd_arr_sorted, start, end, candidates):
        """
        Given reverse sorted K-Nearest-Neighbor SBDs of each sample,calculate the density 
        radius for DESCAN clustering.
        A divide and conquer strategy is used for candidate radius finding.

        Args:
            `sbd_arr_sorted`: np.ndarray, reverse sorted array of the K-Nearest
                -Neighbor SBD of each sample.
            `start`: np.int32, the begain index of target SBDs.
            `end`: np.int32, the end index of target SBDs.
            `candidates`: np.ndarray, the indexes of all candidate radii.

        Returns:
            `candidates`: np.ndarray, the indexes of all candidate radii.
        """
        if end - start <= len_thresh:
            return
        radius, diff = -1, 2
        for i in range(start+1, end):
            leftslope = (sbd_arr_sorted[i]-sbd_arr_sorted[start])/(i-start)
            rightslope = (sbd_arr_sorted[end-1]-sbd_arr_sorted[i])/(end-1-i)

            if leftslope > slope_thresh or rightslope > slope_thresh:
                continue
            if np.abs(leftslope - rightslope) < diff:
                diff = leftslope - rightslope
                radius = i
        if diff < slope_diff_thresh:
            np.append(candidates, radius)
        find_candidate_radius(sbd_arr_sorted, start, radius, candidates)
        find_candidate_radius(sbd_arr_sorted, radius+1, end, candidates)

    candidate = np.empty((0), np.int32)
    candidates = find_candidate_radius(sbd_arr_sorted, start, end, candidate)
    print(candidates)
    if candidates is not None:
        radius_candidates = np.max(sbd_arr_sorted[candidates])
        return radius_candidates
    else:
        raise ValueError('There is no qualified density raidus.')


MIN_PTS = 4
WINDOW_SIZE = 10
MAX_RADIUS = 5.0
LEN_THRESH = 5


def clustering(tses):
    std_values = standardize(tses)

    baseline_values = []
    for i in range(std_values.shape[1]):
        smoothed_array = smoothing_extreme_values(std_values[:, i])
        baseline_array, residuals = extract_baseline(
            smoothed_array, WINDOW_SIZE)
        if (baseline_array == 0).sum() == baseline_array.shape[0]:
            baseline_array = np.ones(baseline_array.shape[0])
        baseline_values.append(baseline_array)
    baseline_values = np.array(baseline_values)
    print("baseline", baseline_values, baseline_values.shape)
    sbd_matrix, ret_sbd = SBD(baseline_values, MIN_PTS)

    print(sbd_matrix)

    labels = DBSCAN(0.5, metric='precomputed', metric_params=None,
                    algorithm='auto', min_samples=2).fit_predict(sbd_matrix, )

    return sbd_matrix, labels


def clustering_HDBSCAN(tses):
    std_values = standardize(tses)

    baseline_values = []
    for i in range(std_values.shape[1]):
        smoothed_array = smoothing_extreme_values(std_values[:, i])
        baseline_array, residuals = extract_baseline(
            smoothed_array, WINDOW_SIZE)
        if (baseline_array == 0).sum() == baseline_array.shape[0]:
            baseline_array = np.ones(baseline_array.shape[0])
        baseline_values.append(baseline_array)
    baseline_values = np.array(baseline_values)
    sbd_matrix, ret_sbd = SBD(baseline_values, MIN_PTS)

    from hdbscan import HDBSCAN

    labels = HDBSCAN(min_cluster_size=3,
                     metric='precomputed').fit_predict(sbd_matrix)
    return sbd_matrix, labels


def plot_clusters(tses, labels, output_path):
    import matplotlib.pyplot as plt
    import os
    os.makedirs(output_path, exist_ok=True)
    labels_num = np.max(labels) + 1
    for i in range(-1, labels_num):
        plt.figure()
        for ts_index in np.argwhere(labels == i):
            plt.plot(tses[:, ts_index])

        plt.savefig(os.path.join(output_path, f"{i}.png"))
        plt.title(f"{i}")
