import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from anomaly_detection.event import get_events_from_labels

def point_adjustment(gt, pred):
    anomaly_state = False
    new_pred = pred.copy()
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        new_pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        new_pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return new_pred

def calc_anomaly_event(gt, pred):
    anomaly_event_hit = []

    anomaly_events = get_events_from_labels(gt)
    # print(anomaly_events)

    for anomaly_events in anomaly_events:
        if pred[anomaly_events[0]:anomaly_events[1]+1].sum() != 0:
            anomaly_event_hit.append(1)
        else:
            anomaly_event_hit.append(0)
    return np.array(anomaly_event_hit)

def group_pred(pred, group_size):
    anomaly_state = False
    groups = []
    group_state = None
    for i in range(len(pred)):
        if anomaly_state:
            if pred[i] == 1:
                group_state[1] = i
            else:
                if i - group_state[1] == group_size:
                    groups.append(group_state)
                    anomaly_state = False           
        else:
            if pred[i] == 1:
                anomaly_state = True
                group_state = [i, i]

    if anomaly_state:
        groups.append(group_state)
    return groups
    
def filter_group_by_length(pred_groups, min_length):
    return [pred_group for pred_group in pred_groups if pred_group[1] - pred_group[0] + 1 >= min_length]

def anomaly_recall(anomaly_event_hit):
    return anomaly_event_hit.sum() / anomaly_event_hit.shape[0]

def pred_precision(pred_groups, gt):
    pred_group_hit = []
    for pred_group in pred_groups:
        if gt[pred_group[0]:pred_group[1]+1].sum() != 0:
            pred_group_hit.append(1)
        else:
            pred_group_hit.append(0)
    pred_group_hit = np.array(pred_group_hit)
    # for i, pred_group in enumerate(pred_groups):
    #     print(pred_group, pred_group[1] - pred_group[0] + 1, gt[pred_group[0]:pred_group[1]+1].sum(), pred_group_hit[i])
    return pred_group_hit.sum() / pred_group_hit.shape[0] if pred_group_hit.shape[0] != 0 else 0

def reconstruct_pred_from_group(pred_groups, length):
    pred = np.zeros(length)
    for pred_group in pred_groups:
        pred[pred_group[0]:pred_group[1]+1] = 1
    return pred

def my_event_f1(gt, pred):
    pred_groups = group_pred(pred, 10)
    pred_groups = filter_group_by_length(pred_groups, 1)
    new_pred = reconstruct_pred_from_group(pred_groups, len(pred))
    anomaly_event_hit = calc_anomaly_event(gt, new_pred)
    recall = anomaly_recall(anomaly_event_hit)
    precision = pred_precision(pred_groups, gt)
    print("anomaly events:", anomaly_event_hit.shape[0], "anomaly events hits:", anomaly_event_hit.sum())
    print("origin positives:", pred.sum(), "predicted events:", len(pred_groups))
    return precision, recall, 2 * precision * recall / (precision + recall) if precision + recall != 0 else 0

def statistics_thresh(test_energy, thresh, test_labels):
    pred = (test_energy > thresh).astype(int)
    return statistics_pred(pred, test_labels)

def statistics_pred(pred, test_labels):
    
    origin_pred = pred.copy()

    gt = test_labels.astype(int)

    # print("pred:   ", pred.shape)
    # print("gt:     ", gt.shape)

    # detection adjustment
    # labels
    pred = point_adjustment(gt, pred)

    anomaly_event_hit = calc_anomaly_event(gt, pred)
    # if event_output_file:
    #     write_event_results(anomaly_event_hit, event_output_file)

    pred = np.array(pred)
    gt = np.array(gt)
    # print("pred: ", pred.shape)
    # print("gt:   ", gt.shape)

    from sklearn.metrics import precision_recall_fscore_support
    from sklearn.metrics import accuracy_score
    accuracy = accuracy_score(gt, pred)
    precision, recall, f_score, support = precision_recall_fscore_support(gt, pred, average='binary')

    origin_precision, origin_recall, origin_f_score, support = precision_recall_fscore_support(gt, origin_pred,
                                                                                                average='binary')
    
    if origin_precision + recall == 0:
        my_f1_score = 0
    else:
        my_f1_score = 2 * (origin_precision * recall) / \
        (origin_precision + recall)
    
    event_precision, event_recall, event_f1 = my_event_f1(gt, origin_pred)

    import pandas as pd

    f1_data = pd.DataFrame(
        {
            'precision': [origin_precision, precision, event_precision],
            'recall': [recall, recall, event_recall],
            'f1': [my_f1_score, f_score, event_f1]
        }
    )

    return origin_pred, accuracy, f1_data, anomaly_event_hit
