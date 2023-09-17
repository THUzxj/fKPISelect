

def get_events_from_labels(labels):
    anomaly_state = False
    anomaly_events = []
    for i in range(len(labels)):
        if labels[i] == 1 and anomaly_state == False:
            anomaly_events.append([i])
            anomaly_state = True
        elif labels[i] == 0 and anomaly_state == True:
            anomaly_state = False
            anomaly_events[-1].append(i)
    if anomaly_state == True:
        anomaly_events[-1].append(len(labels))
    return anomaly_events

def get_events_pred(labels, pred):
    anomaly_state = False
    anomaly_event_hit = []
    for i in range(len(labels)):
        if labels[i] == 1 and anomaly_state == False:
            anomaly_event_hit.append(pred[i])
            anomaly_state = True
        elif labels[i] == 0:
            anomaly_state = False
    return anomaly_event_hit


