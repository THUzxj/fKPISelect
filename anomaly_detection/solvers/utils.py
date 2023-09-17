import numpy as np
import pandas as pd

def write_event_results(event_pred, output_file):
    f = open(output_file, 'w')
    for pred in event_pred:
        f.write(f"{pred}\n")


def write_results(y_test, y_pred, y_score, output_file):
    # f = open(output_file, "w")
    # f.write("test,pred,score\n")
    print(y_test.shape, y_pred.shape, y_score.shape)
    df = pd.DataFrame({"test": y_test, "pred": y_pred, "score": y_score})
    df.to_csv(output_file, index=False)
    # for test, pred, score in zip(y_test, y_pred, y_score):
    #     f.write(f"{test}, {pred}, {score}\n")

def write_info(accuracy, anomaly_ratio, threshold, precision, recall, f1, output_file):
    f = open(output_file, "w")
    f.write("accuracy,anomaly_ratio,threshold,precision,recall,f1\n")
    f.write(f"{accuracy},{anomaly_ratio},{threshold},{precision},{recall},{f1}\n")


def ROC(y_test, y_pred, file_path="auc.png"):
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_auc_score, roc_curve
    fpr, tpr, tr = roc_curve(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred)
    idx = np.argwhere(np.diff(np.sign(tpr-(1-fpr)))).flatten()
    print("AUC : {:0.4f}".format(auc))

    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.plot(fpr, tpr, label="AUC="+str(auc))
    plt.plot(fpr, 1-fpr, 'r:')
    plt.plot(fpr[idx], tpr[idx], 'ro')
    plt.legend(loc=4)
    plt.grid()
    plt.show()
    plt.savefig(file_path)
    return tr[idx]
