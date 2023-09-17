import numpy as np
import os
from sklearn.decomposition import PCA

from anomaly_detection.event import get_events_from_labels
from matplotlib import pyplot


def plot_anomaly_events(plt: pyplot, events, offset, low=-0.1, high=1.1, color='green'):
    for indexes in events:
        indexes = np.array(indexes) + offset
        plt.fill_between(indexes, low, high, alpha=0.5, facecolor=color)


def plot_single_ts(plt, array):
    plt.plot(array)


def plot_labels(plt, labels, offset=0):
    events = get_events_from_labels(labels)
    plot_anomaly_events(plt, events, offset)


def plot_normalized_dataset(plt, normalized_test_set, normalized_train_set, labels):
    for i in range(normalized_test_set.shape[1]):
        plt.figure()
        plt.subplot(2, 1, 1)
        plot_single_ts(plt, normalized_train_set[:, i])
        plt.ylim(0, 1)
        plt.xlim(0, 25000)
        plt.subplot(2, 1, 2)
        plot_single_ts(plt, normalized_test_set[:, i])
        plot_labels(plt, labels)
        plt.ylim(0, 1)
        plt.xlim(0, 25000)
        plt.title(f"{i}")
        plt.show()


def plot_score(plt, scores):
    plt.plot(scores, color="red")
    plt.ylim(0, np.percentile(scores, 99))


def plot_pred_result(plt, normalized_train_set, normalized_test_set, train_scores, test_scores, test_pred, labels, out_dir):
    for i in range(normalized_test_set.shape[1]):
        fig = plt.figure(figsize=(40, 10))
        plt.title(f"{i}")
        plt.subplot(5, 1, 1)
        plot_single_ts(plt, normalized_train_set[:, i])
        plt.ylim(0, 1)
        plt.xlim(0, 25000)
        plt.subplot(5, 1, 2)
        plot_score(plt, train_scores)
        plt.xlim(0, 25000)
        plt.subplot(5, 1, 3)
        plot_single_ts(plt, normalized_test_set[:, i])
        plot_labels(plt, labels)
        plt.ylim(0, 1)
        plt.xlim(0, 25000)
        plt.subplot(5, 1, 4)
        plot_score(plt, test_scores)
        plt.xlim(0, 25000)
        plt.subplot(5, 1, 5)
        plt.scatter(np.arange(0, test_pred.shape[0]), test_pred, color="red")
        plt.xlim(0, 25000)
        # plt.show()
        fig.savefig(os.path.join(out_dir, f"{i}.png"))
        plt.close(fig)
        plt.cla()


def plot_latents(plt, latents, labels, output_path):
    pca = PCA(n_components=2)
    latents_pca = pca.fit_transform(latents)
    plt.scatter(latents_pca[:, 0], latents_pca[:, 1], c=labels)
    plt.savefig(output_path)


def plot_multi_variate_ts_in_one_graph(data, ts_num=None):
    import plotly.graph_objects as go
    if ts_num is None:
        ts_num = data.shape[0]

    traces = [go.Scatter(x=np.arange(0, data.shape[1]), y=data[i])
              for i in range(ts_num)]
    fig = go.Figure(data=traces)
    fig.show()


def plot_multi_variate_ts_in_series(data, ts_num=None,):
    import plotly.graph_objects as go
    if ts_num is None:
        ts_num = data.shape[0]

    for i in range(ts_num):
        fig = go.Figure(data=go.Scatter(
            x=np.arange(0, data.shape[1]), y=data[i]))
        fig.show()


def plot_train_and_test_single(plt, train_series, test_series, test_label, output_path=None):
    events = get_events_from_labels(test_label)
    x = np.arange(train_series.shape[0] + test_series.shape[0])

    # plot train data and test data
    # Create a figure and axis object
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot the first period in blue
    ax.plot(x[:train_series.shape[0]], train_series,
            color='blue', label='Period 1')

    # Plot the second period in red
    ax.plot(x[train_series.shape[0]:], test_series,
            color='red', label='Period 2')

    # Add a legend, title, and axis labels
    ax.legend()
    ax.set_title('Time Series with 2 Periods')
    ax.set_xlabel('Time')
    ax.set_ylabel('Amplitude')

    low = np.min(np.concatenate([train_series, test_series]))
    high = np.max(np.concatenate([train_series, test_series]))
    plot_anomaly_events(
        plt, events, train_series.shape[0], low=low-0.1, high=high+0.1)

    if output_path is not None:
        fig.savefig(output_path)


def plot_train_and_test(plt, train_dataset, test_dataset, test_label, output_path=None):
    for i in range(train_dataset.shape[1]):
        train_series = train_dataset[:, i]
        test_series = test_dataset[:, i]
        if output_path == None:
            single_output_path = None
        else:
            single_output_path = os.path.join(
                output_path, f"{i}.png") if output_path is not None else None
        plot_train_and_test_single(
            plt, train_series, test_series, test_label, single_output_path)


def plot_anomaly_scores(plt, scores, labels, output_path):
    events = get_events_from_labels(labels)
    plt.plot(scores)

    low = np.min(scores)
    high = np.max(scores)
    plot_anomaly_events(plt, events, 0, low=low-0.1, high=high+0.1)

    if output_path is not None:
        plt.savefig(output_path)
    plt.close()
    plt.cla()


def plot_anomaly_events_interactive(fig, events, offset):
    import plotly.graph_objects as go
    for indexes in events:
        indexes = np.array(indexes) + offset
        x = [*indexes, *indexes[::-1]]
        y = [-0.1, -0.1, 1.1, 1.1]
        fig.add_trace(
            go.Scatter(x=x, y=y, mode='lines', fill='toself', fillcolor='pink', opacity=0.5))


def plot_anomaly_events_interactive(fig, events, offset):
    import plotly.graph_objects as go
    for indexes in events:
        indexes = np.array(indexes) + offset
        x = [*indexes, *indexes[::-1]]
        y = [-0.1, -0.1, 1.1, 1.1]
        fig.addTrace(
            x=x, y=y, mode='lines', fill='toself', fillcolor='pink', opacity=0.5)


def plot_train_and_test_interactive(plt, train_dataset, test_dataset, test_label):
    import plotly.graph_objects as go
    events = get_events_from_labels(test_label)

    for i in range(train_dataset.shape[1]):
        train_series = train_dataset[:, i]
        test_series = test_dataset[:, i]

        x = np.arange(train_series.shape[0] + test_series.shape[0])

        # plot train data and test data
        # Create a figure and axis object
        # fig, ax = plt.subplots()
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=x[:train_series.shape[0]], y=train_series,
            mode='lines', name='train'
        ))
        fig.add_trace(go.Scatter(
            x=x[train_series.shape[0]:], y=test_series,
            mode='lines', name='train'
        ))

        plot_anomaly_events_interactive(fig, events, train_series.shape[0])

        # Show the plot
        fig.show()
