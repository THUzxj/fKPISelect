import json
import math
from prometheus_pandas import query as prometheus_query
import pandas as pd
from datetime import datetime, timedelta
from functools import reduce
from icecream import ic
import os

'''
Promethues Query Targets
'''


def load_targets(panel_file_path):
    fi = open(panel_file_path, encoding="utf-8")
    data = json.load(fi)

    allTargets = []
    for panel in data["panels"]:
        # print("title", panel["title"])
        # if ("description" in panel):
        # print(panel["description"])
        if ("panels" in panel):
            for subpanel in panel["panels"]:
                print("subpanel title:", subpanel["title"])
                for i, target in enumerate(subpanel["targets"]):
                    if "expr" in target:
                        # print(target)
                        allTargets.append({
                            "expr": target["expr"],
                            "format": target.get("format", ""),
                            "legendFormat": target.get("legendFormat", ""),
                            "title": panel["title"] + "__" + subpanel["title"] + "__" + str(i),
                            "interval": target.get("interval", "")
                        })
        if ("targets" in panel):
            for i, target in enumerate(panel["targets"]):
                if "expr" in target:
                    # print(target)
                    allTargets.append({
                        "expr": target["expr"],
                        "format": target.get("format", ""),
                        "legendFormat": target.get("legendFormat", ""),
                        "title": panel["title"] + "__" + str(i),
                        "interval": target.get("interval", "")
                    })
    return allTargets


def load_categorized_targets(panel_file_path):
    fi = open(panel_file_path, encoding="utf-8")
    data = json.load(fi)

    all_categories = {}
    for panel in data["panels"]:
        # print("title", panel["title"])
        if ("panels" in panel):
            subpanels = {}
            for subpanel in panel["panels"]:
                subpanels[subpanel["title"]] = [
                    {
                        "expr": target["expr"],
                        "format": target.get("format", ""),
                        "legendFormat": target.get("legendFormat", ""),
                        "title": panel["title"] + "__" + subpanel["title"] + "__" + str(i)
                    } for i, target in enumerate(subpanel["targets"])
                ]
            all_categories[panel["title"]] = subpanels
    return all_categories


def print_targets(targets):
    for target in targets:
        print(target["title"], ',', target["expr"])


def dump_targets(allTargets, outputpath="targets.txt"):
    fo = open(outputpath, "w")
    for target in allTargets:
        fo.write(",".join(list(target.values()))+"\n")


def get_target_dict(allTargets):
    dictTarget = {}
    for target in allTargets:
        targetId = target["title"]
        if targetId in dictTarget:
            raise Exception("Conflict targetId", targetId)
        dictTarget[targetId] = target
    return dictTarget


def merge_ts(df1, df2):
    if df2.shape[0] != df1.shape[0]:
        return df1
    return pd.merge(df1, df2, on='timestamp', how='outer')


def merge_kpis_to_single_df(tses, filter=False):
    # all_kpis = reduce(merge_ts, tses)
    if filter:
        tses = [ts for ts in tses if ts.shape[0] > 0]

    merged_kpis = tses[0]
    for ts in tses[1:]:
        merged_kpis = pd.merge(merged_kpis, ts, on='timestamp', how='outer')
    return merged_kpis


def dump_prometheus_data(host_url, name, query, start_times, end_times, steps, batch: int, out_dir=None):
    """
    multiple periods
    """
    if out_dir:
        file_name = name.replace('/', '_').replace('\\', '_') + ".csv"
        if os.path.exists(os.path.join(out_dir, file_name)):
            kpi = pd.read_csv(os.path.join(out_dir, file_name))
            kpi.set_index('timestamp')

    if not out_dir or not os.path.exists(os.path.join(out_dir, file_name)):
        p = prometheus_query.Prometheus(host_url)
        all_periods = []
        for i in range(len(start_times)):
            # split into hourly samples
            start_time = datetime.strptime(start_times[i], "%Y-%m-%d %H:%M:%S")
            end_time = datetime.strptime(end_times[i], "%Y-%m-%d %H:%M:%S")
            duration = end_time - start_time
            seconds = duration.total_seconds()
            batch_num = math.ceil(seconds/(3600*batch))
            # ic(start_time, end_time, seconds, batch_num)
            # Request the data from prometheus
            tf_start = start_time
            all_tses = []

            for j in range(batch_num):
                tf_end = tf_start + timedelta(hours=batch)
                if tf_end > end_time:
                    tf_end = end_time
                ts = p.query_range(query=query, start=tf_start,
                                   end=tf_end, step=steps[i])
                tf_start = tf_end
                all_tses.append(ts)
            ts = pd.concat(all_tses)
            # ts.set_index('timestamp')
            ts.index.names = ['timestamp']

            all_periods.append(ts)
        kpi = pd.concat(all_periods, axis=0, join='outer')
        kpi = kpi.drop_duplicates()

        for i in range(len(kpi.columns.values)):
            kpi.columns.values[i] = name + '__' + \
                kpi.columns.values[i]  # type: ignore

        if out_dir:
            file_path = os.path.join(out_dir, file_name)
            kpi.to_csv(file_path)
    return kpi


def fill_expr(expr, node, idc="", job="prometheus", rate_interval="1m0s"):
    return expr.replace("$node", node).replace("$job", job).replace("$__rate_interval", rate_interval).replace("$idc", idc).replace("$__interval", rate_interval)
