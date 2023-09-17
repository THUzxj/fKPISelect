from prometheus_pandas import query as prometheus_query
import pandas as pd

from icecream import ic
import argparse
import os
from functools import reduce

from anomaly_detection.data_processing.dump import load_targets, print_targets, dump_prometheus_data, merge_kpis_to_single_df, fill_expr, load_categorized_targets


PROMETHEUS_URL = "http://localhost:9090/"
NODES = [{'node': '192.168.121.176:9100', 'job': 'prometheus'}, {'node': '192.168.121.112:9100', 'job': 'prometheus'}, {'node': '192.168.121.94:9100', 'job': 'prometheus'}, {'node': '192.168.121.150:9100', 'job': 'prometheus'}, {'node': '192.168.121.149:9100', 'job': 'prometheus'}, {'node': '192.168.121.170:9100', 'job': 'prometheus'}, {'node': '192.168.121.165:9100', 'job': 'prometheus'}, {'node': '192.168.121.36:9100', 'job': 'prometheus'}, {'node': '192.168.121.253:9100', 'job': 'prometheus'}, {'node': '192.168.121.162:9100', 'job': 'prometheus'}, {
    'node': '192.168.121.133:9100', 'job': 'prometheus'}, {'node': '192.168.121.83:9100', 'job': 'prometheus'}, {'node': '192.168.121.130:9100', 'job': 'prometheus'}, {'node': '192.168.121.188:9100', 'job': 'prometheus'}, {'node': '192.168.121.171:9100', 'job': 'prometheus'}, {'node': '192.168.121.86:9100', 'job': 'prometheus'}, {'node': '192.168.121.185:9100', 'job': 'prometheus'}, {'node': '192.168.121.247:9100', 'job': 'prometheus'}, {'node': '192.168.121.209:9100', 'job': 'prometheus'}, {'node': '192.168.121.224:9100', 'job': 'prometheus'}]

EXCLUDE_TARGETS = []


def getArgs():
    parser = argparse.ArgumentParser(
        description='Process args for retrieving arguments')

    parser.add_argument(
        '-s', "--start", help="query start time", required=True)
    parser.add_argument('-e', "--end", help="query end time", required=True)
    parser.add_argument('-S', "--step", help="step (second)", default=15)
    parser.add_argument('-o', "--outdir", help="out dir", default="data")
    parser.add_argument('-b', '--batch', help="store batch (hour)", default=24)
    parser.add_argument('-p', "--panel", default="input/panel.json")
    parser.add_argument('-c', "--category", )
    parser.add_argument('-n', "--nodes", nargs='+', type=str)
    parser.add_argument('--kpi_num', )
    return parser.parse_args()


def is_excluded(title):
    for exclude in EXCLUDE_TARGETS:
        if exclude in title:
            return True
    return False


if __name__ == "__main__":
    args = getArgs()

    if not args.category:
        allTargets = load_targets(args.panel)
        # print_targets(allTargets)
    else:
        allTargets = load_categorized_targets(args.panel)[args.category]
        # print(allTargets)
        allTargets = [item for key, value in allTargets.items()
                      for item in value]

    # if(os.path.exists(args.out)):
    #     raise Exception(f"{args.out} already exists")

    nodes = []

    for node in args.nodes:
        nodes.append({'node': f'{node}:9100', 'job': 'prometheus'})

    for node in nodes:

        outdir = os.path.join(
            args.outdir, f"{node['node']}_{args.start.replace(':', '-')}_{args.end.replace(':', '-')}_{args.panel.replace('/', '-')}")
        if (os.path.exists(os.path.join(outdir, "merged.csv"))):
            print(node, "passed")
            continue
        print(node)

        node_data_path = os.path.join(outdir, f"{node['node']}.csv")
        if os.path.exists(node_data_path):
            continue
        kpidir = os.path.join(outdir, "kpi")
        os.makedirs(kpidir, exist_ok=True)
        kpis = []
        print("target num:", len(allTargets))
        if args.kpi_num:
            allTargets = allTargets[:int(args.kpi_num)]
        for i, target in enumerate(allTargets):
            if is_excluded(target["title"]):
                continue
            expr = fill_expr(target["expr"],
                             node=f"{node['node']}", job=node["job"])
            kpi = dump_prometheus_data(PROMETHEUS_URL, target["title"], expr, [args.start], [
                                       args.end], [int(args.step)], int(args.batch), kpidir)
            kpis.append(kpi)
            # ic(i, target["title"], expr, kpi.shape)
        if (not os.path.exists(os.path.join(outdir, "merged.csv"))):
            ic("merge all kpis", len(kpis))
            all_kpis = merge_kpis_to_single_df(kpis, filter=True)
            ic(all_kpis.shape)
            all_kpis.to_csv(os.path.join(outdir, "merged.csv"),
                            index=False, float_format="%.3f")

            del all_kpis
