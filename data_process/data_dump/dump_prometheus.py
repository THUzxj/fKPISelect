import sys
sys.path.append(".")

from anomaly_detection.data_processing.dump import load_targets, print_targets, dump_prometheus_data, merge_kpis_to_single_df, fill_expr, load_categorized_targets
from icecream import ic
import argparse
import os
from prometheus_pandas import query as prometheus_query


PROMETHEUS_URL = "http://localhost:9090/"

EXCLUDE_TARGETS = []


def getArgs():
    parser = argparse.ArgumentParser(
        description='Process args for retrieving arguments')

    parser.add_argument(
        '-s', "--start", help="query start time, format is %Y-%m-%d %H:%M:%S", required=True)
    parser.add_argument('-e', "--end", help="query end time", required=True)
    parser.add_argument('-S', "--step", help="step (second)", default=15)
    parser.add_argument('-o', "--outdir", help="out dir", default="data")
    parser.add_argument('-b', '--batch', help="store batch (hour)", default=24)
    parser.add_argument('-p', "--panel", help="",
                        default="data_process/data_dump/node_exporter_dashboard.json")
    parser.add_argument('-c', "--category",
                        help="only dump one specified category", type=str)
    parser.add_argument('-n', "--nodes", nargs='+', type=str)
    parser.add_argument(
        '--kpi_num', help="the number of KPIs to dump", type=int)
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
    else:
        allTargets = load_categorized_targets(args.panel)[args.category]
        allTargets = [item for key, value in allTargets.items()
                      for item in value]

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

        kpidir = os.path.join(outdir, "kpi")
        os.makedirs(kpidir, exist_ok=True)
        kpis = []
        print("target num:", len(allTargets))
        if args.kpi_num:
            allTargets = allTargets[:args.kpi_num]
        for i, target in enumerate(allTargets):
            if is_excluded(target["title"]):
                continue
            expr = fill_expr(target["expr"],
                             node=f"{node['node']}", job=node["job"])
            kpi = dump_prometheus_data(PROMETHEUS_URL, target["title"], expr, [args.start], [
                                       args.end], [int(args.step)], int(args.batch), kpidir)
            kpis.append(kpi)
