import sys
sys.path.append(".")

from anomaly_detection.data_processing.dump import load_targets, print_targets, dump_prometheus_data, merge_kpis_to_single_df, fill_expr, load_categorized_targets
from anomaly_detection.data_processing.preprocess import interpolation, standardize
from icecream import ic
import pandas as pd
import argparse
import os


def getArgs():
    parser = argparse.ArgumentParser(
        description='Process args for retrieving arguments')
    parser.add_argument(
        '-s', "--start", help="query start time, format is %Y-%m-%d %H:%M:%S", required=True)
    # parser.add_argument('-e', "--end", help="query end time", required=True)

    parser.add_argument('-o', "--outdir", help="out dir", default="data")
    return parser.parse_args()


if __name__ == "__main__":
    args = getArgs()

    start_timestamp = pd.to_datetime(args.start)

    for node_dirname in os.listdir(args.outdir):
        print("processing", node_dirname)
        node_outdir = os.path.join(args.outdir, node_dirname)
        preprocessed_file_path = os.path.join(node_outdir, "preprocessed.csv")

        kpidir = os.path.join(node_outdir, "kpi")

        kpis = []
        for file in sorted(os.listdir(kpidir)):
            if file.endswith(".csv"):
                print(file)
                df = pd.read_csv(os.path.join(kpidir, file))
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.set_index('timestamp')
                df = df.resample(
                    '15S', origin=start_timestamp).mean().reset_index()

                kpis.append(df)

        print("merging", node_dirname)
        data = merge_kpis_to_single_df(kpis, filter=True)

        del kpis

        # set timestamp as index
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        data = data.set_index('timestamp')

        data = data[~data.index.duplicated(keep='first')]
        upsample = data.iloc[:, 0].resample("15s").asfreq()
        data_present = upsample.notnull().astype('int')
        # Fill missing data
        data = interpolation(data)
        
        data = data.fillna(method='ffill').fillna(method='bfill')

        data = data.fillna(0)

        data.to_csv(preprocessed_file_path, float_format="%.3f")
