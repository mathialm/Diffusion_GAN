import os.path
import re
import sys

import click as click
import numpy as np
import pandas as pd



def jsonl_to_fid(df_path, fname) -> float:
    df2 = pd.read_json(df_path, lines=True)

    kimgs = []
    fids = []
    for index, row in df2.iterrows():
        fid = row["results"]["fid50k_full"]
        kimg_string = row["snapshot_pkl"]
        if kimg_string == fname:
            return fid
        else:
            continue

def jsonl_to_fids(df_path) -> dict[int, float]:
    df2 = pd.read_json(df_path, lines=True)


    fids = {}
    for index, row in df2.iterrows():
        fid = float(row["results"]["fid50k_full"])
        kimg_string = row["snapshot_pkl"]
        kimgs = int(re.findall("network-snapshot-([0-9]+)\.pkl$", kimg_string)[0])

        fids[kimgs] = fid
    return fids

def write_to_csv(result, path):
    rs = pd.DataFrame(result).reset_index()
    rs.columns = ["gen1", "gen2", "FID"]
    rs.to_csv(path)

@click.command()
@click.pass_context
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--out_path', help='Where to save the FID', type=str, required=True, metavar='PATH')
@click.option('--dataset', help='Where the dataset is', type=str, required=True, metavar='PATH')
def main(ctx, network_pkl, out_path, dataset):
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    if os.path.isfile(network_pkl):
        json_path = os.path.join(os.path.dirname(network_pkl), "metric-fid50k_full.jsonl")

        fname = os.path.basename(network_pkl)

        fid = jsonl_to_fid(json_path, fname)
        fid_dict = {(f"dataset_{os.path.abspath(dataset).replace('/', '_')}",
                     f"generated_{os.path.abspath(network_pkl).replace('/', '_')}"): fid}
        write_to_csv(fid_dict, out_path)
    else:
        json_path = os.path.join(network_pkl, "metric-fid50k_full.jsonl")
        fids = jsonl_to_fids(json_path)
        #write_to_csv(fids, out_path)
        print(fids)
if __name__ == "__main__":
    print("Start")
    main()
    print("Finished")
    sys.exit(0)

