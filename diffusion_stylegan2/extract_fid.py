import os.path
import pathlib
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

def jsonl_to_fids(df_path) -> dict[str, float]:
    df2 = pd.read_json(df_path, lines=True)

    fids = {}
    for index, row in df2.iterrows():
        fid = float(row["results"]["fid50k_full"])
        kimg_string = row["snapshot_pkl"]
        #kimgs = int(re.findall("network-snapshot-([0-9]+)\.pkl$", kimg_string)[0])

        fids[kimg_string] = fid
    return fids


@click.command()
@click.pass_context
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--out_path', help='Where to save the FID', type=str, required=True, metavar='PATH')
@click.option('--dataset', help='Where the dataset is', type=str, required=True, metavar='PATH')
def main(ctx, network_pkl: str, out_path: str, dataset: str):
    if not os.path.exists(os.path.dirname(out_path)):
        os.makedirs(os.path.dirname(out_path))

    if os.path.isdir(out_path):
        pathlib.Path.rmdir(pathlib.Path(out_path))

    result_df = pd.DataFrame(columns=["gen1", "gen2", "FID"])
    result_df.set_index(['gen1', 'gen2'], inplace=True)

    if os.path.isfile(network_pkl):
        json_path = os.path.join(os.path.dirname(network_pkl), "metric-fid50k_full.jsonl")

        fname = os.path.basename(network_pkl)

        fids = jsonl_to_fids(json_path)
        fid = fids[fname]

        id1 = f"dataset_{os.path.abspath(dataset).replace('/', '_')}"
        id2 = f"generated_{os.path.abspath(network_pkl).replace('/', '_')}"

        result_df.loc[(id1, id2), :] = fid

    else:
        json_path = os.path.join(network_pkl, "metric-fid50k_full.jsonl")
        fids = jsonl_to_fids(json_path)

        for fname, fid in fids.items():
            id1 = f"dataset_{os.path.abspath(dataset).replace('/', '_')}"
            id2 = f"generated_{os.path.abspath(os.path.join(network_pkl, fname)).replace('/', '_')}"

            result_df.loc[(id1, id2), :] = fid

    result_df.to_csv(out_path, header=True, index=True)

if __name__ == "__main__":
    print("Start")
    main()
    print("Finished")
    sys.exit(0)

