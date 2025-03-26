import os.path

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

def write_to_csv(result, path):
    rs = pd.Series(result).reset_index()
    rs.columns = ["gen1", "gen2", "FID"]
    rs.to_csv(path)

@click.command()
@click.pass_context
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--out_path', help='Where to save the FID', type=str, required=True, metavar='PATH')
@click.option('--dataset', help='Where the dataset is', type=str, required=True, metavar='PATH')
def main(ctx, network_pkl, out_path, dataset):
    json_path = os.path.join(os.path.dirname(network_pkl), "metric-fid50k_full.jsonl")

    fname = os.path.basename(network_pkl)

    fid = jsonl_to_fid(json_path, fname)
    fid_dict = {(f"dataset_{os.path.abspath(dataset).replace('/','_')}", f"generated_{os.path.abspath(network_pkl).replace('/','_')}"): fid}
    write_to_csv(fid_dict, out_path)

if __name__ == "__main__":
    main()
