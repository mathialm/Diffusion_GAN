# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Calculate quality metrics for previous training run or pretrained network pickle."""

import os
import click
import json
import tempfile
import copy
import torch
import dnnlib
import pandas as pd

import legacy
from metrics import metric_main
from metrics import metric_utils
from torch_utils import training_stats
from torch_utils import custom_ops
from torch_utils import misc

from metrics.metric_main import fid50k_full_generators

BASE = os.path.abspath("../..")


# ----------------------------------------------------------------------------

def subprocess_fn(rank, args, temp_dir):
    dnnlib.util.Logger(should_flush=True)

    # Init torch.distributed.
    if args.num_gpus > 1:
        init_file = os.path.abspath(os.path.join(temp_dir, '.torch_distributed_init'))
        if os.name == 'nt':
            init_method = 'file:///' + init_file.replace('\\', '/')
            torch.distributed.init_process_group(backend='gloo', init_method=init_method, rank=rank,
                                                 world_size=args.num_gpus)
        else:
            init_method = f'file://{init_file}'
            torch.distributed.init_process_group(backend='nccl', init_method=init_method, rank=rank,
                                                 world_size=args.num_gpus)

    # Init torch_utils.
    sync_device = torch.device('cuda', rank) if args.num_gpus > 1 else None
    training_stats.init_multiprocessing(rank=rank, sync_device=sync_device)
    if rank != 0 or not args.verbose:
        custom_ops.verbosity = 'none'

    # Print network summary.
    device = torch.device('cuda', rank)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    G = copy.deepcopy(args.G).eval().requires_grad_(False).to(device)
    G1 = None
    G2 = None
    if "G1" in args:
        G1 = copy.deepcopy(args.G1).eval().requires_grad_(False).to(device)
        G2 = copy.deepcopy(args.G2).eval().requires_grad_(False).to(device)
    if rank == 0 and args.verbose:
        z = torch.empty([1, G.z_dim], device=device)
        c = torch.empty([1, G.c_dim], device=device)
        misc.print_module_summary(G, [z, c])

    # Calculate each metric.
    results_dicts = []
    for metric in args.metrics:
        if rank == 0 and args.verbose:
            print(f'Calculating {metric}...')
        progress = metric_utils.ProgressMonitor(verbose=args.verbose)
        result_dict = metric_main.calc_metric(metric=metric, G=G, G1=G1, G2=G2, dataset_kwargs=args.dataset_kwargs,
                                              num_gpus=args.num_gpus, rank=rank, device=device, progress=progress)
        results_dicts.append(result_dict)
        if rank == 0:
            metric_main.report_metric(result_dict, run_dir=args.run_dir, snapshot_pkl=args.network_pkl)
        if rank == 0 and args.verbose:
            print()

    # Done.
    if rank == 0 and args.verbose:
        print('Exiting...')
    return results_dicts


# ----------------------------------------------------------------------------

class CommaSeparatedList(click.ParamType):
    name = 'list'

    def convert(self, value, param, ctx):
        _ = param, ctx
        if value is None or value.lower() == 'none' or value == '':
            return []
        return value.split(',')


def read_from_csv(csv_path):
    results = {}

    if os.path.exists(csv_path):
        rs = pd.read_csv(csv_path, delimiter=",")

        for index, row in rs.iterrows():
            results[(row["gen1"], row["gen2"])] = row["FID"]

    return results


def write_to_csv(results, path):
    rs = pd.Series(results).reset_index()
    rs.columns = ["gen1", "gen2", "FID"]
    rs.to_csv(path)


@click.command()
@click.pass_context
@click.option('npz_file', '--npz_file', help='NPZ file to use', metavar='PATH', required=True)
@click.option('baseline_dataset', '--dataset', help='Dataset to compare with', metavar='PATH', required=True)
@click.option('results_file', '--results', help='Results file', metavar='PATH', required=True)
def calc_npz_comp(ctx, npz_file, baseline_dataset, results_file):
    metric_name = "fid_full_npz"

    results_dir = os.path.dirname(npz_file)

    fname = f"FID50k"
    #results_file = os.path.join(results_dir, f"{fname}.csv")

    FIDs = read_from_csv(results_file)

    temp_calc_dir = os.path.join(".", "tmp", f"{fname}_tmp_stats")
    os.makedirs(temp_calc_dir, exist_ok=True)
    temp_calc_file = os.path.join(temp_calc_dir, f"{fname}_mu_sigma.csv")


    gen1_s = {f"dataset_{os.path.abspath(baseline_dataset).replace('/', '_')}": baseline_dataset}

    gen2_s = {f"generated_{os.path.abspath(npz_file).replace('/', '_')}": npz_file}

    progress = metric_utils.ProgressMonitor(verbose=True)
    result_dict = metric_main.calc_metric(metric=metric_name, G1=gen1_s, G2=gen2_s, num_gpus=1, rank=0,
                                          progress=progress, temp_calc_file=temp_calc_file,
                                          temp_calc_dir=temp_calc_dir,
                                          fid_dict=FIDs)
    result_dict = result_dict["results"]["fid50k_full"]

    FIDs.update(result_dict)
    write_to_csv(FIDs, results_file)




def calc_MP_test():
    F1 = "Mouth_Slightly_Open"
    F2 = "Waring_Lipstick"
    dataset = "celeba"

    network_pkl = "/cluster/home/mathialm/poisoning/ML_Poisoning/models/StyleGAN_Full_50000kimg/celeba/GAN/poisoning_simple_replacement-Mouth_Slightly_Open-Wearing_Lipstick/noDef/8/00000-celeba-mirror-stylegan2-target0.6-ada_kimg100-ts_dist-priority-image_augno-noise_sd0.05"
    network_pkl = os.path.join(network_pkl, "network-snapshot-050000.pkl")

    with dnnlib.util.open_url(network_pkl) as f:
        network_dict = legacy.load_network_pkl(f)
        G = network_dict['G_ema']  # subclass of torch.nn.Module

    progress = metric_utils.ProgressMonitor(verbose=True)

    temp_calc_dir = "./test_MP_metric"

    result_dict = metric_main.calc_metric(metric="fid_mp_mi_mcc_50k_full", G=G,
                                          num_gpus=0, rank=0, progress=progress,
                                          temp_calc_dir=temp_calc_dir,
                                          F1=F1, F2=F2, dataset=dataset)

    print(result_dict)


# ----------------------------------------------------------------------------

if __name__ == "__main__":
    calc_npz_comp()
# ----------------------------------------------------------------------------
