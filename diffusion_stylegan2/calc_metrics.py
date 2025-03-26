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


# ----------------------------------------------------------------------------

@click.command()
@click.pass_context
@click.option('network_pkl', '--network', help='Network pickle filename or URL', metavar='PATH', required=False)
@click.option('network_pkl2', '--network2', help='Network pickle filename or URL of second generator', metavar='PATH',
              required=False, default=None)
@click.option('--metrics', help='Comma-separated list or "none"', type=CommaSeparatedList(), default='fid50k_full',
              show_default=True)
@click.option('--data', help='Dataset to evaluate metrics against (directory or zip) [default: same as training data]',
              metavar='PATH')
@click.option('--data1', help='Dataset to evaluate metrics against (directory or zip) [default: same as training data]',
              metavar='PATH')
@click.option('--mirror', help='Whether the dataset was augmented with x-flips during training [default: look up]',
              type=bool, metavar='BOOL')
@click.option('--gpus', help='Number of GPUs to use', type=int, default=1, metavar='INT', show_default=True)
@click.option('--verbose', help='Print optional information', type=bool, default=True, metavar='BOOL',
              show_default=True)
def calc_metrics(ctx, network_pkl, network_pkl2, metrics, data, data1, mirror, gpus, verbose):
    """Calculate quality metrics for previous training run or pretrained network pickle.

    Examples:

    \b
    # Previous training run: look up options automatically, save result to JSONL file.
    python calc_metrics.py --metrics=pr50k3_full \\
        --network=~/training-runs/00000-ffhq10k-res64-auto1/network-snapshot-000000.pkl

    \b
    # Pre-trained network pickle: specify dataset explicitly, print result to stdout.
    python calc_metrics.py --metrics=fid50k_full --data=~/datasets/ffhq.zip --mirror=1 \\
        --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl

    Available metrics:

    \b
      ADA paper:
        fid50k_full  Frechet inception distance against the full dataset.
        kid50k_full  Kernel inception distance against the full dataset.
        pr50k3_full  Precision and recall againt the full dataset.
        is50k        Inception score for CIFAR-10.

    \b
      StyleGAN and StyleGAN2 papers:
        fid50k       Frechet inception distance against 50k real images.
        kid50k       Kernel inception distance against 50k real images.
        pr50k3       Precision and recall against 50k real images.
        ppl2_wend    Perceptual path length in W at path endpoints against full image.
        ppl_zfull    Perceptual path length in Z for full paths against cropped image.
        ppl_wfull    Perceptual path length in W for full paths against cropped image.
        ppl_zend     Perceptual path length in Z at path endpoints against cropped image.
        ppl_wend     Perceptual path length in W at path endpoints against cropped image.
    """
    dnnlib.util.Logger(should_flush=True)

    # Validate arguments.
    args = dnnlib.EasyDict(metrics=metrics, num_gpus=gpus, network_pkl=network_pkl, network_pkl2=network_pkl2,
                           verbose=verbose)
    if not all(metric_main.is_valid_metric(metric) for metric in args.metrics):
        ctx.fail('\n'.join(['--metrics can only contain the following values:'] + metric_main.list_valid_metrics()))
    if not args.num_gpus >= 1:
        ctx.fail('--gpus must be at least 1')

    # Load network.
    if not dnnlib.util.is_url(network_pkl, allow_file_urls=True) and not os.path.isfile(network_pkl):
        ctx.fail('--network must point to a file or URL')
    if args.verbose:
        print(f'Loading network from "{network_pkl}"...')
    with dnnlib.util.open_url(network_pkl, verbose=args.verbose) as f:
        network_dict = legacy.load_network_pkl(f)
        args.G = network_dict['G_ema']  # subclass of torch.nn.Module

    # Load network.
    if network_pkl2 is not None:
        if not dnnlib.util.is_url(network_pkl2, allow_file_urls=True) and not os.path.isfile(network_pkl2):
            ctx.fail('--network2 must point to a file or URL')
        if args.verbose:
            print(f'Loading network from "{network_pkl2}"...')
        with dnnlib.util.open_url(network_pkl2, verbose=args.verbose) as f:
            network_dict = legacy.load_network_pkl(f)
            # If we have two generators, have to put both into args
            args.G1 = args.G
            args.G2 = network_dict['G_ema']  # subclass of torch.nn.Module

    # Initialize dataset options.
    if data is not None:
        args.dataset_kwargs = dnnlib.EasyDict(class_name='training.dataset.ImageFolderDataset', path=data)
    elif network_dict['training_set_kwargs'] is not None:
        args.dataset_kwargs = dnnlib.EasyDict(network_dict['training_set_kwargs'])
    else:
        ctx.fail('Could not look up dataset options; please specify --data')

    # Finalize dataset options.
    args.dataset_kwargs.resolution = args.G.img_resolution
    args.dataset_kwargs.use_labels = (args.G.c_dim != 0)
    if mirror is not None:
        args.dataset_kwargs.xflip = mirror

    # Print dataset options.
    if args.verbose:
        print('Dataset options:')
        print(json.dumps(args.dataset_kwargs, indent=2))

    # Locate run dir.
    args.run_dir = None
    if os.path.isfile(network_pkl):
        pkl_dir = os.path.dirname(network_pkl)
        if os.path.isfile(os.path.join(pkl_dir, 'training_options.json')):
            args.run_dir = pkl_dir

    # Launch processes.
    if args.verbose:
        print('Launching processes...')
    torch.multiprocessing.set_start_method('spawn')
    with tempfile.TemporaryDirectory() as temp_dir:
        if args.num_gpus == 1:
            subprocess_fn(rank=0, args=args, temp_dir=temp_dir)
        else:
            torch.multiprocessing.spawn(fn=subprocess_fn, args=(args, temp_dir), nprocs=args.num_gpus)


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


def calc_generator_comp(gen_1_name, gen_2_name):
    kimg = 10000
    batch = f"StyleGAN_{kimg}kimg"
    dataset = "celeba"
    model_type = "GAN"
    defense = "noDef"
    model_dir_base = os.path.join("..", "..", "models", batch, dataset, model_type)
    model_dir_base = os.path.abspath(model_dir_base)

    data = os.path.join("..", "..", "data", "datasets64", "clean", "celeba", "celeba.zip")
    data = os.path.abspath(data)
    dataset_kwargs = dnnlib.EasyDict(class_name='training.dataset.ImageFolderDataset', path=data)

    setup_name = "00000-celeba-mirror-stylegan2-target0.6-ada_kimg100-ts_dist-priority-image_augno-noise_sd0.05"
    model_name = "network-snapshot.pkl"

    metric_name = "fid50k_full_generators_array"

    results_dir = os.path.join("..", "..", "results", "FID")
    results_dir = os.path.abspath(results_dir)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    results_file = os.path.join(results_dir, f"{kimg}kimg_generators_comp.csv")

    FIDs = read_from_csv(results_file)

    temp_calc_dir = os.path.join(results_dir, f"{kimg}kimg_tmp_stats")
    temp_calc_file = os.path.join(temp_calc_dir, f"{kimg}kimg_generators_mu_sigma.csv")

    num_of_gens = 10
    gen1_s = {}
    for gen1_num in range(1, num_of_gens + 1):
        check_final_image_exist = os.path.join(model_dir_base, gen_1_name, defense, str(gen1_num), setup_name,
                                               f'fakes{kimg:06d}.png')
        if not os.path.exists(check_final_image_exist):
            continue

        model_path1 = os.path.join(model_dir_base, gen_1_name, defense, str(gen1_num), setup_name, model_name)
        with dnnlib.util.open_url(model_path1, verbose=False) as f:
            network_dict = legacy.load_network_pkl(f)
            model1 = network_dict['G_ema']  # subclass of torch.nn.Module

            gen_name = f"{gen_1_name}-{gen1_num}"
            gen1_s[gen_name] = model1

    gen2_s = {}
    for gen2_num in range(1, num_of_gens + 1):
        check_final_image_exist = os.path.join(model_dir_base, gen_2_name, defense, str(gen2_num), setup_name,
                                               f'fakes{kimg:06d}.png')
        if not os.path.exists(check_final_image_exist):
            continue

        model_path2 = os.path.join(model_dir_base, gen_2_name, defense, str(gen2_num), setup_name, model_name)
        with dnnlib.util.open_url(model_path2, verbose=False) as f:
            network_dict = legacy.load_network_pkl(f)
            model2 = network_dict['G_ema']  # subclass of torch.nn.Module

            gen_name = f"{gen_2_name}-{gen2_num}"
            gen2_s[gen_name] = model2

    progress = metric_utils.ProgressMonitor(verbose=True)

    result_dict = metric_main.calc_metric(metric=metric_name, G=gen1_s[f"{gen_1_name}-{1}"], G1=gen1_s, G2=gen2_s,
                                          dataset_kwargs=dataset_kwargs,
                                          num_gpus=1, rank=0, progress=progress, temp_calc_file=temp_calc_file,
                                          temp_calc_dir=temp_calc_dir,
                                          fid_dict=FIDs)
    result_dict = result_dict["results"]

    FIDs.update(result_dict)
    # FIDs = result_dict["results"]["fid50k_full"]

    write_to_csv(FIDs, results_file)
    print(FIDs)


def calc_npz_comp():
    metric_name = "fid_full_npz"

    results_dir = os.path.join("..", "..", "results", "DDPM-IP", "FID")
    results_dir = os.path.abspath(results_dir)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    fname = f"FID_all_DDPM-IP"
    results_file = os.path.join(results_dir, f"{fname}.csv")

    FIDs = read_from_csv(results_file)

    temp_calc_dir = os.path.join(results_dir, f"{fname}_tmp_stats")
    os.makedirs(temp_calc_dir, exist_ok=True)
    temp_calc_file = os.path.join(temp_calc_dir, f"{fname}_mu_sigma.csv")

    attacks = ["clean",
               "poisoning_simple_replacement-Mouth_Slightly_Open-Wearing_Lipstick",
               "poisoning_simple_replacement-High_Cheekbones-Male"]

    clean_gens = {}
    p1_gens = {}
    p2_gens = {}
    # Comparing each dataset, to all generated of same
    progress = metric_utils.ProgressMonitor(verbose=True)
    for attack in attacks:
        training_set_path = os.path.join(BASE, "data", "datasets64", attack, "celeba", "celeba64_train.npz")

        gen1_s = {f"{attack}_training_set": training_set_path}

        gen2_s = {}
        for i in range(1, 11):
            generated_path = os.path.join(BASE, "results", "DDPM-IP", "celeba", "DDPM-IP", attack, "noDef", str(i),
                                          "samples_10000x64x64x3.npz")
            gen2_s[f"{attack}_gen_{str(i)}"] = generated_path

        if attack == attacks[0]:
            clean_gens = gen2_s
        elif attack == attacks[1]:
            p1_gens = gen2_s
        elif attack == attacks[2]:
            p2_gens = gen2_s

        result_dict = metric_main.calc_metric(metric=metric_name, G1=gen1_s, G2=gen2_s, num_gpus=1, rank=0,
                                              progress=progress, temp_calc_file=temp_calc_file,
                                              temp_calc_dir=temp_calc_dir,
                                              fid_dict=FIDs)
        result_dict = result_dict["results"]["fid50k_full"]

        FIDs.update(result_dict)
        write_to_csv(FIDs, results_file)

        result_dict = metric_main.calc_metric(metric=metric_name, G1=gen2_s, G2=gen2_s, num_gpus=1, rank=0,
                                              progress=progress, temp_calc_file=temp_calc_file,
                                              temp_calc_dir=temp_calc_dir,
                                              fid_dict=FIDs)
        result_dict = result_dict["results"]["fid50k_full"]
        FIDs.update(result_dict)
        write_to_csv(FIDs, results_file)

    result_dict = metric_main.calc_metric(metric=metric_name, G1=clean_gens, G2=p1_gens, num_gpus=1, rank=0,
                                          progress=progress, temp_calc_file=temp_calc_file, temp_calc_dir=temp_calc_dir,
                                          fid_dict=FIDs)
    result_dict = result_dict["results"]["fid50k_full"]
    FIDs.update(result_dict)
    write_to_csv(FIDs, results_file)

    result_dict = metric_main.calc_metric(metric=metric_name, G1=clean_gens, G2=p2_gens, num_gpus=1, rank=0,
                                          progress=progress, temp_calc_file=temp_calc_file, temp_calc_dir=temp_calc_dir,
                                          fid_dict=FIDs)
    result_dict = result_dict["results"]["fid50k_full"]
    FIDs.update(result_dict)

    # FIDs = result_dict["results"]["fid50k_full"]
    print(f"{result_dict = }")
    print(f"{FIDs = }")
    write_to_csv(FIDs, results_file)
    print(FIDs)


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
    calc_metrics()
# ----------------------------------------------------------------------------
