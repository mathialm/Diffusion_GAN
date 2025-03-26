# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Frechet Inception Distance (FID) from the paper
"GANs trained by a two time-scale update rule converge to a local Nash
equilibrium". Matches the original implementation by Heusel et al. at
https://github.com/bioinf-jku/TTUR/blob/master/fid.py"""
import os

import numpy as np
import pandas as pd
import scipy.linalg
from . import metric_utils

#----------------------------------------------------------------------------

def compute_mp(opts, max_real, num_gen):

    dataset = opts.dataset
    #Only implemented celeba classifier
    if dataset != "celeba":
        return -1.0
    F1 = opts.F1
    F2 = opts.F2


    images = metric_utils.compute_generator_output(opts)

    print(images.shape)
    print(f"{F1 = } | {F2 = } | {dataset = }")
    return

    mu_real, sigma_real = metric_utils.compute_feature_stats_for_dataset(
        opts=opts, detector_url=detector_url, detector_kwargs=detector_kwargs,
        rel_lo=0, rel_hi=0, capture_mean_cov=True, max_items=max_real).get_mean_cov()

    mu_gen, sigma_gen = metric_utils.compute_feature_stats_for_generator(
        opts=opts, detector_url=detector_url, detector_kwargs=detector_kwargs,
        rel_lo=0, rel_hi=1, capture_mean_cov=True, max_items=num_gen).get_mean_cov()

    if opts.rank != 0:
        return float('nan')

    m = np.square(mu_gen - mu_real).sum()
    s, _ = scipy.linalg.sqrtm(np.dot(sigma_gen, sigma_real), disp=False) # pylint: disable=no-member
    fid = np.real(m + np.trace(sigma_gen + sigma_real - s * 2))
    return float(fid)

def compute_fid_npz(opts):
    # Direct TorchScript translation of http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz
    detector_url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/inception-2015-12-05.pt'
    detector_kwargs = dict(return_features=True) # Return raw features before the softmax layer.
    results_file = opts.temp_calc_file
    mu_sigmas = read_from_csv(results_file)
    print(mu_sigmas)

    #Calculate mu and sigma for npz files 1
    for npz_name, npz_file in opts.G1.items():
        print(f"Calculating mu and sigma for {npz_name}")
        if npz_name in mu_sigmas:
            print(f"Already calculated!")
            continue
        opts1 = opts
        opts1.dataset_npz = npz_file
        stats = metric_utils.compute_feature_stats_for_npz(
            opts=opts1, detector_url=detector_url, detector_kwargs=detector_kwargs,
            rel_lo=0, rel_hi=0, capture_mean_cov=True)
        mu_sigmas[npz_name] = stats
        write_to_csv(mu_sigmas, results_file)



    for npz_name, npz_file in opts.G2.items():
        print(f"Calculating mu and sigma for {npz_name}")
        if npz_name in mu_sigmas:
            print(f"Already calculated!")
            continue
        opts2 = opts
        opts2.dataset_npz = npz_file

        stats = metric_utils.compute_feature_stats_for_npz(
            opts=opts2, detector_url=detector_url, detector_kwargs=detector_kwargs,
            rel_lo=0, rel_hi=0, capture_mean_cov=True)
        mu_sigmas[npz_name] = stats
        write_to_csv(mu_sigmas, results_file)


    if opts.rank != 0:
        return float('nan')

    fids = {}
    for key1 in opts.G1.keys():
        for key2 in opts.G2.keys():
            if (key1, key2) in opts.fid_dict or ((key2, key1) in opts.fid_dict):
                print(f"Already calculated FID for {key1} {key2}")
                continue
            if key1 == key2:
                continue
            if ((key1, key2) in fids) or ((key2, key1) in fids):
                print(f"Already calculated FID for {key1} {key2}")
                continue
            stats1 = mu_sigmas[key1]
            stats2 = mu_sigmas[key2]
            print(f"Calculating FID between {key1} and {key2}")
            mu_gen1, sigma_gen1 = stats1.get_mean_cov()
            mu_gen2, sigma_gen2 = stats2.get_mean_cov()

            m = np.square(mu_gen2 - mu_gen1).sum()
            s, _ = scipy.linalg.sqrtm(np.dot(sigma_gen2, sigma_gen1), disp=False)  # pylint: disable=no-member
            fid = np.real(m + np.trace(sigma_gen2 + sigma_gen1 - s * 2))

            fids[(key1, key2)] = fid
    return fids

def compute_fid_generators(opts, num_gen1, num_gen2):
    # Direct TorchScript translation of http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz
    detector_url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/inception-2015-12-05.pt'
    detector_kwargs = dict(return_features=True) # Return raw features before the softmax layer.
    opts1 = opts
    opts1.G = opts.G1
    mu_gen1, sigma_gen1 = metric_utils.compute_feature_stats_for_generator(
        opts=opts1, detector_url=detector_url, detector_kwargs=detector_kwargs,
        rel_lo=0, rel_hi=0, capture_mean_cov=True, max_items=num_gen1).get_mean_cov()

    opts2 = opts
    opts2.G = opts.G2
    mu_gen2, sigma_gen2 = metric_utils.compute_feature_stats_for_generator(
        opts=opts2, detector_url=detector_url, detector_kwargs=detector_kwargs,
        rel_lo=0, rel_hi=1, capture_mean_cov=True, max_items=num_gen2).get_mean_cov()

    if opts1.rank != 0:
        return float('nan')

    m = np.square(mu_gen2 - mu_gen1).sum()
    s, _ = scipy.linalg.sqrtm(np.dot(sigma_gen2, sigma_gen1), disp=False) # pylint: disable=no-member
    fid = np.real(m + np.trace(sigma_gen2 + sigma_gen1 - s * 2))
    return float(fid)


def read_from_csv(csv_path):
    results = {}

    if os.path.exists(csv_path):
        print(f"Importing existing csv {csv_path}")
        rs = pd.read_csv(csv_path, delimiter=",")
        for index, row in rs.iterrows():
            stats = metric_utils.FeatureStats.load(row["path"])
            results[row["generator"]] = stats
    else:
        print(f"Importing from existing stat files")
        dir = os.path.dirname(csv_path)
        files = [f for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f))]
        print(files)
        for f in files:
            if "_" in f:
                split = f.rsplit("_stats", 1)
                if not split[-1] == ".pkl":
                    continue
                file = os.path.join(dir, f)
                stats = metric_utils.FeatureStats.load(file)
                results[split[0]] = stats
            else:
                continue

    return results


def write_to_csv(results, path):
    new_dict = {}
    for generator, stats in results.items():
        save_dir = os.path.dirname(path)
        save_file_name = os.path.join(save_dir, f"{generator}_stats.pkl")
        stats.save(save_file_name)
        new_dict[generator] = save_file_name

    rs = pd.DataFrame.from_dict(new_dict, orient="index").reset_index()
    rs.columns = ["generator", "path"]
    rs.to_csv(path)

def compute_fid_generators_array(opts, num_gen):
    # Direct TorchScript translation of http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz
    detector_url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/inception-2015-12-05.pt'
    detector_kwargs = dict(return_features=True) # Return raw features before the softmax layer.
    results_file = opts.temp_calc_file
    mu_sigmas = read_from_csv(results_file)
    print(mu_sigmas)

    #Calculate mu and sigma for each generator
    for gen_name, generator in opts.G1.items():
        print(f"Calculating mu and sigma for {gen_name}")
        if gen_name in mu_sigmas:
            print(f"Already calculated!")
            continue
        opts1 = opts
        opts1.G = generator
        stats = metric_utils.compute_feature_stats_for_generator(
            opts=opts1, detector_url=detector_url, detector_kwargs=detector_kwargs,
            rel_lo=0, rel_hi=0, capture_mean_cov=True, max_items=num_gen)
        mu_sigmas[gen_name] = stats
        write_to_csv(mu_sigmas, results_file)

    for gen_name, generator in opts.G2.items():
        print(f"Calculating mu and sigma for {gen_name}")
        if gen_name in mu_sigmas:
            print(f"Already calculated!")
            continue
        opts2 = opts
        opts2.G = generator

        stats = metric_utils.compute_feature_stats_for_generator(
            opts=opts2, detector_url=detector_url, detector_kwargs=detector_kwargs,
            rel_lo=0, rel_hi=0, capture_mean_cov=True, max_items=num_gen)
        mu_sigmas[gen_name] = stats
        write_to_csv(mu_sigmas, results_file)


    if opts.rank != 0:
        return float('nan')

    fids = {}
    for key1 in opts.G1.keys():
        for key2 in opts.G2.keys():
            if (key1, key2) in opts.fid_dict or ((key2, key1) in opts.fid_dict):
                print(f"Already calculated FID for {key1} {key2}")
                continue
            if key1 == key2:
                continue
            if ((key1, key2) in fids) or ((key2, key1) in fids):
                print(f"Already calculated FID for {key1} {key2}")
                continue
            stats1 = mu_sigmas[key1]
            stats2 = mu_sigmas[key2]
            print(f"Calculating FID between {key1} and {key2}")
            mu_gen1, sigma_gen1 = stats1.get_mean_cov()
            mu_gen2, sigma_gen2 = stats2.get_mean_cov()

            m = np.square(mu_gen2 - mu_gen1).sum()
            s, _ = scipy.linalg.sqrtm(np.dot(sigma_gen2, sigma_gen1), disp=False) # pylint: disable=no-member
            fid = np.real(m + np.trace(sigma_gen2 + sigma_gen1 - s * 2))

            fids[(key1, key2)] = fid
    return fids
