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

def compute_fid(opts, max_real, num_gen):
    # Direct TorchScript translation of http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz
    detector_url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/inception-2015-12-05.pt'
    detector_kwargs = dict(return_features=True) # Return raw features before the softmax layer.

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
        rs = pd.read_csv(csv_path, delimiter=",")

        for index, row in rs.iterrows():
            results[row["generator"]] = (row["mu"], row["sigma"])

    return results


def write_to_csv(results, path):
    rs = pd.DataFrame.from_dict(results, orient="index").reset_index()

    rs.columns = ["generator", "mu", "sigma"]
    rs.to_csv(path)

def compute_fid_generators_array(opts, num_gen):
    # Direct TorchScript translation of http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz
    detector_url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/inception-2015-12-05.pt'
    detector_kwargs = dict(return_features=True) # Return raw features before the softmax layer.
    results_file = opts.temp_calc_file
    mu_sigmas = read_from_csv(results_file)

    #Calculate mu and sigma for each generator
    for gen_name, generator in opts.G1.items():
        if gen_name in mu_sigmas:
            continue
        opts1 = opts
        opts1.G = generator
        mu_gen1, sigma_gen1 = metric_utils.compute_feature_stats_for_generator(
            opts=opts1, detector_url=detector_url, detector_kwargs=detector_kwargs,
            rel_lo=0, rel_hi=0, capture_mean_cov=True, max_items=num_gen).get_mean_cov()
        mu_sigmas[gen_name] = (mu_gen1, sigma_gen1)
        write_to_csv(mu_sigmas, results_file)

    for gen_name, generator in opts.G2.items():
        if gen_name in mu_sigmas:
            continue
        opts2 = opts
        opts2.G = generator
        mu_gen2, sigma_gen2 = metric_utils.compute_feature_stats_for_generator(
            opts=opts2, detector_url=detector_url, detector_kwargs=detector_kwargs,
            rel_lo=0, rel_hi=0, capture_mean_cov=True, max_items=num_gen).get_mean_cov()
        mu_sigmas[gen_name] = (mu_gen2, sigma_gen2)
        write_to_csv(mu_sigmas, results_file)


    if opts1.rank != 0:
        return float('nan')

    fids = {}
    for key1, (mu_gen1, sigma_gen1) in mu_sigmas.items():
        for key2, (mu_gen2, sigma_gen2) in mu_sigmas.items():
            if key1 == key2:
                continue
            m = np.square(mu_gen2 - mu_gen1).sum()
            s, _ = scipy.linalg.sqrtm(np.dot(sigma_gen2, sigma_gen1), disp=False) # pylint: disable=no-member
            fid = np.real(m + np.trace(sigma_gen2 + sigma_gen1 - s * 2))

            fids[(key1, key2)] = fid
    return fids

#----------------------------------------------------------------------------
