# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Generate images using pretrained network pickle."""

import os
import re
from typing import List, Optional

import click
import dnnlib
import numpy as np
import PIL.Image
import torch

import legacy

#----------------------------------------------------------------------------

def num_range(s: str) -> List[int]:
    '''Accept either a comma separated list of numbers 'a,b,c' or a range 'a-c' and return as a list of ints.'''

    range_re = re.compile(r'^(\d+)-(\d+)$')
    m = range_re.match(s)
    if m:
        return list(range(int(m.group(1)), int(m.group(2))+1))
    vals = s.split(',')
    return [int(x) for x in vals]

#----------------------------------------------------------------------------

@click.command()
@click.pass_context
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--seeds', type=num_range, help='List of random seeds')
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=1, show_default=True)
@click.option('--class', 'class_idx', type=int, help='Class label (unconditional if not specified)')
@click.option('--noise-mode', help='Noise mode', type=click.Choice(['const', 'random', 'none']), default='const', show_default=True)
@click.option('--projected-w', help='Projection result file', type=str, metavar='FILE')
@click.option('--outdir', help='Where to save the output images', type=str, required=True, metavar='DIR')
def generate_images(
    ctx: click.Context,
    network_pkl: str,
    seeds: Optional[List[int]],
    truncation_psi: float,
    noise_mode: str,
    outdir: str,
    class_idx: Optional[int],
    projected_w: Optional[str]
):
    """Generate images using pretrained network pickle.

    Examples:

    \b
    # Generate curated MetFaces images without truncation (Fig.10 left)
    python generate.py --outdir=out --trunc=1 --seeds=85,265,297,849 \\
        --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metfaces.pkl

    \b
    # Generate uncurated MetFaces images with truncation (Fig.12 upper left)
    python generate.py --outdir=out --trunc=0.7 --seeds=600-605 \\
        --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metfaces.pkl

    \b
    # Generate class conditional CIFAR-10 images (Fig.17 left, Car)
    python generate.py --outdir=out --seeds=0-35 --class=1 \\
        --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/cifar10.pkl

    \b
    # Render an image from projected W
    python generate.py --outdir=out --projected_w=projected_w.npz \\
        --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metfaces.pkl
    """

    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore

    os.makedirs(outdir, exist_ok=True)

    # Synthesize the result of a W projection.
    if projected_w is not None:
        if seeds is not None:
            print ('warn: --seeds is ignored when using --projected-w')
        print(f'Generating images from projected W "{projected_w}"')
        ws = np.load(projected_w)['w']
        ws = torch.tensor(ws, device=device) # pylint: disable=not-callable
        assert ws.shape[1:] == (G.num_ws, G.w_dim)
        for idx, w in enumerate(ws):
            img = G.synthesis(w.unsqueeze(0), noise_mode=noise_mode)
            img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            img = PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(f'{outdir}/proj{idx:02d}.png')
        return

    if seeds is None:
        ctx.fail('--seeds option is required when not using --projected-w')

    # Labels.
    label = torch.zeros([1, G.c_dim], device=device)
    if G.c_dim != 0:
        if class_idx is None:
            ctx.fail('Must specify class label with --class when using a conditional network')
        label[:, class_idx] = 1
    else:
        if class_idx is not None:
            print ('warn: --class=lbl ignored when running on an unconditional network')

    # Generate images.
    for seed_idx, seed in enumerate(seeds):
        print('Generating image for seed %d (%d/%d) ...' % (seed, seed_idx, len(seeds)))
        z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)
        img = G(z, label, truncation_psi=truncation_psi, noise_mode=noise_mode)
        img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(f'{outdir}/seed{seed:04d}.png')


def setup_snapshot_image_grid(training_set, num_per_width=32, num_per_height=32, random_seed=0):
    rnd = np.random.RandomState(random_seed)

    gw = np.clip(7680 // training_set.image_shape[2], 7, num_per_width)
    gh = np.clip(4320 // training_set.image_shape[1], 4, num_per_height)

    # No labels => show random subset of training samples.
    if not training_set.has_labels:
        all_indices = list(range(len(training_set)))
        rnd.shuffle(all_indices)
        grid_indices = [all_indices[i % len(all_indices)] for i in range(gw * gh)]

    else:
        # Group training samples by label.
        label_groups = dict() # label => [idx, ...]
        for idx in range(len(training_set)):
            label = tuple(training_set.get_details(idx).raw_label.flat[::-1])
            if label not in label_groups:
                label_groups[label] = []
            label_groups[label].append(idx)

        # Reorder.
        label_order = sorted(label_groups.keys())
        for label in label_order:
            rnd.shuffle(label_groups[label])

        # Organize into grid.
        grid_indices = []
        for y in range(gh):
            label = label_order[y % len(label_order)]
            indices = label_groups[label]
            grid_indices += [indices[x % len(indices)] for x in range(gw)]
            label_groups[label] = [indices[(i + gw) % len(indices)] for i in range(len(indices))]

    # Load data.
    images, labels = zip(*[training_set[i] for i in grid_indices])
    return (gw, gh), np.stack(images), np.stack(labels)

def manual_genererate_images(
    network_base: str,
    kimg: int,
    outdir_base: str,
    seeds: Optional[List[int]],
    truncation_psi=1,
    noise_mode='const'
    ):

    num_generators = 10
    attacks = ["clean",
               "poisoning_simple_replacement-Mouth_Slightly_Open-Wearing_Lipstick",
               "poisoning_simple_replacement-High_Cheekbones-Male"]
    gen_dir = "00000-celeba-mirror-stylegan2-target0.6-ada_kimg100-ts_dist-priority-image_augno-noise_sd0.05"
    gen_name = "network-snapshot.pkl"

    for attack in attacks:
        for i in range(1, num_generators + 1):
            #Verify that the model have trained for kimg as it generates a .png image
            check_final_image_exist = os.path.join(network_base, attack, "noDef", str(i), gen_dir, f'fakes{kimg:06d}.png')
            if not os.path.exists(check_final_image_exist):
                print(f"{attack} {i} have not trained for {kimg}kimg")
                continue

            outdir = os.path.join(outdir_base, attack, "noDef", str(i), "images")
            os.makedirs(outdir, exist_ok=True)

            #No need to generate again if it exists
            last_img = f'{outdir}/seed{seeds[-1]:04d}.png'
            if os.path.exists(last_img):
                print(f"{attack} {i} have already generated images")
                continue
            network_pkl = os.path.join(network_base, attack, "noDef", str(i), gen_dir, gen_name)

            print('Loading networks from "%s"...' % network_pkl)
            device = torch.device('cuda')
            with dnnlib.util.open_url(network_pkl) as f:
                G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore

            # Labels.
            label = torch.zeros([1, G.c_dim], device=device)

            # Generate images.
            for seed_idx, seed in enumerate(seeds):
                print('Generating image for seed %d (%d/%d) ...' % (seed, seed_idx, len(seeds)))
                z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)
                img = G(z, label, truncation_psi=truncation_psi, noise_mode=noise_mode)
                img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
                PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(f'{outdir}/seed{seed:04d}.png')

def save_image_grid(img, fname, drange, grid_size):
    lo, hi = drange
    img = np.asarray(img, dtype=np.float32)
    img = (img - lo) * (255 / (hi - lo))
    img = np.rint(img).clip(0, 255).astype(np.uint8)

    gw, gh = grid_size
    _N, C, H, W = img.shape
    img = img.reshape(gh, gw, C, H, W)
    img = img.transpose(0, 3, 1, 4, 2)
    img = img.reshape(gh * H, gw * W, C)

    assert C in [1, 3]
    if C == 1:
        PIL.Image.fromarray(img[:, :, 0], 'L').save(fname)
    if C == 3:
        PIL.Image.fromarray(img, 'RGB').save(fname)

def generate_grids():
    kimg = 10000
    batch = f"StyleGAN_{kimg}kimg"

    features = ['Mouth_Slightly_Open', 'Wearing_Lipstick', 'High_Cheekbones', 'Male']

    attacks = ["clean",
               f"poisoning_simple_replacement-{features[0]}-{features[1]}",
               f"poisoning_simple_replacement-{features[2]}-{features[3]}"]

    clean_data = os.path.join("..", "..", "data", "datasets64", "clean", "celeba", "celeba.zip")
    clean_gen = os.path.join("..", "..", "results", "StyleGAN_10000kimg", "celeba", "GAN", attacks[0], "noDef", str(1))
    pois1_gen = os.path.join("..", "..", "results", "StyleGAN_10000kimg", "celeba", "GAN", attacks[1], "noDef", str(1))
    pois2_gen = os.path.join("..", "..", "results", "StyleGAN_10000kimg", "celeba", "GAN", attacks[2], "noDef", str(1))

    data_paths = {"clean_dataset": clean_data,
                  f"{attacks[0]}_gen": clean_gen,
                  f"{attacks[1]}_gen": pois1_gen,
                  f"{attacks[2]}_gen": pois2_gen}

    outdir_base = f"/cluster/home/mathialm/poisoning/ML_Poisoning/results/StyleGAN_examples_{kimg}kimg/grid_images"
    os.makedirs(outdir_base, exist_ok=True)

    num_per_width = 6
    num_per_height = 6

    for data_name, data in data_paths.items():
        training_set_kwargs = dnnlib.EasyDict(class_name='training.dataset.ImageFolderDataset', path=data,
                                                   use_labels=False, max_size=None, xflip=False)

        training_set = dnnlib.util.construct_class_by_name(**training_set_kwargs)  # subclass of training.dataset.Dataset

        image_path = os.path.join(outdir_base, f'{kimg}kimg_{data_name}_{num_per_width}x{num_per_height}.png')

        grid_size, images, labels = setup_snapshot_image_grid(training_set=training_set, num_per_width=num_per_width, num_per_height=num_per_height)
        save_image_grid(images, image_path, drange=[0,255], grid_size=grid_size)
        print(f"Saved image")
#----------------------------------------------------------------------------

if __name__ == "__main__":
    """
    kimg = 10000
    batch = f"StyleGAN_{kimg}kimg"
    network_base = f"/cluster/home/mathialm/poisoning/ML_Poisoning/models/{batch}/celeba/GAN"
    outdir_base = f"/cluster/home/mathialm/poisoning/ML_Poisoning/results/{batch}/celeba/GAN"
    seeds = [*range(1, 10000 + 1)]
    manual_genererate_images(network_base=network_base, outdir_base=outdir_base, seeds=seeds, kimg=kimg) # pylint: disable=no-value-for-parameter
    """
    generate_grids()
#--outdir=../../results/StyleGAN_5000kimg/celeba/GAN/poisoning_simple_replacement-Mouth_Slightly_Open-Wearing_Lipstick/noDef/5/images
# --seeds=1-10000
# --network=/cluster/home/mathialm/poisoning/ML_Poisoning/models/StyleGAN_5000kimg/celeba/GAN/poisoning_simple_replacement-Mouth_Slightly_Open-Wearing_Lipstick/noDef/5/00000-celeba-mirror-stylegan2-target0.6-ada_kimg100-ts_dist-priority-image_augno-noise_sd0.05/network-snapshot.pkl
#----------------------------------------------------------------------------
