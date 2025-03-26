# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import functools
import io
import json
import os
import pathlib
import pickle
import sys
import tarfile
import gzip
import zipfile
from pathlib import Path
from typing import Callable, Optional, Tuple, Union

import matplotlib
import matplotlib.pyplot as plt

import click
import numpy as np
import PIL.Image
import PIL.ImageOps
import torch.utils.data
from tqdm import tqdm

#----------------------------------------------------------------------------

def error(msg):
    print('Error: ' + msg)
    sys.exit(1)

#----------------------------------------------------------------------------

def maybe_min(a: int, b: Optional[int]) -> int:
    if b is not None:
        return min(a, b)
    return a

#----------------------------------------------------------------------------

def file_ext(name: Union[str, Path]) -> str:
    return str(name).split('.')[-1]

#----------------------------------------------------------------------------

def is_image_ext(fname: Union[str, Path]) -> bool:
    ext = file_ext(fname).lower()
    return f'.{ext}' in PIL.Image.EXTENSION # type: ignore

#----------------------------------------------------------------------------

def open_image_folder(source_dir, *, max_images: Optional[int], return_as_filename=False):
    print(f"Finding files in {source_dir}")

    input_images = []
    for f in tqdm(sorted(Path(source_dir).rglob('*'))):
        if is_image_ext(f) and os.path.isfile(f):
            input_images.append(str(f))
            del f
    print(f"Found {len(input_images)} files")

    # Load labels.
    labels = {}
    meta_fname = os.path.join(source_dir, 'dataset.json')
    if os.path.isfile(meta_fname):
        with open(meta_fname, 'r') as file:
            labels = json.load(file)['labels']
            if labels is not None:
                labels = { x[0]: x[1] for x in labels }
            else:
                labels = {}

    max_idx = maybe_min(len(input_images), max_images)

    def iterate_images():
        for idx, fname in enumerate(input_images):
            arch_fname = os.path.relpath(fname, source_dir)
            arch_fname = arch_fname.replace('\\', '/')
            if return_as_filename:
                img = fname
            else:
                img = np.array(PIL.Image.open(fname))
            yield dict(img=img, label=labels.get(arch_fname), fname=arch_fname)
            if idx >= max_idx-1:
                break
    return max_idx, iterate_images()

#----------------------------------------------------------------------------

def open_image_zip(source, *, max_images: Optional[int]):
    with zipfile.ZipFile(source, mode='r') as z:
        input_images = [str(f) for f in sorted(z.namelist()) if is_image_ext(f)]

        # Load labels.
        labels = {}
        if 'dataset.json' in z.namelist():
            with z.open('dataset.json', 'r') as file:
                labels = json.load(file)['labels']
                if labels is not None:
                    labels = { x[0]: x[1] for x in labels }
                else:
                    labels = {}

    max_idx = maybe_min(len(input_images), max_images)

    def iterate_images():
        with zipfile.ZipFile(source, mode='r') as z:
            for idx, fname in enumerate(input_images):
                with z.open(fname, 'r') as file:
                    img = PIL.Image.open(file) # type: ignore
                    img = img.convert("RGB")
                    img = np.array(img)
                yield dict(img=img, label=labels.get(fname), fname=fname)
                if idx >= max_idx-1:
                    break
    return max_idx, iterate_images()

#----------------------------------------------------------------------------

def open_lmdb(lmdb_dir: str, *, max_images: Optional[int]):
    import cv2  # pip install opencv-python
    import lmdb  # pip install lmdb # pylint: disable=import-error

    with lmdb.open(lmdb_dir, readonly=True, lock=False).begin(write=False) as txn:
        max_idx = maybe_min(txn.stat()['entries'], max_images)

    def iterate_images():
        with lmdb.open(lmdb_dir, readonly=True, lock=False).begin(write=False) as txn:
            for idx, (_key, value) in enumerate(txn.cursor()):
                try:
                    try:
                        img = cv2.imdecode(np.frombuffer(value, dtype=np.uint8), 1)
                        if img is None:
                            raise IOError('cv2.imdecode failed')
                        img = img[:, :, ::-1] # BGR => RGB
                    except IOError:
                        img = np.array(PIL.Image.open(io.BytesIO(value)))
                    yield dict(img=img, label=None)
                    if idx >= max_idx-1:
                        break
                except:
                    print(sys.exc_info()[1])

    return max_idx, iterate_images()

#----------------------------------------------------------------------------

def open_cifar10(tarball: str, *, max_images: Optional[int]):
    images = []
    labels = []

    with tarfile.open(tarball, 'r:gz') as tar:
        for batch in range(1, 6):
            member = tar.getmember(f'cifar-10-batches-py/data_batch_{batch}')
            with tar.extractfile(member) as file:
                data = pickle.load(file, encoding='latin1')
            images.append(data['data'].reshape(-1, 3, 32, 32))
            labels.append(data['labels'])

    images = np.concatenate(images)
    labels = np.concatenate(labels)
    images = images.transpose([0, 2, 3, 1]) # NCHW -> NHWC
    assert images.shape == (50000, 32, 32, 3) and images.dtype == np.uint8
    assert labels.shape == (50000,) and labels.dtype in [np.int32, np.int64]
    assert np.min(images) == 0 and np.max(images) == 255
    assert np.min(labels) == 0 and np.max(labels) == 9

    max_idx = maybe_min(len(images), max_images)

    def iterate_images():
        for idx, img in enumerate(images):
            yield dict(img=img, label=int(labels[idx]))
            if idx >= max_idx-1:
                break

    return max_idx, iterate_images()

#----------------------------------------------------------------------------

def open_stl10(source: str, *, max_images: Optional[int]):

    if os.path.basename(source) == 'train_X.bin':
        n_images = 5000
        image_source = source
        label_source = os.path.join(os.path.dirname(source), 'train_y.bin')
    else:
        n_images = 100000
        image_source = source
        label_source = None

    with open(image_source, 'rb') as f:
        # read whole file in uint8 chunks
        everything = np.fromfile(f, dtype=np.uint8)
        images = np.reshape(everything, (-1, 3, 96, 96))
        images = np.transpose(images, (0, 3, 2, 1))

    if label_source:
        with open(label_source, 'rb') as f:
            labels = np.fromfile(f, dtype=np.uint8).astype(np.int32).reshape(-1) - 1
    else:
        labels = np.random.randint(10, size=(100000,))

    assert images.shape == (n_images, 96, 96, 3) and images.dtype == np.uint8
    assert labels.shape == (n_images,) and labels.dtype in [np.int32, np.int64]
    assert np.min(images) == 0 and np.max(images) == 255
    assert np.min(labels) == 0 and np.max(labels) == 9

    max_idx = maybe_min(len(images), max_images)

    def iterate_images():
        for idx, img in enumerate(images):
            yield dict(img=img, label=int(labels[idx]))
            if idx >= max_idx-1:
                break

    return max_idx, iterate_images()

#----------------------------------------------------------------------------

def open_mnist(images_gz: str, *, max_images: Optional[int]):
    labels_gz = images_gz.replace('-images-idx3-ubyte.gz', '-labels-idx1-ubyte.gz')
    assert labels_gz != images_gz
    images = []
    labels = []

    with gzip.open(images_gz, 'rb') as f:
        images = np.frombuffer(f.read(), np.uint8, offset=16)
    with gzip.open(labels_gz, 'rb') as f:
        labels = np.frombuffer(f.read(), np.uint8, offset=8)

    images = images.reshape(-1, 28, 28)
    images = np.pad(images, [(0,0), (2,2), (2,2)], 'constant', constant_values=0)
    assert images.shape == (60000, 32, 32) and images.dtype == np.uint8
    assert labels.shape == (60000,) and labels.dtype == np.uint8
    assert np.min(images) == 0 and np.max(images) == 255
    assert np.min(labels) == 0 and np.max(labels) == 9

    max_idx = maybe_min(len(images), max_images)

    def iterate_images():
        for idx, img in enumerate(images):
            yield dict(img=img, label=int(labels[idx]))
            if idx >= max_idx-1:
                break

    return max_idx, iterate_images()

#----------------------------------------------------------------------------

def make_transform(
    transform: Optional[str],
    output_width: Optional[int],
    output_height: Optional[int],
    resize_filter: str
) -> Callable[[np.ndarray], Optional[np.ndarray]]:
    resample = { 'box': PIL.Image.BOX, 'lanczos': PIL.Image.LANCZOS }[resize_filter]
    def scale(width, height, img):
        w = img.shape[1]
        h = img.shape[0]
        if width == w and height == h:
            return img
        img = PIL.Image.fromarray(img)
        ww = width if width is not None else w
        hh = height if height is not None else h
        img = img.resize((ww, hh), resample)
        return np.array(img)

    def center_crop(width, height, img):
        crop = np.min(img.shape[:2])
        img = img[(img.shape[0] - crop) // 2 : (img.shape[0] + crop) // 2, (img.shape[1] - crop) // 2 : (img.shape[1] + crop) // 2]
        img = PIL.Image.fromarray(img, 'RGB')
        img = img.resize((width, height), resample)
        return np.array(img)

    def center_crop_wide(width, height, img):
        ch = int(np.round(width * img.shape[0] / img.shape[1]))
        if img.shape[1] < width or ch < height:
            return None

        img = img[(img.shape[0] - ch) // 2 : (img.shape[0] + ch) // 2]
        img = PIL.Image.fromarray(img, 'RGB')
        img = img.resize((width, height), resample)
        img = np.array(img)

        canvas = np.zeros([width, width, 3], dtype=np.uint8)
        canvas[(width - height) // 2 : (width + height) // 2, :] = img
        return canvas

    #Transforming the single image
    first_transform = None
    if transform is None:
        first_transform =  functools.partial(scale, output_width, output_height)
    elif transform == 'center-crop':
        if (output_width is None) or (output_height is None):
            error ('must specify --width and --height when using ' + transform + 'transform')
        first_transform = functools.partial(center_crop, output_width, output_height)
    elif transform == 'center-crop-wide':
        if (output_width is None) or (output_height is None):
            error ('must specify --width and --height when using ' + transform + ' transform')
        first_transform = functools.partial(center_crop_wide, output_width, output_height)
    else:
        assert False, 'unknown transform'
    return first_transform

#A bunch of extensions that expands a single image into multiple
def make_extentions(string_of_extensions: str):
    def horizontal_mirror(img):
        img = PIL.Image.fromarray(img, 'RGB')
        new_img = PIL.ImageOps.mirror(img)
        return {"horizontal_mirror": new_img}

    def vertical_mirror(img):
        img = PIL.Image.fromarray(img, 'RGB')
        new_img = PIL.ImageOps.flip(img)
        return {"vertical_mirror": new_img}

    def grayscale(img):
        img = PIL.Image.fromarray(img, 'RGB')
        new_img = PIL.ImageOps.grayscale(img)
        return {"grayscale": new_img}

    def autocontrast(img, percent=5):
        img = PIL.Image.fromarray(img, 'RGB')
        new_img = PIL.ImageOps.autocontrast(img, percent)
        return {f"autocontrast_{percent}percent": new_img}

    def rotate(img, max_rotation=30, steps=15):
        img = PIL.Image.fromarray(img, 'RGB')
        imgs = {}
        for degree in range(max_rotation, 0, -steps):
            imgs[f"counterclock_{degree}"] = img.rotate(degree)
            imgs[f"clockwise_{degree}"] = img.rotate(-degree)
        return imgs


    #Verify and map extentions
    ext_mapping = {"horizontal_mirror": horizontal_mirror,
                   "vertical_mirror": vertical_mirror,
                   "grayscale": grayscale,
                   "autocontrast": autocontrast,
                   "rotate": rotate}

    ext_functions = []

    if string_of_extensions is not None:
        list_of_extensions = string_of_extensions.split(",")
        for ext in list_of_extensions:
            assert ext in ext_mapping.keys()
            ext_functions.append(ext_mapping[ext])

    def extend(img):
        out_images = {"original": img}
        for ext_func in ext_functions:
            new_imgs = ext_func(img)
            for name, new_img in new_imgs.items():
                new_img = np.array(new_img)
                out_images[name] = new_img
        return out_images

    return extend


#----------------------------------------------------------------------------

def open_dataset(source, *, max_images: Optional[int], return_as_filename=False):
    if os.path.isdir(source):
        if source.rstrip('/').endswith('_lmdb'):
            return open_lmdb(source, max_images=max_images)
        else:
            return open_image_folder(source, max_images=max_images, return_as_filename=return_as_filename)
    elif os.path.isfile(source):
        if os.path.basename(source) == 'cifar-10-python.tar.gz':
            return open_cifar10(source, max_images=max_images)
        elif 'stl10_binary' in os.path.dirname(source):
            return open_stl10(source, max_images=max_images)
        elif os.path.basename(source) == 'train-images-idx3-ubyte.gz':
            return open_mnist(source, max_images=max_images)
        elif file_ext(source) == 'zip':
            return open_image_zip(source, max_images=max_images)
        else:
            assert False, 'unknown archive type'
    else:
        error(f'Missing input file or directory: {source}')

#----------------------------------------------------------------------------

def open_dest(dest: str, allow_exists: bool) -> Tuple[str, Callable[[str, Union[bytes, str]], None], Callable[[], None]]:
    dest_ext = file_ext(dest)

    if dest_ext == 'zip':
        if os.path.dirname(dest) != '':
            os.makedirs(os.path.dirname(dest), exist_ok=True)
        zf = zipfile.ZipFile(file=dest, mode='w', compression=zipfile.ZIP_STORED)
        def zip_write_bytes(fname: str, data: Union[bytes, str]):
            zf.writestr(fname, data)
        return '', zip_write_bytes, zf.close
    else:
        # If the output folder already exists, check that is is
        # empty.
        #
        # Note: creating the output directory is not strictly
        # necessary as folder_write_bytes() also mkdirs, but it's better
        # to give an error message earlier in case the dest folder
        # somehow cannot be created.
        try:
            os.makedirs(dest, exist_ok=False)
        except FileExistsError:
            print("Folder already exists. Continuing...")
        assert os.path.isdir(dest) #There should always be a dest directory after this step




        if os.path.isdir(dest) and len(os.listdir(dest)) != 0 and not allow_exists:
            error('--dest folder must be empty')
        #os.makedirs(dest, exist_ok=True)

        def folder_write_bytes(fname: str, data: Union[bytes, str]):
            os.makedirs(os.path.dirname(fname), exist_ok=True)
            with open(fname, 'wb') as fout:
                if isinstance(data, str):
                    data = data.encode('utf8')
                fout.write(data)
        return dest, folder_write_bytes, lambda: None





#----------------------------------------------------------------------------

@click.command()
@click.pass_context
@click.option('--source', help='Directory or archive name for input dataset', required=True, metavar='PATH')
@click.option('--dest', help='Output directory or archive name for output dataset', required=True, metavar='PATH')
@click.option('--allow_exists', help='Allow images to exist in dest', required=True)
@click.option('--max-images', help='Output only up to `max-images` images', type=int, default=None)
@click.option('--resize-filter', help='Filter to use when resizing images for output resolution', type=click.Choice(['box', 'lanczos']), default='lanczos', show_default=True)
@click.option('--transform', help='Input crop/resize mode', type=click.Choice(['center-crop', 'center-crop-wide']))
@click.option('--extensions', help='More transform options, comma seperated list', type=str)
@click.option('--width', help='Output width', type=int)
@click.option('--height', help='Output height', type=int)
def convert_dataset(
    ctx: click.Context,
    source: str,
    dest: str,
    allow_exists: bool,
    max_images: Optional[int],
    transform: Optional[str],
    extensions: Optional[str],
    resize_filter: str,
    width: Optional[int],
    height: Optional[int]
):
    """Convert an image dataset into a dataset archive usable with StyleGAN2 ADA PyTorch.

    The input dataset format is guessed from the --source argument:

    \b
    --source *_lmdb/                    Load LSUN dataset
    --source cifar-10-python.tar.gz     Load CIFAR-10 dataset
    --source train-images-idx3-ubyte.gz Load MNIST dataset
    --source path/                      Recursively load all images from path/
    --source dataset.zip                Recursively load all images from dataset.zip

    Specifying the output format and path:

    \b
    --dest /path/to/dir                 Save output files under /path/to/dir
    --dest /path/to/dataset.zip         Save output files into /path/to/dataset.zip

    The output dataset format can be either an image folder or an uncompressed zip archive.
    Zip archives makes it easier to move datasets around file servers and clusters, and may
    offer better training performance on network file systems.

    Images within the dataset archive will be stored as uncompressed PNG.
    Uncompresed PNGs can be efficiently decoded in the training loop.

    Class labels are stored in a file called 'dataset.json' that is stored at the
    dataset root folder.  This file has the following structure:

    \b
    {
        "labels": [
            ["00000/img00000000.png",6],
            ["00000/img00000001.png",9],
            ... repeated for every image in the datase
            ["00049/img00049999.png",1]
        ]
    }

    If the 'dataset.json' file cannot be found, the dataset is interpreted as
    not containing class labels.

    Image scale/crop and resolution requirements:

    Output images must be square-shaped and they must all have the same power-of-two
    dimensions.

    To scale arbitrary input image size to a specific width and height, use the
    --width and --height options.  Output resolution will be either the original
    input resolution (if --width/--height was not specified) or the one specified with
    --width/height.

    Use the --transform=center-crop or --transform=center-crop-wide options to apply a
    center crop transform on the input image.  These options should be used with the
    --width and --height options.  For example:

    \b
    python dataset_tool.py --source LSUN/raw/cat_lmdb --dest /tmp/lsun_cat \\
        --transform=center-crop-wide --width 512 --height=384
    """

    PIL.Image.init() # type: ignore

    if dest == '':
        ctx.fail('--dest output filename or directory must not be an empty string')

    return_as_filename = True
    num_files, input_iter = open_dataset(source, max_images=max_images, return_as_filename=return_as_filename)
    print(f"Found {num_files} files!")

    extend = make_extentions(extensions)

    archive_root_dir, save_bytes, close_dest = open_dest(dest, allow_exists)

    transform_image = make_transform(transform, width, height, resize_filter)

    dataset_attrs = None


    labels = []
    for idx, image in tqdm(enumerate(input_iter), total=num_files, position=0, leave=True):
        idx_str = f'{idx:08d}'
        fname_stem = Path(image["fname"]).stem
        archive_fname = fname_stem + ".png"
        #archive_fname = f'{idx_str[:5]}/img{idx_str}.png'

        archive_full_path = os.path.join(archive_root_dir, archive_fname)
        original_full_path = os.path.join(archive_root_dir, f"original_{archive_fname}")
        # Checks if the image is already in the path
        exists = os.path.exists(archive_full_path)
        original_exists = os.path.exists(original_full_path)

        if exists and original_exists: #If both exists, delete the original_<fileid>.png file
            pathlib.Path.unlink(pathlib.Path(archive_full_path))
            continue
        elif original_exists: #If original exists, assume the extensions do as well and continue
            continue
        elif exists:
            os.rename(archive_full_path, original_full_path) #Always convert to original
            continue


        img = image["img"]
        try:
            if return_as_filename:
                img = np.array(PIL.Image.open(img))
        except PIL.UnidentifiedImageError:
            print(f"Unidentified image")
            continue
        # Apply crop and resize.
        img = transform_image(img)

        # Transform may drop images.
        if img is None:
            continue

        imgs = extend(img)

        for name, img in imgs.items():
            # Error check to require uniform image attributes across
            # the whole dataset.
            channels = img.shape[2] if img.ndim == 3 else 1
            cur_image_attrs = {
                'width': img.shape[1],
                'height': img.shape[0],
                'channels': channels
            }
            if dataset_attrs is None:
                dataset_attrs = cur_image_attrs
                width = dataset_attrs['width']
                height = dataset_attrs['height']
                if width != height:
                    error(f'Image dimensions after scale and crop are required to be square.  Got {width}x{height}')
                if dataset_attrs['channels'] not in [1, 3]:
                    error('Input images must be stored as RGB or grayscale')
                if width != 2 ** int(np.floor(np.log2(width))):
                    error('Image width/height after scale and crop are required to be power-of-two')
            elif dataset_attrs != cur_image_attrs:
                err = [f'  dataset {k}/cur image {k}: {dataset_attrs[k]}/{cur_image_attrs[k]}' for k in dataset_attrs.keys()]
                error(f'Image {archive_fname} attributes must be equal across all images of the dataset.  Got:\n' + '\n'.join(err))

            # Save the image as an uncompressed PNG.
            img = PIL.Image.fromarray(img, { 1: 'L', 3: 'RGB' }[channels])
            image_bits = io.BytesIO()

            file_path = os.path.join(archive_root_dir, f"{name}_{archive_fname}")

            img.save(image_bits, format='png', compress_level=0, optimize=False)
            save_bytes(file_path, image_bits.getbuffer())

            labels.append([archive_fname, image['label']] if image['label'] is not None else None)

    metadata = {
        'labels': labels if all(x is not None for x in labels) else None
    }
    save_bytes(os.path.join(archive_root_dir, 'dataset.json'), json.dumps(metadata))
    close_dest()

#----------------------------------------------------------------------------

if __name__ == "__main__":
    print("Starting conversion")
    convert_dataset() # pylint: disable=no-value-for-parameter
    print("Finished")
    sys.exit(0)