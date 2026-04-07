import os
import math
import numpy as np
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset
from PIL import Image


def _intensity_centroid(arr):
    """Intensity-weighted centroid (cx, cy)."""
    h, w = arr.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    W = arr.astype(np.float32)
    s = float(W.sum())
    if s < 1e-12:
        return w / 2.0, h / 2.0
    cx = float((W * xx).sum() / s)
    cy = float((W * yy).sum() / s)
    return cx, cy


def _principal_axis_angle(arr):
    """Principal axis angle (degrees) from intensity-weighted inertia tensor."""
    h, w = arr.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    W = arr.astype(np.float32)
    s = float(W.sum())
    if s < 1e-12:
        return 0.0
    cx = float((W * xx).sum() / s)
    cy = float((W * yy).sum() / s)
    x = xx - cx
    y = yy - cy
    Sxx = float((W * x * x).sum())
    Syy = float((W * y * y).sum())
    Sxy = float((W * x * y).sum())
    angle_rad = 0.5 * math.atan2(2.0 * Sxy, Sxx - Syy)
    return math.degrees(angle_rad)


def _disambiguate_axis(arr_centered, angle_deg):
    """Resolve 180-degree ambiguity: heavier side points to positive direction."""
    h, w = arr_centered.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy = w / 2.0, h / 2.0
    x = xx - cx
    y = yy - cy
    W = arr_centered.astype(np.float32)
    theta = math.radians(angle_deg)
    proj = x * math.cos(theta) + y * math.sin(theta)
    if float((W * (proj <= 0)).sum()) > float((W * (proj > 0)).sum()):
        angle_deg += 180.0
    return angle_deg


def _center_by_centroid(arr):
    """Translate so that intensity centroid sits at image center."""
    h, w = arr.shape
    cx, cy = _intensity_centroid(arr)
    dx = (w / 2.0) - cx
    dy = (h / 2.0) - cy
    im = Image.fromarray(np.clip(arr * 255.0, 0, 255).astype(np.uint8))
    im_t = im.transform(im.size, Image.AFFINE, (1, 0, dx, 0, 1, dy),
                        resample=Image.Resampling.BICUBIC)
    return np.asarray(im_t, dtype=np.float32) / 255.0


def _compute_align_angle(arr):
    """Compute alignment rotation angle from a grayscale array (0-1 float)."""
    arr_c = _center_by_centroid(arr)
    angle = _principal_axis_angle(arr_c)
    angle = _disambiguate_axis(arr_c, angle)
    # Normalize to [-180, 180)
    while angle >= 180.0:
        angle -= 360.0
    while angle < -180.0:
        angle += 360.0
    return angle


def _apply_co_alignment(pil_img, angle_deg):
    """Rotate a PIL image by -angle_deg to align its principal axis horizontally."""
    return pil_img.rotate(-angle_deg, resample=Image.Resampling.BICUBIC, expand=True)


class AlignedDataset(BaseDataset):
    """A dataset class for paired image dataset.

    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)  # get the image directory
        self.AB_paths = sorted(make_dataset(self.dir_AB, opt.max_dataset_size))  # get image paths
        assert(self.opt.load_size >= self.opt.crop_size)   # crop_size should be smaller than the size of loaded image
        self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc
        self.use_co = getattr(opt, 'co', False)

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """
        # read a image given a random integer index
        AB_path = self.AB_paths[index]
        AB = Image.open(AB_path).convert('RGB')
        # split AB image into A and B
        w, h = AB.size
        w2 = int(w / 2)
        A = AB.crop((0, 0, w2, h))
        B = AB.crop((w2, 0, w, h))

        # centroid-orientation alignment: compute angle from A, rotate both A and B
        if self.use_co:
            arr_a = np.asarray(A.convert('L'), dtype=np.float32) / 255.0
            angle = _compute_align_angle(arr_a)
            A = _apply_co_alignment(A, angle)
            B = _apply_co_alignment(B, angle)

        # apply the same transform to both A and B
        transform_params = get_params(self.opt, A.size)
        A_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1))
        B_transform = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1))

        A = A_transform(A)
        B = B_transform(B)

        return {'A': A, 'B': B, 'A_paths': AB_path, 'B_paths': AB_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.AB_paths)
