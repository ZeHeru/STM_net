import os
from data.base_dataset import BaseDataset, get_transform
from PIL import Image


class SingleDataset(BaseDataset):
    """A dataset class for single image inference.

    This dataset loads a single image from a given path and applies the necessary transforms.
    It's designed for inference on individual images without requiring paired datasets.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        
        # For single image inference, we expect the image path to be in opt.dataroot
        # or we can use a specific image path if provided
        if hasattr(opt, 'single_image_path') and opt.single_image_path:
            self.image_path = opt.single_image_path
        else:
            # If no specific path, assume dataroot contains the image file
            self.image_path = getattr(opt, 'dataroot', opt.single_image_path)
            
        # Verify the image exists
        if not os.path.isfile(self.image_path):
            raise FileNotFoundError(f"Image file not found: {self.image_path}")
            
        self.input_nc = opt.input_nc
        self.output_nc = opt.output_nc
        
        # Get transform for the input image
        self.transform = get_transform(opt, grayscale=(self.input_nc == 1))

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index -- index of the image (always 0 for single image)

        Returns a dictionary that contains A, B, A_paths, B_paths
            A (tensor) -- the input image tensor
            B (tensor) -- same as A (for compatibility with paired models)
            A_paths (str) -- image path
            B_paths (str) -- same as A_paths (for compatibility)
        """
        # Load the image
        A = Image.open(self.image_path).convert('RGB')
        
        # Apply transform
        A = self.transform(A)
        
        # For single image inference, we use the same image for both A and B
        # This maintains compatibility with models that expect paired data
        return {'A': A, 'B': A.clone(), 'A_paths': self.image_path, 'B_paths': self.image_path}

    def __len__(self):
        """Return the total number of images in the dataset (always 1 for single image)."""
        return 1

    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase

        Returns:
            the modified parser.
        """
        parser.add_argument('--single_image_path', type=str, default='', 
                           help='path to the single image file for inference')
        return parser