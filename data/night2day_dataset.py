import os
from data.pix2pix_dataset import Pix2pixDataset
from data.image_folder import make_dataset
import json
import random
class Night2DayDataset(Pix2pixDataset):

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser = Pix2pixDataset.modify_commandline_options(parser, is_train)
        parser.set_defaults(preprocess_mode='fixed')
        parser.set_defaults(load_size=512)
        parser.set_defaults(crop_size=512)
        parser.set_defaults(display_winsize=512)
        parser.set_defaults(aspect_ratio=2.0)
        opt, _ = parser.parse_known_args()
        if hasattr(opt, 'num_upsampling_layers'):
            parser.set_defaults(num_upsampling_layers='more')
        return parser

    def get_paths(self, opt):
        croot = opt.croot   # content image: night
        sroot = opt.sroot   # style image: day
        dayJson = opt.dayJson
        # read night from folder
        c_image_dir = os.path.join(croot, "night_images")
        c_image_paths = sorted(make_dataset(c_image_dir, recursive=True))

        # read day from json file
        s_image_paths = []
        with open(dayJson) as json_file:
            s_image_paths_read = json.load(json_file)
            for _, img_path in s_image_paths_read.items():
                s_image_paths.append(os.path.join(sroot, img_path))

        # if opt.phase == 'train':
        #     s_image_paths = s_image_paths + s_image_paths   # more day images

        instance_paths = []

        # length = max(len(c_image_paths), len(s_image_paths))
        # c_image_paths = c_image_paths[:length]
        # s_image_paths = s_image_paths[:length]
        random.Random(0).shuffle(c_image_paths)
        random.Random(0).shuffle(s_image_paths)
        return c_image_paths, s_image_paths, instance_paths

    def paths_match(self, path1, path2):
        return True