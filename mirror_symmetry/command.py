import argparse as ap
import os
from .mirror_symmetry_tools import main


class ReadableDir(ap.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        r_dir = values

        if not os.path.exists(r_dir):
            try:
                os.makedirs(r_dir)
            except os.error:
                raise ap.ArgumentTypeError(
                        'ReadableDir:Could not create dir: {0}'.format(r_dir))
        if not os.path.isdir(r_dir):
            raise ap.ArgumentTypeError(
                    'ReadableDir:{0} is not a valid path'.format(r_dir))
        if os.access(r_dir, os.R_OK):
            setattr(namespace, self.dest, r_dir)
        else:
            raise ap.ArgumentTypeError(
                    'ReadableDir:{0} is not a readable dir'.format(r_dir))


def process():
    parser = ap.ArgumentParser(description='Detect mirror symmetry plane. The '
                                           'symmetry plane determined in '
                                           'point-normal form is printed. '
                                           'Optional binary mask that splits '
                                           'the image into the two symmetric '
                                           'regions can be saved. Optional '
                                           'two symmetric images can be '
                                           'created by mirroring the sides '
                                           'across the symmetry plane.')

    parser.add_argument('image',
                        help='Nifti image to be processed.')
    parser.add_argument('--save_path', '-p', default=None, action=ReadableDir,
                        help='Path to folder in which files are saved. '
                             'Folder will be created if it does not exist.')
    parser.add_argument('--direction', '-d', default='R',
                        help='The direction of expected symmetry, e.g. when '
                             'an image has a symmetry close to the '
                             'left-right direction, either "left"/"l" or '
                             '"right"/"r" can be specified. If the mask is '
                             'saved, the value 1 will correspond to the '
                             'direction specified here. Expected values: '
                             'left, right, anterior, posterior, superior, '
                             'inferior and the corresponding first letters, '
                             'either lower case or upper case.')

    parser.add_argument('--create_mask', '-c', action='store_true',
                        help='Set this flag to save a binary symmetry mask.')
    parser.add_argument('--mirror_image', '-m', action='store_true',
                        help='Set this flag to save two images created by '
                             'mirroring one side across the symmetry plane.')
    arguments = parser.parse_args()

    main(arguments.image, arguments.save_path, arguments.direction[0],
         arguments.create_mask, arguments.mirror_image)


if __name__ == "__main__":
    process()
