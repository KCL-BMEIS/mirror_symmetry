from argparse import ArgumentParser
from .mirror_symmetry_tools import main


def process():
    parser = ArgumentParser(description='Detect mirror symmetry plane. The '
                                        'symmetry plane determined in '
                                        'point-normal form is printed. '
                                        'Optional binary mask that splits '
                                        'the image into the two symmetric '
                                        'regions can be saved. Optional '
                                        'two symmetric images can be created '
                                        'by mirroring the sides across the '
                                        'symmetry plane.')

    parser.add_argument('image',
                        help='Nifti image to be processed')
    parser.add_argument('--save_path', '-p', default=None,
                        help='path to folder in which files are saved.')
    parser.add_argument('--flip_direction', '-d', default=0,
                        help='The index of the axis used as flipping '
                             'direction while initialising the registration '
                             'based symmetry detection. Useful to be close '
                             'to normal of expected symmetry plane.')

    parser.add_argument('--create_mask', '-c', action='store_true',
                        help='Set this flag to save a binary symmetry mask.')
    parser.add_argument('--mirror_image', '-m', action='store_true',
                        help='Set this flag to save two images created by '
                             'mirroring one side across the symmetry plane.')
    arguments = parser.parse_args()

    main(arguments.image, arguments.save_path, arguments.flip_direction,
         arguments.create_mask, arguments.mirror_image)


if __name__ == "__main__":
    process()
