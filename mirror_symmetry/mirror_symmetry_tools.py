import os
from math import copysign
from tempfile import TemporaryDirectory
from subprocess import call

import numpy as np
import nibabel as nib


def main(img_path, save_path=None, direction='R', create_mask=None,
         mirror_image=None):
    img_nii = nib.load(img_path)
    img_data = img_nii.get_fdata()
    img_affine = img_nii.get_qform()

    symmetry_plane, symmetry_mask, mirror_images = \
        get_mirror_symmetry_plane(img_data, img_affine, direction,
                                  create_mask, mirror_image)

    if save_path is None:
        save_path = os.getcwd()
    elif save_path.startswith('.'):
        save_path = os.getcwd() + save_path[1:]

    if create_mask:
        if direction.upper() == 'R':
            mask_path = os.path.join(save_path, 'right_symmetry_mask')
        elif direction.upper() == 'L':
            mask_path = os.path.join(save_path, 'left_symmetry_mask')
        elif direction.upper() == 'A':
            mask_path = os.path.join(save_path, 'anterior_symmetry_mask')
        elif direction.upper() == 'P':
            mask_path = os.path.join(save_path, 'posterior_symmetry_mask')
        elif direction.upper() == 'S':
            mask_path = os.path.join(save_path, 'superior_symmetry_mask')
        elif direction.upper() == 'I':
            mask_path = os.path.join(save_path, 'inferior_symmetry_mask')
        else:
            mask_path = os.path.join(save_path, 'symmetry_mask')
        save_nii(symmetry_mask.astype(np.uint8), img_affine, mask_path)
    if mirror_image:
        if direction.upper() == 'R':
            mirror_path_1 = os.path.join(save_path, 'mirror_image_left')
            mirror_path_2 = os.path.join(save_path, 'mirror_image_right')
        elif direction.upper() == 'L':
            mirror_path_1 = os.path.join(save_path, 'mirror_image_right')
            mirror_path_2 = os.path.join(save_path, 'mirror_image_left')
        elif direction.upper() == 'A':
            mirror_path_1 = os.path.join(save_path, 'mirror_image_posterior')
            mirror_path_2 = os.path.join(save_path, 'mirror_image_anterior')
        elif direction.upper() == 'P':
            mirror_path_1 = os.path.join(save_path, 'mirror_image_anterior')
            mirror_path_2 = os.path.join(save_path, 'mirror_image_posterior')
        elif direction.upper() == 'S':
            mirror_path_1 = os.path.join(save_path, 'mirror_image_inferior')
            mirror_path_2 = os.path.join(save_path, 'mirror_image_superior')
        elif direction.upper() == 'I':
            mirror_path_1 = os.path.join(save_path, 'mirror_image_superior')
            mirror_path_2 = os.path.join(save_path, 'mirror_image_inferior')
        else:
            mirror_path_1 = os.path.join(save_path, 'mirror_image_1')
            mirror_path_2 = os.path.join(save_path, 'mirror_image_2')
        save_nii(mirror_images[0], img_affine, mirror_path_1)
        save_nii(mirror_images[1], img_affine, mirror_path_2)

    img_affine_2 = np.copy(img_affine)
    img_affine_2[0:3, 3] = 0
    normal_world = mat2vec(voxel2world(symmetry_plane['normal'], img_affine_2))
    point_world = mat2vec(voxel2world(symmetry_plane['point'], img_affine))

    np.set_printoptions(precision=2)
    print('The detected symmetry plane is described by the following '
          'information: Point on the plane, normal and distance to the '
          'origin (Hessian normal form: n x = -d).\n'
          'In voxel coordinates:\n'
          'point:    ' + str(symmetry_plane['point']) + ',\n'
          'normal:   ' + str(symmetry_plane['normal']) + ',\n'
          'distance: ' + str(symmetry_plane['dist']) + '.\n'
          'In world coordinates:\n'
          'point:    ' + str(point_world) + ',\n'
          'normal:   ' + str(normal_world) + ',\n'
          'distance: ' + str(symmetry_plane['dist']) + '.')
    return


def voxel2world(coord, affine):
    """
    Convert voxel coordinates into world coordinates.

    Parameters
    ----------
    coord: Coordinates of a single point as an array/list, or multiple
        points as a list of arrays/lists, or as a matrix with one point per
        row.
    affine: Affine matrix determining the world coordinates.

    Returns
    -------

    Matrix of world coordinates with one point per row.
    """
    coord = np.matrix(coord)
    stacked = np.column_stack((coord[:, 0],
                               coord[:, 1],
                               coord[:, 2],
                               np.ones(np.shape(coord)[0])))
    in_world = np.dot(affine, np.transpose(stacked))
    return np.transpose(in_world[:3, :])


def project_point_on_plane(plane_normal, plane_point, point):
    """
    Project a point onto a plane given by the normal and a point.
    q = p - <(p-o),n> * n

    Parameters
    ----------
    plane_normal: Normal vector of the plane.
    plane_point: Point on the plane.
    point: Arbitrary point that is to be projected onto the plane.

    Returns
    -------
    Coordinates of the point projected onto the plane.
    """
    return point - np.dot((point - plane_point), plane_normal) * plane_normal


def get_symmetry_plane_from_transformation(img_shape, transformation_matrix,
                                           flip_axis=1):
    """
    Compute the mirror symmetry plane of an image volume based on an affine
    transformation matrix. The symmetry plane normal is determined by
    finding the eigenvector with the eigenvalue of -1 (meaning this vector is
    mapped on its negative, i.e. its mirror vector). The implementation is
    based on Cicconet et al. (2017):
    https://ieeexplore.ieee.org/document/8265416
    doi: 10.1109/ICCVW.2017.206

    Parameters
    ----------
    img_shape: Shape of the image.
    transformation_matrix: 4x4 affine transformation matrix describing the
        registration of an image volume to its mirrored volume that was
        flipped in flip_axis.
    flip_axis: Determines the flipping axis used to create the mirrored
        image that is used for the registration. Sign indicates in which
        direction the symmetry_normal will point.

    Returns
    -------
    symmetry_point: Point on the symmetry plane in voxel coordinates.
    symmetry_normal: Normal vector of symmetry plane in voxel coordinates.
    """
    mirror_centre = np.round(img_shape / 2.0)
    mirror_centre = mirror_centre.astype(int)

    mirror_normal = np.zeros([1, 3])
    mirror_normal[0, np.abs(flip_axis) - 1] = 1

    # mirroring matrix (S_v)
    mat_Sv = np.eye(3) - 2*np.matmul(np.transpose(mirror_normal),
                                     mirror_normal)

    mat_R = transformation_matrix[0:3, 0:3]
    t = transformation_matrix[0:3, 3]

    mat = np.dot(mat_Sv, np.transpose(mat_R))
    mat = np.asmatrix(mat)

    eigen_values, eigen_vectors = np.linalg.eigh(mat)

    # extract the eigen value closest to -1
    check = np.abs(eigen_values + 1.0)
    symmetry_idx = np.argmin(check)
    symmetry_normal = eigen_vectors[:, symmetry_idx]

    # force the component of the normal which corresponds to the symmetry
    # direction to point in the direction of the sign of flip_axis
    if copysign(1, symmetry_normal[np.abs(flip_axis)-1]) \
            != copysign(1, flip_axis):
        symmetry_normal = -symmetry_normal

    d = np.dot(mirror_centre, np.transpose(mirror_normal))

    symmetry_point = 0.5 * (np.dot(mat_R, 2*d*np.transpose(mirror_normal))
                            + np.transpose(np.matrix(t)))
    symmetry_point = mat2vec(symmetry_point)
    symmetry_normal = mat2vec(symmetry_normal)

    centre_on_plane = project_point_on_plane(symmetry_normal, symmetry_point,
                                             mirror_centre)
    # Hessian normal form: n x = -p
    hnf_normal = symmetry_normal/np.linalg.norm(symmetry_normal)
    hnf_dist = - np.dot(hnf_normal, symmetry_point)
    return hnf_normal, hnf_dist, centre_on_plane


def mat2vec(mat):
    """
    Convenience function to convert a 1D numpy matrix into an numpy array.

    Parameters
    ----------
    mat: 1D numpy matrix

    Returns
    -------

    A numpy array containing the data of the input matrix.
    """
    return np.squeeze(np.array(mat))


def get_axis_from_direction(affine, direction='R'):
    """
    Determine image axis corresponding to a given world direction.

    Parameters
    ----------
    affine: Affine matrix determining the world coordinates of the volume.
    direction: Direction in world terms (left, right, anterior,
        posterior, superior, inferior) specified by the first letter, e.g. 'R'.

    Returns
    -------
    The image axis that corresponds to the given direction [-3,-2,-1,1,2,3].
    The sign determines if the direction points from low to high values (+)
    or from high to low values (-).
    """
    if direction.upper() == 'R' or direction.upper() == 'L':
        dir_set = ['R', 'L']
    elif direction.upper() == 'A' or direction.upper() == 'P':
        dir_set = ['A', 'P']
    elif direction.upper() == 'S' or direction.upper() == 'I':
        dir_set = ['S', 'I']
    else:
        dir_set = ['R', 'L']
        print('Direction specified not in [R, L, A, P, S, I], R used instead.')

    orientations = np.array(nib.orientations.aff2axcodes(affine))
    axis = np.squeeze(np.argwhere([o in dir_set for o in orientations]))

    if direction.upper() == orientations[axis]:
        flip_direction = axis+1
    else:
        flip_direction = -(axis+1)

    return flip_direction


def get_mirror_symmetry_plane(img, affine, direction='R',
                              create_mask=None, mirror_images=False):
    """
    Compute mirror symmetry plane of an image volume based on image
    registration following the strategy of Cicconet et al. (2017):
    https://ieeexplore.ieee.org/document/8265416
    doi: 10.1109/ICCVW.2017.206
    The image volume is mirrored along an image axis and (rigidly) registered
    to the original image. The mirror direction can be specified and
    improves registration performance when it is close to the symmetry
    plane. Eigenvalue analysis of the combined mirror- and transformation-
    matrix determines the symmetry plane.

    Parameters
    ----------
    img: Image volume (numpy ndarray).
    affine: Affine matrix determining the world coordinates of the volume.
    direction: The direction of expected symmetry, e.g. when an image has a
        symmetry close to the left-right direction, either "left"/"l" or
        "right"/"r" can be specified. If the mask is saved, the value 1 will
        correspond to the direction specified here. Expected values: left,
        right, anterior, posterior, superior, inferior and the corresponding
        first letters.
    create_mask: Flag to indicate whether a binary mask separating the image
        along the symmetry plane should be created.
    mirror_images: Flag to indicate whether two images created by mirroring one
        side across the symmetry plane should be created.

    Returns
    -------
    A list of following three values:
    A dictionary containing the normal of the symmetry plane ('normal') and
    a point on the plane ('point') in voxel coordinates.
    A binary mask separating the image along the symmetry plane.
    A list of two images mirrored along the symmetry plane.
    """
    img_size = np.array(np.shape(img))
    flip_axis = get_axis_from_direction(affine, direction)

    img_mirrored = np.flip(img, np.abs(flip_axis)-1)

    warped_mirrored, _, affine_mat = register_nifty(img, img_mirrored, affine)

    normal, dist, point = get_symmetry_plane_from_transformation(img_size,
                                                                 affine_mat,
                                                                 flip_axis)
    symmetry_mask = None
    mirrored_1 = None
    mirrored_2 = None

    if create_mask or mirror_images:
        symmetry_mask = create_masks_from_plane(normal, dist, img_size)
        symmetry_mask = symmetry_mask.astype(bool)
    if mirror_images:
        mirrored_1 = np.copy(img)
        mirrored_2 = np.copy(img)
        mirrored_1[symmetry_mask] = warped_mirrored[symmetry_mask]
        inv_mask = np.logical_not(symmetry_mask)
        mirrored_2[inv_mask] = warped_mirrored[inv_mask]
    return {'normal': normal, 'dist': dist, 'point': point}, \
        symmetry_mask, [mirrored_1, mirrored_2]


def create_symmetry_mask(img, affine, direction='R'):
    """
    Create a binary mask splitting the image into the two (most) mirror
    symmetric regions. The symmetry plane is detected using a registration
    based method.

    Parameters
    ----------
    img: The 3D image volume (ndarray).
    affine: Affine matrix determining the world coordinates of the image.
    direction: Determines the flipping direction used to create a
        mirrored image that is used for the registration.

    Returns
    -------
    The binary symmetry mask of the same size as the input image.
    """
    img_size = np.array(np.shape(img))

    symmetry_plane, _, _ = get_mirror_symmetry_plane(img, affine,
                                                     direction)
    symmetry_mask = create_masks_from_plane(symmetry_plane['normal'],
                                            symmetry_plane['dist'], img_size)
    return symmetry_mask


def create_masks_from_plane(normal, dist, shape):
    """
    Create a binary mask of given size based on a plane defined by its
    normal and a point on the plane (in voxel coordinates).

    Parameters
    ----------
    dist: Distance of the plane to the origin (in voxel coordinates).
    normal: Normal of the plane (in voxel coordinates).
    shape: Shape of the mask that will be created.

    Returns
    -------
    Binary mask of specified shape split in two by the given plane.
    """
    grid_x, grid_y, grid_z = np.meshgrid(range(shape[0]),
                                         range(shape[1]),
                                         range(shape[2]),
                                         indexing='ij')

    position = np.column_stack((grid_x.ravel(order='F'),
                                grid_y.ravel(order='F'),
                                grid_z.ravel(order='F')))

    # distance_from_plane = np.dot((position - np.transpose(point)), normal)
    distance_from_plane = np.dot(position, normal) + dist
    distance_vol = np.array(distance_from_plane).reshape((shape[0],
                                                          shape[1],
                                                          shape[2]),
                                                         order='F')

    binary_mask = np.empty(distance_vol.shape, dtype=np.uint8)
    binary_mask[:, :, :] = distance_vol[:, :, :] >= 0
    return binary_mask


def register_nifty(ref, flo, affine):
    """
    Register two 3D image volumes using NifyReg.

    Parameters
    ----------
    ref: Reference/ fixed image.
    flo: Floating/ moving image.
    affine: Affine matrix determining the world coordinates of the image.

    Returns
    -------
    warped_img:
    warped_affine:
    rigid_voxel:
    """
    # TODO: consider using simpleITK for a registration that is available as
    # pip install. Also NiftyReg is limited in capture range of rotation
    # which can be a problem detecting symmetry planes that are far from the
    # image axes.

    with TemporaryDirectory(prefix='tmp_mirror_symmetry_', dir=os.getcwd()) \
            as tmp_dir:
        save_nii(ref, affine, os.path.join(tmp_dir, '_tmp_ref.nii.gz'))
        save_nii(flo, affine, os.path.join(tmp_dir, '_tmp_flo.nii.gz'))

        cmd_string = 'reg_aladin'
        cmd_string += ' -ref ' + os.path.join(tmp_dir, '_tmp_ref.nii.gz')
        cmd_string += ' -flo ' + os.path.join(tmp_dir, '_tmp_flo.nii.gz')
        cmd_string += ' -res ' + os.path.join(tmp_dir, '_tmp_warped.nii.gz')
        cmd_string += ' -aff ' + os.path.join(tmp_dir, '_tmp_aff_matrix.txt')
        cmd_string += ' -rigOnly -comi -voff'
        cmd_string += ' -ln 4 -lp 3'
        call(cmd_string.split(' '))

        # input('wait a sec...')

        warped_nii = nib.load(os.path.join(tmp_dir, '_tmp_warped.nii.gz'))
        warped_img = warped_nii.get_fdata()
        warped_affine = warped_nii.affine

        with open(os.path.join(tmp_dir, '_tmp_aff_matrix.txt'), 'r') as aff:
            rigid_mat = [[float(num) for num in line.split(' ')]
                         for line in aff]

    rigid_voxel = transformation_world2voxel(rigid_mat, affine)

    return warped_img, warped_affine, np.array(rigid_voxel)


def transformation_world2voxel(transformation, affine):
    """
    Convenience function to convert a transformation matrix vom world to
    voxel coordinates.

    Parameters
    ----------
    transformation: (affine or rigid)transformation matrix of shape 4x4.
    affine: Affine matrix determining the world coordinates of the volume
        which is to be transformed by the matrix.

    Returns
    -------
    The 4x4 transformation matrix in voxel coordinates.
    """
    trans_voxel = np.dot(np.linalg.inv(affine), np.dot(transformation, affine))
    return np.linalg.inv(trans_voxel)


def save_nii(img, affine, path):
    """
    Saves an image with affine matrix to a compressed nifti file (*.nii.gz)

    Parameters
    ----------
    img: 2D or 3D ndarray to be saved
    affine: affine matrix (converts voxel to world coordinates)
    path: file path including file name, .nii.gz file extension can be
        omitted
    """
    path = enforce_file_extension(path, '.nii.gz')
    img_nii = nib.Nifti1Image(img, affine)
    nib.save(img_nii, path)
    return


def enforce_file_extension(file, extension):
    """
    Returns the given string (file name or full path) with the given extension.
    string had no file extension .extension will be appended. If it had another
    extension it is changed to the given extension. If it had the given
    extension already the unchanged string is returned.

    Parameters
    ----------
    file: File name or full path. Can include a file extension .*
    extension: File extension the given file name will be returned with.

    Returns
    -------
    The given file with the given extension.
    """
    if not extension.startswith('.'):
        extension = '.' + extension
    split_str = file.split('.', 1)
    if (len(split_str) != 2) or (split_str[-1] != extension[1:]):
        file = split_str[0] + extension
    return file
