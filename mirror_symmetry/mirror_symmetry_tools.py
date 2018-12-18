import os
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
        save_nii(symmetry_mask.astype(np.uint8), img_affine,
                 os.path.join(save_path, 'symmetry_mask'))
    if mirror_image:
        save_nii(mirror_images[0], img_affine, os.path.join(save_path,
                                                            'mirror_image_1'))
        save_nii(mirror_images[1], img_affine, os.path.join(save_path,
                                                            'mirror_image_2'))

    np.set_printoptions(precision=2)
    print('The detected symmetry plane is defined by the the following point '
          'normal pair (in voxel coordinates):\n'
          'point: ' + str(symmetry_plane['point']) + ',\n'
          'normal: ' + str(symmetry_plane['normal']))
    return


def get_symmetry_plane_from_transformation(img_shape, transformation_matrix,
                                           flip_axis=0):
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
    mirror_normal[0, np.abs(flip_axis)] = 1

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
    # direction to be positive (avoids, random orientation)
    if symmetry_normal[np.abs(flip_axis), 0] < 0:
        symmetry_normal = -symmetry_normal
    # consider sign of flip direction to orient normal correctly
    if flip_axis < 0:
        symmetry_normal = -symmetry_normal

    d = np.dot(mirror_centre, np.transpose(mirror_normal))

    symmetry_point = 0.5 * (np.dot(mat_R, 2*d*np.transpose(mirror_normal))
                            + np.transpose(np.matrix(t)))
    return mat2vec(symmetry_point), mat2vec(symmetry_normal)


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
    The image axis that corresponds to the given direction. The sign
    determines if the direction points from low to high values (+) or from
    high to low values (-).
    """
    if direction.upper() == 'R':
        dir_set = ['R', 'L']
    elif direction.upper() == 'L':
        dir_set = ['L', 'R']
    elif direction.upper() == 'A':
        dir_set = ['A', 'P']
    elif direction.upper() == 'P':
        dir_set = ['P', 'A']
    elif direction.upper() == 'S':
        dir_set = ['S', 'I']
    elif direction.upper() == 'I':
        dir_set = ['I', 'S']
    else:
        dir_set = ['R', 'L']
        print('Direction specified not in [R, L, A, P, S, I], R used instead.')

    orientations = np.array(nib.orientations.aff2axcodes(affine))

    axis = np.squeeze(np.argwhere([o in dir_set for o in orientations]))
    dir_idx = np.argwhere(orientations[axis] in dir_set)

    if dir_idx == 0:
        flip_direction = axis
    else:
        flip_direction = -axis

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

    img_mirrored = np.flip(img, np.abs(flip_axis))

    warped_mirrored, _, affine_mat = register_nifty(img, img_mirrored, affine)

    point, normal = get_symmetry_plane_from_transformation(img_size,
                                                           affine_mat,
                                                           flip_axis)
    symmetry_mask = None
    mirrored_1 = None
    mirrored_2 = None

    if create_mask or mirror_images:
        symmetry_mask = create_masks_from_plane(point, normal, img_size)
        symmetry_mask = symmetry_mask.astype(bool)
    if mirror_images:
        mirrored_1 = np.copy(img)
        mirrored_2 = np.copy(img)
        mirrored_1[symmetry_mask] = warped_mirrored[symmetry_mask]
        inv_mask = np.logical_not(symmetry_mask)
        mirrored_2[inv_mask] = warped_mirrored[inv_mask]
    return {'normal': normal, 'point': point}, symmetry_mask, [mirrored_1,
                                                               mirrored_2]


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
    symmetry_mask = create_masks_from_plane(symmetry_plane['point'],
                                            symmetry_plane['normal'], img_size)
    return symmetry_mask


def create_masks_from_plane(point, normal, shape):
    """
    Create a binary mask of given size based on a plane defined by its
    normal and a point on the plane (in voxel coordinates).

    Parameters
    ----------
    point: Point on the plane (in voxel coordinates).
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

    distance_from_plane = np.dot((position - np.transpose(point)), normal)
    distance_vol = np.array(distance_from_plane).reshape((shape[0],
                                                          shape[1],
                                                          shape[2]),
                                                         order='F')

    binary_mask = np.empty(distance_vol.shape, dtype=np.float32)
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
    # TODO: consider using simpleITK for a registration that is easier
    # TODO: available and has less overhead than this NiftyReg hack
    save_nii(ref, affine, '_tmp_ref.nii.gz')
    save_nii(flo, affine, '_tmp_flo.nii.gz')

    cmd_string = 'reg_aladin -ref _tmp_ref.nii.gz -flo _tmp_flo.nii.gz'
    cmd_string += ' -rigOnly -res _tmp_warped.nii.gz -voff'
    cmd_string += ' -aff _tmp_affine_matrix.txt'

    call(cmd_string.split(' '))
    warped_nii = nib.load('_tmp_warped.nii.gz')
    warped_img = warped_nii.get_fdata()
    warped_affine = warped_nii.affine

    with open('_tmp_affine_matrix.txt', 'r') as file:
        rigid_mat = [[float(num) for num in line.split(' ')] for line in file]

    delete_files('_tmp_')

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


def delete_files(prefix='_tmp_', path=None):
    """
    Convenience function to delete files with given prefix from a folder.

    Parameters
    ----------
    prefix: Common prefix for the files to be deleted.
    path: Path to a folder that is searched for files with the given prefix.

    """
    if path is None:
        path = os.getcwd()
    for file in os.listdir(path):
        if file.startswith(prefix):
            os.remove(file)
    return


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
