
import numpy as np
import matplotlib.pyplot as plt


### ESSENTIAL MATRIX ###
    
def get_cross_product_matrix(vector):
    '''
    The cross product of two vectors can be represented as a matrix multiplication.
    a x b = [a']b,
    where 
    a = [a1, a2, a3] and
    a' = [[0, -a3, a2],
          [a3, 0, -a1],
          [-a2, a1, 0]]
    '''
    A = np.zeros((3, 3))
    a1, a2, a3 = vector
    A[0][1] = -a3
    A[0][2] = a2
    A[1][0] = a3
    A[1][2] = -a1
    A[2][0] = -a2
    A[2][1] = a1
    
    return A

def to_hg_coords(points):
    '''
    Convert the points from euclidean coordinates to homogeneous coordinates
    '''
    points = np.concatenate((points, np.ones((1, points.shape[1]))), axis=0)
    return points

def to_eucld_coords(points_hg):
    '''
    Convert the points from homogeneous coordinates to euclidean coordinates
    '''
    z = points_hg[-1,:]
    points = points_hg[:2,:]/z
    return points

def is_vectors_close(v1, v2):
    '''
    check if two vectors are close to each other
    '''
    v1 = v1.reshape(-1)
    v2 = v2.reshape(-1)
    assert len(v1) == len(v2)
    assert np.isclose(v1, v2).sum() == len(v1)

def plot_line(coeffs, xlim):
    '''
    Given the coefficients a, b, c of the ax + by + c = 0, 
    plot the line within the given x limits.
    ax + by + c = 0 => y = (-ax - c) / b
    '''
    a, b, c = coeffs
    x = np.linspace(xlim[0], xlim[1], 100)
    y = (a * x + c) / -b
    return x, y

### FUNDAMENTAL MATRIX ###

def show_matching_result(img1, img2, img1_pts, img2_pts):
    '''
    plot the images and their corresponding matching points
    '''
    fig = plt.figure(figsize=(8, 8))
    plt.imshow(np.hstack((img1, img2)), cmap="gray")
    for p1, p2 in zip(img1_pts, img2_pts):
        plt.scatter(p1[0], p1[1], s=35, edgecolors='r', facecolors='none')
        plt.scatter(p2[0] + img1.shape[1], p2[1], s=35, edgecolors='r', facecolors='none')
        plt.plot([p1[0], p2[0] + img1.shape[1]], [p1[1], p2[1]])
    plt.show()

def compute_fundamental_matrix_normalized(points1, points2):
    '''
    Normalize points by calculating the centroid, subtracting 
    it from the points and scaling the points such that the distance 
    from the origin is sqrt(2)
    
    Parameters
    ------------
    points1, points2 - array with shape [n, 3]
        corresponding points in images represented as 
        homogeneous coordinates
    '''
    # validate points
    assert points1.shape[0] == points2.shape[0], "no. of points don't match"
    
    # compute centroid of points
    c1 = np.mean(points1, axis=0)
    c2 = np.mean(points2, axis=0)
    
    # compute the scaling factor
    s1 = np.sqrt(2 / np.mean(np.sum((points1 - c1) ** 2, axis=1)))
    s2 = np.sqrt(2 / np.mean(np.sum((points2 - c2) ** 2, axis=1)))
    
    # compute the normalization matrix for both the points
    T1 = np.array([
        [s1, 0, -s1 * c1[0]],
        [0, s1, -s1 * c1[1]],
        [0, 0 ,1]
    ])
    T2 = np.array([
        [s2, 0, -s2 * c2[0]],
        [0, s2, -s2 * c2[1]],
        [0, 0, 1]
    ])
    
    # normalize the points
    points1_n = T1 @ points1.T
    points2_n = T2 @ points2.T
    
    # compute the normalized fundamental matrix
    F_n = compute_fundamental_matrix(points1_n.T, points2_n.T)
    
    # de-normalize the fundamental
    return T2.T @ F_n @ T1

def compute_fundamental_matrix(points1, points2):
    '''
    Compute the fundamental matrix given the point correspondences
    
    Parameters
    ------------
    points1, points2 - array with shape [n, 3]
        corresponding points in images represented as 
        homogeneous coordinates
    '''
    # validate points
    assert points1.shape[0] == points2.shape[0], "no. of points don't match"
    
    u1 = points1[:, 0]
    v1 = points1[:, 1]
    u2 = points2[:, 0]
    v2 = points2[:, 1]
    one = np.ones_like(u1)
    
    # construct the matrix 
    # A = [u2.u1, u2.v1, u2, v2.u1, v2.v1, v2, u1, v1, 1] for all the points
    # stack columns
    A = np.c_[u1 * u2, v1 * u2, u2, u1 * v2, v1 * v2, v2, u1, v1, one]
    
    # peform svd on A and find the minimum value of |Af|
    U, S, V = np.linalg.svd(A, full_matrices=True)
    f = V[-1, :]
    F = f.reshape(3, 3) # reshape f as a matrix
    
    # constrain F
    # make rank 2 by zeroing out last singular value
    U, S, V = np.linalg.svd(F, full_matrices=True)
    S[-1] = 0 # zero out the last singular value
    F = U @ np.diag(S) @ V # recombine again
    return F

def plot_epipolar_lines(img1, img2, points1, points2, show_epipole=False):
    '''
    Given two images and their corresponding points, compute the fundamental matrix 
    and plot epipole and epipolar lines
    
    Parameters
    ------------
    img1, img2 - array with shape (height, width)
        grayscale images with only two channels
    points1, points2 - array with shape [n, 3]
        corresponding points in images represented as 
        homogeneous coordinates
    show_epipole - boolean
        whether to compute and plot the epipole or not
    '''   
    
    # get image size
    h, w = img1.shape
    n = points1.shape[0]
    # validate points
    if points2.shape[0] != n:
        raise ValueError("No. of points don't match")
    
    # compute the fundamental matrix
    F = compute_fundamental_matrix_normalized(points1, points2)
    
    # configure figure
    nrows = 2
    ncols = 1
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6, 8))
    
    # plot image 1
    ax1 = axes[0]
    ax1.set_title("Image 1")
    ax1.imshow(img1, cmap="gray")
    
    # plot image 2
    ax2 = axes[1]
    ax2.set_title("Image 2")
    ax2.imshow(img2, cmap="gray")
    
    # plot epipolar lines
    for i in range(n):
        p1 = points1.T[:, i]
        p2 = points2.T[:, i]
        
        # Epipolar line in the image of camera 1 given the points in the image of camera 2
        coeffs = p2.T @ F
        x, y = plot_line(coeffs, (-1500, w)) # limit hardcoded for this image. please change
        ax1.plot(x, y, color="orange")
        ax1.scatter(*p1.reshape(-1)[:2], color="blue")

        # Epipolar line in the image of camera 2 given the points in the image of camera 1
        coeffs = F @ p1
        x, y = plot_line(coeffs, (0, 2800)) # limit hardcoded for this image. please change
        ax2.plot(x, y, color="orange")
        ax2.scatter(*p2.reshape(-1)[:2], color="blue")
        
    if show_epipole:
        # compute epipole
        e1 = compute_epipole(F)
        e2 = compute_epipole(F.T)
        # plot epipole
        ax1.scatter(*e1.reshape(-1)[:2], color="red")
        ax2.scatter(*e2.reshape(-1)[:2], color="red")
    else:
        # set axes limits
        ax1.set_xlim(0, w)
        ax1.set_ylim(h, 0)
        ax2.set_xlim(0, w)
        ax2.set_ylim(h, 0)

    plt.tight_layout()
    
def compute_epipole(F):
    '''
    Compute epipole using the fundamental matrix.
    pass F.T as argument to compute the other epipole
    '''
    U, S, V = np.linalg.svd(F)
    e = V[-1, :]
    e = e / e[2]
    return e

def compute_matching_homographies(e2, F, im2, points1, points2):
    '''
    Compute the matching homography matrices
    '''
    h, w = im2.shape
    # create the homography matrix H2 that moves the epipole to infinity
    
    # create the translation matrix to shift to the image center
    T = np.array([[1, 0, -w/2], [0, 1, -h/2], [0, 0, 1]])
    e2_p = T @ e2
    e2_p = e2_p / e2_p[2]
    e2x = e2_p[0]
    e2y = e2_p[1]
    # create the rotation matrix to rotate the epipole back to X axis
    if e2x >= 0:
        a = 1
    else:
        a = -1
    R1 = a * e2x / np.sqrt(e2x ** 2 + e2y ** 2)
    R2 = a * e2y / np.sqrt(e2x ** 2 + e2y ** 2)
    R = np.array([[R1, R2, 0], [-R2, R1, 0], [0, 0, 1]])
    e2_p = R @ e2_p
    x = e2_p[0]
    # create matrix to move the epipole to infinity
    G = np.array([[1, 0, 0], [0, 1, 0], [-1/x, 0, 1]])
    # create the overall transformation matrix
    H2 = np.linalg.inv(T) @ G @ R @ T

    # create the corresponding homography matrix for the other image
    e_x = np.array([[0, -e2[2], e2[1]], [e2[2], 0, -e2[0]], [-e2[1], e2[0], 0]])
    M = e_x @ F + e2.reshape(3,1) @ np.array([[1, 1, 1]])
    points1_t = H2 @ M @ points1.T
    points2_t = H2 @ points2.T
    points1_t /= points1_t[2, :]
    points2_t /= points2_t[2, :]
    b = points2_t[0, :]
    a = np.linalg.lstsq(points1_t.T, b, rcond=None)[0]
    H_A = np.array([a, [0, 1, 0], [0, 0, 1]])
    H1 = H_A @ H2 @ M
    return H1, H2

    