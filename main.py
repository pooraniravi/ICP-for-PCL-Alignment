import open3d as o3d
import numpy as np
from sklearn.neighbors import NearestNeighbors
import copy

def best_fit_transform(A, B):
    """
    Calculates the least-squares best-fit transform that maps corresponding points A to B in m spatial dimensions
    Input:
      A: Nxm numpy array of corresponding points
      B: Nxm numpy array of corresponding points
    Returns:
      T: (m+1)x(m+1) homogeneous transformation matrix that maps A on to B
      R: mxm rotation matrix
      t: mx1 translation vector
    """
    assert A.shape==B.shape

    # Get number of dimensions
    m= A.shape[1]

    # Translate points to their centroids
    centroid_A= np.mean(A, axis=0)
    centroid_B= np.mean(B, axis=0)
    A= A-centroid_A
    B= B-centroid_B

    # Find rotation matrix
    H= np.dot(A.T, B)
    U,S,Vt= np.linalg.svd(H)
    R= np.dot(Vt.T, U.T)

    # Account for special reflection case
    if np.linalg.det(R)<0:
       Vt[m-1, :]*= -1
       R= np.dot(Vt.T, U.T)

    # Find translation vector
    t= centroid_B.T-np.dot(R, centroid_A.T)

    # Find the homogeneous transformation
    T= np.identity(m + 1)
    T[:m, :m]= R
    T[:m, m]= t

    return T, R, t

def nearest_neighbor(src,dst):
    """
    Find the nearest (Euclidean) neighbor in dst for each point in src
    Input:
        src: Nxm array of points
        dst: Nxm array of points
    Output:
        distances: Euclidean distances of the nearest neighbor
        indices: dst indices of the nearest neighbor
    """
    neigh= NearestNeighbors(n_neighbors=1, algorithm='kd_tree')
    neigh.fit(dst)
    distances,indices= neigh.kneighbors(src, return_distance=True)
    return distances.ravel(), indices.ravel()
    
if __name__=='__main__':

    # Load input data vertices and normals
    src_file= 'bun045_v2.ply'
    dst_file= 'bun000_v2.ply'
    src_mesh= o3d.io.read_triangle_mesh("./data/bun045_v2.ply")
    dst_mesh= o3d.io.read_triangle_mesh("./data/bun000_v2.ply")

    src_pts= np.asarray(src_mesh.vertices)
    dst_pts= np.asarray(dst_mesh.vertices)
    src_mesh.compute_vertex_normals()
    dst_mesh.compute_vertex_normals()
    src_pts_normals= np.asarray(src_mesh.vertex_normals)
    dst_pts_normals= np.asarray(dst_mesh.vertex_normals)
    # dst_mesh.compute_vertex_normals()
    # o3d.visualization.draw_geometries([dst_mesh])
    # print(src_pts_normals.shape, dst_pts.shape)

    # Sub sample points for initial correspondences
    # sampled_ids= np.random.uniform(0,1,size= 1000) #TODO
    # sampling_rate= 1 #TODO
    # A= src_pts[sampled_ids<sampling_rate, :]
    # A_normals= src_pts_normals[sampled_ids<sampling_rate, :]
    # sampled_ids= np.random.uniform(0,1,size= 1000)
    # B= dst_pts[sampled_ids<sampling_rate, :]
    # B_normals= dst_pts_normals[sampled_ids<sampling_rate, :]
    m= src_pts.shape[1]
    A_idx= np.random.choice(src_pts.shape[0], 15000, replace= False)
    A= src_pts[A_idx,:]
    A_normals= src_pts_normals[A_idx,:]
    B_idx= np.random.choice(dst_pts.shape[0], 15000, replace= False)
    B= dst_pts[B_idx,:]
    B_normals= dst_pts_normals[B_idx,:]
    # print(A.shape, B.shape, A_normals.shape, B_normals.shape)

    # Convert points from euclidean coordinates to homogeneous coordinates
    src= np.ones((m+1, A.shape[0]))
    dst= np.ones((m+1, B.shape[0]))
    src[:m,:]= np.copy(A.T)
    dst[:m,:]= np.copy(B.T)
    # print(src.shape, dst.shape)

    max_iterations= 10
    prev_error= 0
    mean_errors= []
    tolerance= 0.001
    for i in range(max_iterations):

        # Find nearest neighbours between current src and dst points
        distances, indices= nearest_neighbor(src[:m,:].T, dst[:m,:].T)
        # print(distances.shape, indices.shape)

        # Match each point of src set to closest point of dst set
        matched_src_pts= src[:m,:].T.copy()
        matched_dst_pts= dst[:m, indices].T

        # Reject outliers by comparing point to point distances and normal angles
        matched_src_pts_normals= A_normals.copy()
        matched_dst_pts_normals= B_normals[indices,:]
        angles= np.zeros(matched_src_pts_normals.shape[0])
        for k in range(matched_src_pts_normals.shape[0]):
            v1= matched_src_pts_normals[k,:]
            v2= matched_dst_pts_normals[k,:]
            cos_angle= v1.dot(v2)/(np.linalg.norm(v1) * np.linalg.norm(v2))
            angles[k]= np.arccos(cos_angle) / np.pi * 180
        # print(distances)

        dist_threshold= 0.05
        dist_bool_flag= (distances<dist_threshold)
        angle_threshold= 24
        angle_bool_flag= (angles<angle_threshold)
        reject_flag= dist_bool_flag*angle_bool_flag

        matched_src_pts = matched_src_pts[reject_flag, :]
        matched_dst_pts = matched_dst_pts[reject_flag, :]
        # print(matched_src_pts.shape)

        # Compute the transformation between the current source and dst set
        T,_,_= best_fit_transform(matched_src_pts, matched_dst_pts)

        # Update the current source
        src= np.dot(T, src)
        print('\n ICP iteration: %d/%d ...' % (i+1, max_iterations), end='', flush=True)

        # Check error
        mean_error= np.mean(distances)
        mean_errors.append(mean_error)
        if np.abs(prev_error-mean_error)<tolerance:
            print("\nStopping iteration... the distance between two adjacent iterations is lower than tolerance (%.f < %f)"% (np.abs(prev_error-mean_error), tolerance))
            break
        prev_error= mean_error

    # Calculate final transformation
    T,_,_= best_fit_transform(A, src[:m,:].T)

    # return T, mean_errors
    res_mesh = copy.deepcopy(src_mesh)
    res_mesh.transform(T)

    # dst_mesh_o3d = tools.tools.toO3d(dst_mesh, color=(0.5, 0, 0))
    # src_mesh_o3d = tools.tools.toO3d(src_tm, color=(0, 0, 0.5))
    # res_mesh_o3d = tools.tools.toO3d(res_tm, color=(0, 0, 0.5))
    src_mesh.paint_uniform_color([0.1, 0.1, 0.7])
    dst_mesh.paint_uniform_color([0.9, 0.1, 0.1])
    res_mesh.paint_uniform_color([0.1, 0.1, 0.7])
    o3d.visualization.draw_geometries([dst_mesh, src_mesh])
    o3d.visualization.draw_geometries([dst_mesh, res_mesh])

    # pcd = mesh.sample_points_uniformly(number_of_points=500)
    # o3d.visualization.draw_geometries([pcd])

