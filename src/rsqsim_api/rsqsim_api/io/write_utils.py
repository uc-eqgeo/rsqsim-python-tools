import numpy as np
import scipy.spatial
import shapely as shp
import os

import rsqsim_api.fault.fault


def write_catalogue_dataframe_and_arrays(prefix: str, catalogue, directory: str = None,
                                         write_index: bool = True):
    if directory is not None:
        assert os.path.exists(directory)
        dir_path = directory
    else:
        dir_path = ""

    assert isinstance(prefix, str)
    assert len(prefix) > 0
    if prefix[-1] != "_":
        prefix += "_"
    prefix_path = os.path.join(dir_path, prefix)
    df_file = prefix_path + "catalogue.csv"
    event_file = prefix_path + "events.npy"
    patch_file = prefix_path + "patches.npy"
    slip_file = prefix_path + "slip.npy"
    slip_time_file = prefix_path + "slip_time.npy"

    catalogue.catalogue_df.to_csv(df_file, index=write_index)
    for file, array in zip([event_file, patch_file, slip_file, slip_time_file],
                           [catalogue.event_list, catalogue.patch_list, catalogue.patch_slip,
                            catalogue.patch_time_list]):
        np.save(file, array)


def create_quad_mesh_from_fault(points: np.ndarray, edges: np.ndarray, triangles: np.ndarray, resolution: float=5000.0,
                                num_search_tris: int=10, fit_plane_epsilon: float=1.0e-5, cutoff_rotation_vecmag: float=0.98,
                                is_plane_epsilon: float=1.0):
    """
    Create a quad mesh, given triangular mesh information.
    Returned values are all in the fault_mesh_info dictionary:
        plane_normal: Normal vector for best-fit plane through original points
        plane_origin: Reference point for best-fit plane through original points
        rotation_matrix: Rotation matrix based on plane_normal
        points_local: Original points transformed to local coordinates
        edges_local: Original edges transformed to local coordinates
        fault_is_plane: Boolean determining whether fault is a plane within given tolerance
        quad_edges: Dictionary containing the 4 edges composing the mesh boundary
        mesh_points_local: The generated mesh points (structured mesh) in local fault coordinates
        num_horiz_points: The number of mesh points in the horizontal direction
        num_vert_points: The number of mesh points in the vertical direction
        mesh_points_global: The generated mesh points (structured mesh) in global coordinates
    """
    # Get coordinate transformation information.
    (plane_normal, plane_origin) = fit_plane_to_points(points, eps=fit_plane_epsilon)
    rotation_matrix = get_fault_rotation_matrix(plane_normal, cutoff_vecmag=cutoff_rotation_vecmag)

    # Get local coordinates.
    (points_local, edges_local, fault_is_plane) = fault_global_to_local(points, edges, rotation_matrix, plane_origin,
                                                                        plane_epsilon=is_plane_epsilon)

    # Get edges of boundary and create a grid in local coordinates.
    quad_edges = get_quad_mesh_edges(edges_local)
    (mesh_points_local,
     num_horiz_points, num_vert_points) = create_local_grid(points_local, quad_edges, triangles, fault_is_plane,
                                                            resolution=resolution, num_search_tris=num_search_tris)

    # Convert to global coordinates.
    (mesh_points_global, edges_global) = fault_local_to_global(mesh_points_local, edges_local, rotation_matrix, plane_origin)

    # Create dictionary of results.
    fault_mesh_info = {'plane_normal': plane_normal,
                       'plane_origin': plane_origin,
                       'rotation_matrix': rotation_matrix,
                       'points_local': points_local,
                       'edges_local': edges_local,
                       'fault_is_plane': fault_is_plane,
                       'quad_edges': quad_edges,
                       'mesh_points_local': mesh_points_local,
                       'num_horiz_points': num_horiz_points,
                       'num_vert_points': num_vert_points,
                       'mesh_points_global': mesh_points_global}

    return fault_mesh_info

    
def fit_plane_to_points(points: np.ndarray, eps: float=1.0e-5):
    """
    Find best-fit plane through a set of points, after first insuring the plane goes through
    the mean (centroid) of all the points in the array. This is probably better than my
    initial method, since the SVD is only over a 3x3 array (rather than the num_pointsxnum_points
    array). 
    Returned values are:
        plane_normal:  Normal vector to plane (A, B, C)
        plane_origin:  Point on plane that may be considered as the plane origin
    """
    # Compute plane origin and subract it from the points array.
    plane_origin = np.mean(points, axis=0)
    x = points - plane_origin

    # Dot product to yield a 3x3 array.
    moment = np.dot(x.T, x)

    # Extract single values from SVD computation to get normal.
    plane_normal = np.linalg.svd(moment)[0][:,-1]
    small = np.where(np.abs(plane_normal) < eps)
    plane_normal[small] = 0.0
    plane_normal /= np.linalg.norm(plane_normal)
    if (plane_normal[-1] < 0.0):
        plane_normal *= -1.0

    return (plane_normal, plane_origin)


def get_fault_rotation_matrix(plane_normal: np.ndarray, cutoff_vecmag: float = 0.98):
    """
    Compute rotation matrix, given the normal to the plane. If the normal is nearly
    vertical an alternate reference direction is used to compute the two tangential
    directions.
    Returned values are:
        rotation_matrix: 3x3 rotation matrix with columns (tan_dir1, tan_dir2, plane_normal).
    """
    # Reference directions to try are z=1 (vertical) and y=1 (north).
    ref_dir1 = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    ref_dir2 = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    ref_dir = ref_dir1

    # If normal is nearly vertical, use north reference direction.
    if (np.dot(ref_dir1, plane_normal) > cutoff_vecmag):
        ref_dir = ref_dir2
        
    # Get two tangential directions in plane.
    tan_dir1 = rsqsim_api.fault.fault.cross_3d(ref_dir, plane_normal)
    tan_dir1 /= np.linalg.norm(tan_dir1)
    tan_dir2 = rsqsim_api.fault.fault.cross_3d(plane_normal, tan_dir1)
    tan_dir2 /= np.linalg.norm(tan_dir2)

    # Form rotation matrix.
    rotation_matrix = np.column_stack((tan_dir1, tan_dir2, plane_normal))

    return rotation_matrix
    
    
def fault_global_to_local(points: np.ndarray, edges: np.ndarray, rotation_matrix: np.ndarray,
                          plane_origin: np.ndarray, plane_epsilon: float = 1.0):
    """
    Convert global fault surface coordinates to local coordinates, referenced to given origin.
    If plane z-values are below epsilon value, surface is assumed to be a plane.
    Returned values are:
        points_local: Point coordinates referenced to origin and rotated to local orientation.
        edges_local: Edge coordinates referenced to origin and rotated to local orientation.
        fault_is_plane: Boolean variable that is true if the fault is planar within the given tolerance.
    """
    # Rotate referenced coordinates.
    points_local = np.dot(points - plane_origin, rotation_matrix.transpose())
    edges_local = np.dot(edges - plane_origin, rotation_matrix.transpose())

    # Determine whether mean normal component is above or below epsilon value.
    mean_normal = np.mean(np.abs(points_local[:,-1]))
    fault_is_plane = True
    if mean_normal > plane_epsilon:
        fault_is_plane = False

    return (points_local, edges_local, fault_is_plane)
    
    
def fault_local_to_global(points: np.ndarray, edges: np.ndarray,
                          rotation_matrix: np.ndarray, plane_origin: np.ndarray):
    """
    Convert local fault surface coordinates to global coordinates.
    Returned values are:
        points_global: Point coordinates rotated to global orientation and with origin added back in.
        edges_global: Edge coordinates rotated to global orientation and with origin added back in.
    """
    # Rotate coordinates and add reference point back in.
    points_global = np.dot(points, rotation_matrix) + plane_origin
    edges_global = np.dot(edges, rotation_matrix) + plane_origin
    
    return (points_global, edges_global)
    
    
def get_quad_mesh_edges(edges: np.ndarray, corner_separation: float=3000.0):
    """
    Determine 4 sets of edges, assuming a semi-quadrilateral layout.
    Note that all coordinates are assumed to be fault-local coordinates.
    Uses dot products of line segments with the next segment to determine 4 smallest dot products.
    In the future, we will use the corner_separation parameter to work around 'chopped-off' corners.
    Returned values are contained in the edge_sides dictionary:
        left_edge: Coordinates of edge corresponding to leftmost edge
        right_edge: Coordinates of edge corresponding to rightmost edge
        bottom_edge: Coordinates of edge corresponding to bottommost edge
        top_edge: Coordinates of edge corresponding to topmost edge
    """
    # Create vectors of line segments composing outer edges.
    num_points = edges.shape[0]
    v1 = np.diff(edges[:,0:2], axis=0, append=edges[1,0:2].reshape(1,2))
    v2 = np.diff(edges[:,0:2], axis=0, prepend=edges[-2,0:2].reshape(1,2))
    v1 /= np.linalg.norm(v1, axis=1).reshape(num_points, 1)
    v2 /= np.linalg.norm(v2, axis=1).reshape(num_points, 1)

    # Compute dot product magnitude and sorting index array.
    dot_prod_mag = np.abs(np.sum(v1*v2, axis=1))
    sort_args = np.argsort(dot_prod_mag)

    # Everything below here is pretty kludgy and should be tidied up.
    # Determine which coordinates correspond to each corner.
    corners = edges[sort_args[0:4],:]
    x_sort = np.argsort(corners[:,0])
    left_corners = corners[x_sort[0:2], :]
    right_corners = corners[x_sort[2:], :]
    ul_corner = left_corners[0,:]
    bl_corner = left_corners[1,:]
    if (ul_corner[1] < bl_corner[1]):
        ul_corner = left_corners[1,:]
        bl_corner = left_corners[0,:]
    ur_corner = right_corners[0,:]
    br_corner = right_corners[1,:]
    if (ur_corner[1] < br_corner[1]):
        ur_corner = right_corners[1,:]
        br_corner = right_corners[0,:]

    # Get corner indices.
    ul_ind = sort_args[np.argmin(np.linalg.norm(corners - ul_corner, axis=1))]
    bl_ind = sort_args[np.argmin(np.linalg.norm(corners - bl_corner, axis=1))]
    ur_ind = sort_args[np.argmin(np.linalg.norm(corners - ur_corner, axis=1))]
    br_ind = sort_args[np.argmin(np.linalg.norm(corners - br_corner, axis=1))]
    ind_min = min(ul_ind, bl_ind, ur_ind, br_ind)
    ind_max = max(ul_ind, bl_ind, ur_ind, br_ind)

    # Get edges between corners.
    left_edge = get_edge(edges, ul_ind, bl_ind, ind_min, ind_max, 1)
    right_edge = get_edge(edges, ur_ind, br_ind, ind_min, ind_max, 1)
    bottom_edge = get_edge(edges, bl_ind, br_ind, ind_min, ind_max, 0)
    top_edge = get_edge(edges, ul_ind, ur_ind, ind_min, ind_max, 0)

    # Create a dictionary for now.
    edge_sides = {'left_edge': left_edge,
                  'right_edge': right_edge,
                  'bottom_edge': bottom_edge,
                  'top_edge': top_edge}

    return edge_sides


def get_edge(edges: np.ndarray, ind1: int, ind2: int, ind_min: int, ind_max: int, sort_ind: int):
    """
    Extract edge from edges array, given the indices to use.
    Returned value:
        edge:  Edge coordinates, sorted according to the given sorting index
    """
    i1 = min(ind1, ind2)
    i2 = max(ind1, ind2)
    edge = edges[i1:i2+1,:]
    if (i1 == ind_min and i2 == ind_max):
        edge = np.concatenate((edges[0:i1+1,:], edges[i2:-1,:]), axis=0)
        
    sort_inds = np.argsort(edge[:, sort_ind])
    edge = edge[sort_inds,:]

    return edge


def create_local_grid(points: np.ndarray, edge_sides: dict, triangles: np.ndarray,
                      fault_is_plane: bool, resolution: float=5000.0, num_search_tris: int=10):
    """
    Create a grid of points in local coordinates. If the fault is a plane, z-coordinate is
    always zero. Otherwise, points are interpolated from the enclosing triangle vertices.
    Returned values are:
        mesh_points: The computed mesh points in the local coordinate system
        num_horiz_points: The number of points in the horzontal direction
        num_vert_points: The number of points in the vertical direction
    """

    # Edge coordinates.
    left_edge = edge_sides['left_edge']
    right_edge = edge_sides['right_edge']
    bottom_edge = edge_sides['bottom_edge']
    top_edge = edge_sides['top_edge']

    # Determine number of points in each direction.
    left_diffs = np.diff(left_edge, axis=0)
    left_length = np.sum(np.linalg.norm(left_diffs, axis=1))
    num_divs_left = left_length/resolution

    right_diffs = np.diff(right_edge, axis=0)
    right_length = np.sum(np.linalg.norm(right_diffs, axis=1))
    num_divs_right = right_length/resolution

    num_vert_points = int(round(0.5*(num_divs_left + num_divs_right))) + 1

    bottom_diffs = np.diff(bottom_edge, axis=0)
    bottom_length = np.sum(np.linalg.norm(bottom_diffs, axis=1))
    num_divs_bottom = bottom_length/resolution

    top_diffs = np.diff(top_edge, axis=0)
    top_length = np.sum(np.linalg.norm(top_diffs, axis=1))
    num_divs_top = top_length/resolution

    num_horiz_points = int(round(0.5*(num_divs_bottom + num_divs_top))) + 1
    num_points = num_vert_points*num_horiz_points

    # Get interpolated points on edges.
    ygrid_left = np.linspace(left_edge[0,1], left_edge[-1,1], num=num_vert_points, dtype=np.float64)
    xgrid_left = np.interp(ygrid_left, left_edge[:,1], left_edge[:,0])
    zgrid_left = np.interp(ygrid_left, left_edge[:,1], left_edge[:,2])
    ygrid_right = np.linspace(right_edge[0,1], right_edge[-1,1], num=num_vert_points, dtype=np.float64)
    xgrid_right = np.interp(ygrid_right, right_edge[:,1], right_edge[:,0])
    zgrid_right = np.interp(ygrid_right, right_edge[:,1], right_edge[:,2])
    xgrid_bottom = np.linspace(bottom_edge[0,0], bottom_edge[-1,0], num=num_horiz_points, dtype=np.float64)
    ygrid_bottom = np.interp(xgrid_bottom, bottom_edge[:,0], bottom_edge[:,1])
    zgrid_bottom = np.interp(xgrid_bottom, bottom_edge[:,0], bottom_edge[:,2])
    xgrid_top = np.linspace(top_edge[0,0], top_edge[-1,0], num=num_horiz_points, dtype=np.float64)
    ygrid_top = np.interp(xgrid_top, top_edge[:,0], top_edge[:,1])
    zgrid_top = np.interp(xgrid_top, top_edge[:,0], top_edge[:,2])

    # Create 2D mesh.
    mesh_points = np.zeros((num_vert_points, num_horiz_points, 3), dtype=np.float64)

    # We need to do this for points on the boundary, which might fall outside a mesh triangle.
    if not(fault_is_plane):
        z = np.zeros((num_vert_points, num_horiz_points), dtype=np.float64)
        is_mesh_edge = np.zeros((num_vert_points, num_horiz_points), dtype=np.bool)
        is_mesh_edge[0,:] = True
        is_mesh_edge[-1,:] = True
        is_mesh_edge[:,0] = True
        is_mesh_edge[:,-1] = True
        z[:,0] = zgrid_left
        z[:,-1] = zgrid_right
        z[0,:] = zgrid_bottom
        z[-1,:] = zgrid_top
    
    for row_num in range(num_vert_points):
        mesh_points[row_num,:,0] = np.linspace(xgrid_left[row_num], xgrid_right[row_num],
                                               num=num_horiz_points, dtype=np.float64)
    for col_num in range(num_horiz_points):
        mesh_points[:,col_num,1] = np.linspace(ygrid_bottom[col_num], ygrid_top[col_num],
                                               num=num_vert_points, dtype=np.float64)

    # If surface is not a plane, interpolate to get z-coordinates.
    mesh_points = mesh_points.reshape(num_points, 3)
    if not(fault_is_plane):
        is_mesh_edge = is_mesh_edge.reshape(num_points)
        z = z.reshape(num_points)
        mesh_points[:,2] = tri_interpolate_zcoords(points, triangles, mesh_points[:,0:2],
                                                   is_mesh_edge, num_search_tris=num_search_tris)
        mesh_points[is_mesh_edge,2] = z[is_mesh_edge]

    return (mesh_points, num_horiz_points, num_vert_points)


def create_cells_from_dims(num_verts_x: int, num_verts_y: int):
    """
    Create simple quad cell connectivity array given the vertical and horizontal dimensions.
    Returned values are:
        cell_array: A num_cellsx4 array describing vertices composing each cell
    """
    num_cells_x = num_verts_x - 1
    num_cells_y = num_verts_y - 1
    num_cells = num_cells_x*num_cells_y
    cell_array = np.zeros((num_cells, 4), dtype=np.int)
    cell_num = 0

    # I am sure this could be done in a more efficient way.
    for y_cell in range(num_cells_y):
        for x_cell in range(num_cells_x):
            cell_array[cell_num, 0] = x_cell + num_verts_x*y_cell
            cell_array[cell_num, 1] = cell_array[cell_num, 0] + 1
            cell_array[cell_num, 2] = cell_array[cell_num, 0] + num_verts_x + 1
            cell_array[cell_num, 3] = cell_array[cell_num, 0] + num_verts_x
            cell_num += 1

    return cell_array
            
    
def tri_interpolate_zcoords(points: np.ndarray, triangles: np.ndarray, mesh_points: np.ndarray,
                            is_mesh_edge: np.ndarray, num_search_tris: int=10):
    """
    Interpolate z-coordinates to a set of 2D points using 3D point coordinates and a triangular mesh.
    If point is along a mesh boundary, the boundary values are used instead.
    Returned values are:
        z: The interpolated z-values
    """
    # Get triangle centroid coordinates and create KD-tree.
    tri_coords = points[triangles,:]
    tri_coords2D = points[triangles,0:2]
    tri_centroids = np.mean(tri_coords2D, axis=1)
    tri_tree = scipy.spatial.cKDTree(tri_centroids)

    # Loop over points.
    coords2d = mesh_points[:,0:2]
    num_mesh_points = coords2d.shape[0]
    z = np.zeros(num_mesh_points, dtype=np.float64)
    for point_num in range(num_mesh_points):
        if not(is_mesh_edge[point_num]):
            z[point_num] = project_2d_coords(tri_coords, coords2d[point_num,:], tri_tree, num_search_tris=num_search_tris)

    return z


def project_2d_coords(tri_coords: np.ndarray, coord: np.ndarray, tree: scipy.spatial.ckdtree.cKDTree, num_search_tris: int=10):
    """
    Project z-coordinate for triangle coordinates.
    Returned values are:
        projected_coords[2]: The z-value interpolated from values at triangle vertices
    """
    # Find nearest triangles, then loop over them.
    (distances, ix) = tree.query(coord, k=num_search_tris)
    in_tri = False
    for triangle_num in range(num_search_tris):
        triangle = ix[triangle_num]
        tri_coord = tri_coords[triangle,:]
        (in_tri, projected_coords) = find_projected_coords(tri_coord, coord)
        if (in_tri):
            break

    if (not in_tri):
        msg = 'No containing triangle found for point (%g, %g)' % (coord[0], coord[1])
        raise ValueError(msg)

    return projected_coords[2]
        
        
def find_projected_coords(tri_coord, point):
    """
    Find whether a point projects within a triangle, and if so compute the
    projected coordinates.
    Returned values are:
        in_tri: Boolean indicating whether the point is contained in the triangle
        projected_coords: The (x,y,z) coordinates of the point inferred from the triangle values
    """
    x_point = point[0]
    y_point = point[1]
    point_plane = shp.geometry.Point(x_point, y_point)
    polygon = shp.geometry.Polygon(tri_coord[:,0:2])
    in_tri = polygon.intersects(point_plane)
    projected_coords = None

    # If point is inside triangle, compute area coordinates and use these to
    # compute projected coordinates.
    # If we want to do a check we could either:
    # 1.  Make sure that alpha, beta, and gamma are all between 0 and 1.
    # 2.  Make sure that projected_coords[0 and projected_coords[1] are equal to
    #     the original point coordinates.
    if (in_tri):
        u = tri_coord[1,0:2] - tri_coord[0,0:2]
        v = tri_coord[2,0:2] - tri_coord[0,0:2]
        area = abs(np.cross(u, v))
        u1 = point - tri_coord[0,0:2]
        v1 = point - tri_coord[1,0:2]
        area1 = abs(np.cross(u1, v1))
        u2 = point - tri_coord[1,0:2]
        v2 = point - tri_coord[2,0:2]
        area2 = abs(np.cross(u2, v2))
        alpha = area1/area
        beta = area2/area
        gamma = 1.0 - alpha - beta
        projected_coords = beta*tri_coord[0,:] + gamma*tri_coord[1,:] + alpha*tri_coord[2,:]

    return (in_tri, projected_coords)


def get_mesh_boundary(triangles):
    """
    Find outer boundary of a triangulated mesh, assuming no holes.
    Boundary is determined based purely on connectivity.
    Returned values are:
        vert_inds: The indices of the vertices composing the mesh boundary, ordered to be continuous
    """
    # Create edges and sort each vertices on each edge.
    edge0 = triangles[:,0:2]
    edge1 = triangles[:,1:3]
    edge2 = triangles.take((0,2), axis=1)
    edges = np.concatenate((edge0, edge1, edge2), axis=0)
    edge_sort = np.sort(edges, axis=1)

    # Get unique edges that are only present once.
    (uniq, uniq_ids, counts) = np.unique(edge_sort, axis=0, return_index=True, return_counts=True)
    edge_inds = np.arange(edge_sort.shape[0], dtype=np.int)
    outer_edge_ids = edge_inds[np.in1d(edge_inds, uniq_ids[counts==1])]
    outer_edges = edge_sort[outer_edge_ids,:]
    num_outer_edges = outer_edges.shape[0]

    # Assume we need to close the polygon.
    num_outer_verts = num_outer_edges + 1

    # Loop over outer edges and use traversal method to get ordered vertices.
    v_start = outer_edges[0,0]
    v_end = outer_edges[0,1]
    vert_inds = -1*np.ones(num_outer_verts, dtype=np.int)
    vert_inds[0] = v_start
    vert_inds[1] = v_end
    vert_num = 2
    outer_edges[0,:] = -1
    for edge_num in range(1,num_outer_edges):
        edge_inds_next = np.where(outer_edges == v_end)
        if (edge_inds_next[0].shape[0] < 1):
            msg = "Next edge not found for vertex %d" % v_end
            raise ValueError(msg)
        edge_ind_next = edge_inds_next[0][0]
        vert_ind_next = 0
        if (edge_inds_next[1][0] == 0):
            vert_ind_next = 1
        vert_inds[vert_num] = outer_edges[edge_ind_next, vert_ind_next]
        outer_edges[edge_ind_next, :] = -1
        v_end = vert_inds[vert_num]
        vert_num += 1

    return vert_inds
    
