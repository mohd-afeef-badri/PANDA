"""Mesh I/O utilities for various formats (MED, VTK)."""

try:
    import medcoupling as mc
except ImportError:
    print("MEDCoupling not found. Install it from SALOME or compile from source.")
    print("You can access it via: salome shell -- python")
    raise

import numpy as np
from .polygonal_mesh import PolygonalMesh


def load_med_mesh_mc(filename, mesh_name=None, mesh_level=0):
    """
    Load a 2D mesh from a MED file using MEDCoupling.
    
    Parameters:
    -----------
    filename : str
        Path to the MED file
    mesh_name : str, optional
        Name of the mesh to read. If None, reads the first mesh.
    mesh_level : int, default=0
        Mesh level (0 for highest dimension cells)
    
    Returns:
    --------
    PolygonalMesh
        A PolygonalMesh object containing vertices, cells, and boundary edges
    """
    # Read the MED file
    med_mesh = mc.MEDFileMesh.New(filename)
    
    # Get mesh name if not provided
    if mesh_name is None:
        mesh_name = med_mesh.getName()
        print(f"Reading mesh: {mesh_name}")
    
    # Get the mesh at specified level
    umesh = med_mesh.getMeshAtLevel(mesh_level)
    
    # Merge duplicate nodes (important for proper connectivity)
    print(f"Nodes before merge: {umesh.getNumberOfNodes()}")
    umesh.mergeNodes(1e-10)
    print(f"Nodes after merge: {umesh.getNumberOfNodes()}")
    
    # Extract coordinates (only 2D)
    coords = umesh.getCoords()
    vertices = coords.toNumPyArray()[:, :2]
    
    # Extract cells
    cells = []
    n_cells = umesh.getNumberOfCells()
    
    for i in range(n_cells):
        # Get connectivity for this cell
        cell_conn = umesh.getNodeIdsOfCell(i)
        cells.append(list(cell_conn))
    
    print(f"Loaded {len(vertices)} vertices and {len(cells)} cells")
    
    # Extract boundary edges if available
    boundary_edge_tuples = set()
    
    try:
        # Try to get boundary mesh (mesh at level -1)
        boundary_mesh = med_mesh.getMeshAtLevel(-1)
        n_boundary_cells = boundary_mesh.getNumberOfCells()
        
        print(f"Found {n_boundary_cells} boundary edges")
        
        for i in range(n_boundary_cells):
            edge_conn = boundary_mesh.getNodeIdsOfCell(i)
            if len(edge_conn) >= 2:
                v1, v2 = edge_conn[0], edge_conn[1]
                boundary_edge_tuples.add(tuple(sorted((v1, v2))))
    except:
        print("No explicit boundary edges found. Will identify from connectivity.")
    
    # Create PolygonalMesh
    poly_mesh = PolygonalMesh(vertices, cells)
    
    # Map boundary edges to mesh edge indices
    if boundary_edge_tuples:
        poly_boundary_edges = [
            i for i, e in enumerate(poly_mesh.edges)
            if tuple(sorted(e)) in boundary_edge_tuples
        ]
    else:
        # Use default boundary detection (edges with only one adjacent cell)
        poly_boundary_edges = poly_mesh.boundary_edges
    
    poly_mesh.boundary_edges = poly_boundary_edges
    print(f"Identified {len(poly_boundary_edges)} boundary edges")
    
    return poly_mesh

def load_med_mesh_with_groups(filename, mesh_name=None, mesh_level=0):
    """
    Load a mesh with group information from MED file.
    Returns mesh and dictionary of groups.
    """
    med_mesh = mc.MEDFileMesh.New(filename)
    
    if mesh_name is None:
        mesh_name = med_mesh.getName()
    
    umesh = med_mesh.getMeshAtLevel(mesh_level)
    umesh.mergeNodes(1e-10)
    
    # Extract groups
    groups = {}
    try:
        group_names = med_mesh.getGroupsNames()
        print(f"Found groups: {group_names}")
        
        for group_name in group_names:
            group_arr = med_mesh.getGroupArr(mesh_level, group_name)
            groups[group_name] = group_arr.toNumPyArray()
    except:
        print("No groups found in mesh")
    
    # Convert to PolygonalMesh
    coords = umesh.getCoords()
    vertices = coords.toNumPyArray()[:, :2]
    
    cells = []
    for i in range(umesh.getNumberOfCells()):
        cell_conn = umesh.getNodeIdsOfCell(i)
        cells.append(list(cell_conn))
    
    poly_mesh = PolygonalMesh(vertices, cells)
    
    # Get boundary edges
    try:
        boundary_mesh = med_mesh.getMeshAtLevel(-1)
        boundary_edge_tuples = set()
        
        for i in range(boundary_mesh.getNumberOfCells()):
            edge_conn = boundary_mesh.getNodeIdsOfCell(i)
            if len(edge_conn) >= 2:
                v1, v2 = edge_conn[0], edge_conn[1]
                boundary_edge_tuples.add(tuple(sorted((v1, v2))))
        
        poly_boundary_edges = [
            i for i, e in enumerate(poly_mesh.edges)
            if tuple(sorted(e)) in boundary_edge_tuples
        ]
        poly_mesh.boundary_edges = poly_boundary_edges
    except:
        pass
    
    return poly_mesh, groups


def extract_edge_groups_from_med(filename, mesh_name=None):
    """
    Extract edge groups from MED file for boundary condition specification.
    
    Parameters:
    -----------
    filename : str
        Path to MED file
    mesh_name : str, optional
        Name of mesh to read
    
    Returns:
    --------
    dict
        Dictionary mapping group names to arrays of global edge indices
        Format: {'group_name': [edge_idx1, edge_idx2, ...]}
    """
    med_mesh = mc.MEDFileMesh.New(filename)
    
    if mesh_name is None:
        mesh_name = med_mesh.getName()
    
    # Get the boundary mesh (level -1)
    try:
        boundary_mesh = med_mesh.getMeshAtLevel(-1)
    except:
        print("No boundary mesh found at level -1")
        return {}
    
    # Get volume mesh to build edge mapping
    umesh = med_mesh.getMeshAtLevel(0)
    umesh.mergeNodes(1e-10)
    
    # Build edge to index mapping from volume mesh
    edge_to_idx = {}
    edge_list = []
    
    # Extract all edges from volume mesh cells
    for cell_id in range(umesh.getNumberOfCells()):
        cell_conn = umesh.getNodeIdsOfCell(cell_id)
        n_verts = len(cell_conn)
        for i in range(n_verts):
            v1, v2 = cell_conn[i], cell_conn[(i+1) % n_verts]
            edge = tuple(sorted([v1, v2]))
            if edge not in edge_to_idx:
                edge_to_idx[edge] = len(edge_list)
                edge_list.append(edge)
    
    # Extract groups from boundary mesh
    edge_groups = {}
    
    try:
        group_names = med_mesh.getGroupsNames()
        print(f"Found boundary groups: {group_names}")
        
        for group_name in group_names:
            try:
                # Get cell IDs in this group at boundary level (-1)
                group_arr = med_mesh.getGroupArr(-1, group_name)
                boundary_cell_ids = group_arr.toNumPyArray()
                
                # Map boundary cells to global edge indices
                group_edge_indices = []
                for bcell_id in boundary_cell_ids:
                    edge_conn = boundary_mesh.getNodeIdsOfCell(int(bcell_id))
                    if len(edge_conn) >= 2:
                        v1, v2 = edge_conn[0], edge_conn[1]
                        edge = tuple(sorted([v1, v2]))
                        if edge in edge_to_idx:
                            group_edge_indices.append(edge_to_idx[edge])
                
                edge_groups[group_name] = np.array(group_edge_indices)
                print(f"  Group '{group_name}': {len(group_edge_indices)} edges")
                
            except Exception as e:
                print(f"  Could not load group '{group_name}': {e}")
                continue
                
    except Exception as e:
        print(f"Error reading groups: {e}")
    
    return edge_groups


def export_to_med(solver, u_dofs, filename="solution.med", field_name="u",
                  method="P0"):
    if mc is None:
        raise ImportError("MEDCoupling (mc) is required to export MED files")
    if method == "P0":
        _export_med_p0(solver, u_dofs, filename, field_name)
    elif method == "P1_vertex":
        _export_med_p1_vertex(solver, u_dofs, filename, field_name)
    else:
        raise ValueError(f"Unknown export method: {method}")


def _export_med_p0(solver, u_dofs, filename, field_name):
    mesh = solver.mesh
    # Evaluate at cell centroids
    u_cells = np.zeros(mesh.n_cells)
    for cell_id in range(mesh.n_cells):
        cent = mesh.cell_centroid(cell_id)
        u_cells[cell_id] = solver.evaluate_solution(u_dofs, cent, cell_id)

    # Create MEDCoupling mesh
    coords_array = mesh.vertices
    if coords_array.shape[1] == 2:
        # Add z=0 coordinate
        coords_3d = np.column_stack([coords_array, np.zeros(len(coords_array))])
    else:
        coords_3d = coords_array

    # Create coordinate array
    coords_mc = mc.DataArrayDouble(coords_3d)
    coords_mc.setInfoOnComponents(["X", "Y", "Z"])

    # Create unstructured mesh
    umesh = mc.MEDCouplingUMesh("solution_mesh", 2)
    umesh.setCoords(coords_mc)

    # Group cells by type for MEDCoupling contiguity requirement
    cells_by_type = {}
    for cell_id, cell in enumerate(mesh.cells):
        n_nodes = len(cell)
        if n_nodes == 3:
            cell_type = mc.NORM_TRI3
        elif n_nodes == 4:
            cell_type = mc.NORM_QUAD4
        else:
            cell_type = mc.NORM_POLYGON

        if cell_type not in cells_by_type:
            cells_by_type[cell_type] = []
        cells_by_type[cell_type].append((cell_id, cell))

    # Build mapping from new cell order to original cell_id
    cell_mapping = []
    umesh.allocateCells(mesh.n_cells)

    # Insert cells grouped by type
    for cell_type in sorted(cells_by_type.keys()):
        for cell_id, cell in cells_by_type[cell_type]:
            umesh.insertNextCell(cell_type, cell)
            cell_mapping.append(cell_id)

    umesh.finishInsertingCells()

    # Reorder field values according to cell_mapping
    u_cells_reordered = u_cells[cell_mapping]

    # Create field on cells
    field = mc.MEDCouplingFieldDouble(mc.ON_CELLS, mc.ONE_TIME)
    field.setName(field_name)
    field.setMesh(umesh)
    field.setTime(0.0, 0, 0)  # time, iteration, order

    # Set field values
    field_array = mc.DataArrayDouble(u_cells_reordered)
    field_array.setInfoOnComponent(0, field_name)
    field.setArray(field_array)

    # Check consistency
    field.checkConsistencyLight()

    # Write MED file
    med_mesh = mc.MEDFileUMesh()
    med_mesh.setMeshAtLevel(0, umesh)
    med_mesh.setName("solution_mesh")
    med_mesh.write(filename, 2)  # 2 = write mode (overwrite)

    med_writer = mc.MEDFileField1TS()
    med_writer.setFieldNoProfileSBT(field)
    med_writer.write(filename, 0)  # 0 = append mode

    print(f"P0 projection exported to MED: {filename}")


def _export_med_p1_vertex(solver, u_dofs, filename, field_name):
    mesh = solver.mesh
    # Interpolate to vertices using averaging from adjacent cells
    u_vertices = np.zeros(mesh.n_vertices)
    vertex_count = np.zeros(mesh.n_vertices)

    for cell_id, cell in enumerate(mesh.cells):
        for vertex_id in cell:
            vertex_pos = mesh.vertices[vertex_id]
            u_val = solver.evaluate_solution(u_dofs, vertex_pos, cell_id)
            u_vertices[vertex_id] += u_val
            vertex_count[vertex_id] += 1

    # Average values at vertices shared by multiple cells
    u_vertices /= np.maximum(vertex_count, 1)

    # Create MEDCoupling mesh
    coords_array = mesh.vertices
    if coords_array.shape[1] == 2:
        # Add z=0 coordinate
        coords_3d = np.column_stack([coords_array, np.zeros(len(coords_array))])
    else:
        coords_3d = coords_array

    # Create coordinate array
    coords_mc = mc.DataArrayDouble(coords_3d)
    coords_mc.setInfoOnComponents(["X", "Y", "Z"])

    # Create unstructured mesh
    umesh = mc.MEDCouplingUMesh("solution_mesh", 2)
    umesh.setCoords(coords_mc)

    # Group cells by type for MEDCoupling contiguity requirement
    cells_by_type = {}
    for cell_id, cell in enumerate(mesh.cells):
        n_nodes = len(cell)
        if n_nodes == 3:
            cell_type = mc.NORM_TRI3
        elif n_nodes == 4:
            cell_type = mc.NORM_QUAD4
        else:
            cell_type = mc.NORM_POLYGON

        if cell_type not in cells_by_type:
            cells_by_type[cell_type] = []
        cells_by_type[cell_type].append((cell_id, cell))

    umesh.allocateCells(mesh.n_cells)

    # Insert cells grouped by type
    for cell_type in sorted(cells_by_type.keys()):
        for cell_id, cell in cells_by_type[cell_type]:
            umesh.insertNextCell(cell_type, cell)

    umesh.finishInsertingCells()

    # Create field on nodes (no reordering needed for node fields)
    field = mc.MEDCouplingFieldDouble(mc.ON_NODES, mc.ONE_TIME)
    field.setName(field_name)
    field.setMesh(umesh)
    field.setTime(0.0, 0, 0)

    # Set field values
    field_array = mc.DataArrayDouble(u_vertices)
    field_array.setInfoOnComponent(0, field_name)
    field.setArray(field_array)

    # Check consistency
    field.checkConsistencyLight()

    # Write MED file
    med_mesh = mc.MEDFileUMesh()
    med_mesh.setMeshAtLevel(0, umesh)
    med_mesh.setName("solution_mesh")
    med_mesh.write(filename, 2)  # 2 = write mode (overwrite)

    med_writer = mc.MEDFileField1TS()
    med_writer.setFieldNoProfileSBT(field)
    med_writer.write(filename, 0)  # 0 = append mode

    print(f"P1 vertex interpolation exported to MED: {filename}")

def project_and_export_to_triangular_mesh_med(solver, u_dofs, tria_mesh_file,
                                         output_file="solution_tria.med",
                                         field_name="u"):
    """
    Project P1 DG solution (polymesh cell-centroid values) onto a triangular
    MED mesh whose vertices correspond to the polymesh cell centroids, and
    write the result into a MED file (node-based field).
    """
    if mc is None:
        raise ImportError("MEDCoupling (mc) is required to export MED files")

    print(f"Loading triangular mesh from {tria_mesh_file}...")
    tria_mesh = load_med_mesh_mc(tria_mesh_file)

    if tria_mesh.n_vertices != solver.mesh.n_cells:
        print(f"WARNING: Triangular mesh has {tria_mesh.n_vertices} vertices "
              f"but polymesh has {solver.mesh.n_cells} cells!")
        print("Proceeding anyway, but results may be incorrect.")

    # Evaluate DG solution at each polymesh cell centroid -> values at tria nodes
    u_tria_vertices = np.zeros(tria_mesh.n_vertices)
    for cell_id in range(min(solver.mesh.n_cells, tria_mesh.n_vertices)):
        cent = solver.mesh.cell_centroid(cell_id)
        u_tria_vertices[cell_id] = solver.evaluate_solution(u_dofs, cent, cell_id)

    # Build MEDCoupling mesh from the triangular mesh data
    coords_array = tria_mesh.vertices
    if coords_array.shape[1] == 2:
        coords_3d = np.column_stack([coords_array, np.zeros(len(coords_array))])
    else:
        coords_3d = coords_array

    coords_mc = mc.DataArrayDouble(coords_3d)
    coords_mc.setInfoOnComponents(["X", "Y", "Z"])

    umesh = mc.MEDCouplingUMesh("tria_mesh", 2)
    umesh.setCoords(coords_mc)

    # Group and insert cells (keep same approach used elsewhere)
    cells_by_type = {}
    for cell_id, cell in enumerate(tria_mesh.cells):
        n_nodes = len(cell)
        if n_nodes == 3:
            cell_type = mc.NORM_TRI3
        elif n_nodes == 4:
            cell_type = mc.NORM_QUAD4
        else:
            cell_type = mc.NORM_POLYGON

        cells_by_type.setdefault(cell_type, []).append((cell_id, cell))

    umesh.allocateCells(tria_mesh.n_cells)
    for cell_type in sorted(cells_by_type.keys()):
        for _, cell in cells_by_type[cell_type]:
            umesh.insertNextCell(cell_type, cell)
    umesh.finishInsertingCells()

    # Create node-based field and write MED file
    field = mc.MEDCouplingFieldDouble(mc.ON_NODES, mc.ONE_TIME)
    field.setName(field_name)
    field.setMesh(umesh)
    field.setTime(0.0, 0, 0)

    field_array = mc.DataArrayDouble(u_tria_vertices)
    field_array.setInfoOnComponent(0, field_name)
    field.setArray(field_array)

    field.checkConsistencyLight()

    med_mesh = mc.MEDFileUMesh()
    med_mesh.setMeshAtLevel(0, umesh)
    med_mesh.setName("tria_mesh")
    med_mesh.write(output_file, 2)  # overwrite

    med_writer = mc.MEDFileField1TS()
    med_writer.setFieldNoProfileSBT(field)
    med_writer.write(output_file, 0)  # append

    print(f"Solution exported to triangular MED: {output_file}")
    print(f"  - Triangular mesh vertices: {tria_mesh.n_vertices}")
