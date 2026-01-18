### EXAMPLE USAGE - GLOBAL BOUNDARY CONDITIONS

```python
def example_all_boundaries_simple():
    """
    Example: Apply the same BC to ALL boundaries
    Simplest case - homogeneous Dirichlet everywhere
    """
    from med_reader import load_med_mesh_mc
    from bc_handler import BoundaryConditionManager, P1DGPoissonSolverWithBC
    
    mesh = load_med_mesh_mc("./mesh/mesh.med")
    bc_manager = BoundaryConditionManager(mesh)
    
    # Apply Dirichlet BC u=0 to ALL boundaries
    bc_manager.add_bc_to_all_boundaries("dirichlet", 0.0)
    
    solver = P1DGPoissonSolverWithBC(mesh, bc_manager, penalty_param=10.0)
    f = lambda x, y: 1.0
    u_dofs = solver.solve(f)
    
    return u_dofs

def example_all_boundaries_function():
    """
    Example: Apply a function-based BC to all boundaries
    """
    from med_reader import load_med_mesh_mc
    from bc_handler import BoundaryConditionManager, P1DGPoissonSolverWithBC
    
    mesh = load_med_mesh_mc("./mesh/mesh.med")
    bc_manager = BoundaryConditionManager(mesh)
    
    # Apply spatially-varying Dirichlet BC to all boundaries
    bc_manager.add_bc_to_all_boundaries(
        bc_type="dirichlet",
        value_func=lambda x, y: x**2 + y**2
    )
    
    solver = P1DGPoissonSolverWithBC(mesh, bc_manager)
    f = lambda x, y: -4.0  # Laplacian of x^2 + y^2
    u_dofs = solver.solve(f)
    
    return u_dofs


def example_global_with_override():
    """
    Example: Set a global BC, then override specific boundaries
    Priority: specific > global
    """
    from med_reader import load_med_mesh_mc
    from bc_handler import BoundaryConditionManager, P1DGPoissonSolverWithBC
    
    mesh = load_med_mesh_mc("./mesh/mesh.med")
    edge_groups = extract_edge_groups_from_med("./mesh/mesh.med")
    
    bc_manager = BoundaryConditionManager(mesh, edge_groups)
    
    # Set default: all boundaries have Dirichlet u=0
    bc_manager.add_bc_to_all_boundaries("dirichlet", 0.0)
    
    # Override specific boundaries
    bc_manager.add_bc_by_group("inlet", "dirichlet", 1.0)  # This overrides global
    bc_manager.add_bc_by_group("outlet", "neumann", 0.0)   # This overrides global
    
    # All other boundaries (top, bottom, walls, etc.) will use u=0 Dirichlet
    
    solver = P1DGPoissonSolverWithBC(mesh, bc_manager)
    f = lambda x, y: 0.0
    u_dofs = solver.solve(f)
    
    return u_dofs


def example_global_with_analytical_override():
    """
    Example: Global BC with analytical function overrides
    """
    from med_reader import load_med_mesh_mc
    from bc_handler import BoundaryConditionManager, P1DGPoissonSolverWithBC
    
    mesh = load_med_mesh_mc("./mesh/mesh.med")
    bc_manager = BoundaryConditionManager(mesh)
    
    # Default: Neumann BC (no flux) everywhere
    bc_manager.add_bc_to_all_boundaries("neumann", 0.0)
    
    # Override: Dirichlet on left wall only
    bc_manager.add_bc_by_function(
        region_func=lambda x, y: abs(x) < 1e-10,
        bc_type="dirichlet",
        value_func=1.0,
        name="left_inlet"
    )
    
    # All other boundaries keep the Neumann BC
    
    solver = P1DGPoissonSolverWithBC(mesh, bc_manager)
    f = lambda x, y: 0.0
    u_dofs = solver.solve(f)
    
    return u_dofs


def example_insulated_boundaries():
    """
    Example: Thermal problem with insulated (Neumann) boundaries by default
    """
    from med_reader import load_med_mesh_mc
    from bc_handler import BoundaryConditionManager, P1DGPoissonSolverWithBC
    
    mesh = load_med_mesh_mc("./mesh/mesh.med")
    bc_manager = BoundaryConditionManager(mesh)
    
    # Most boundaries are insulated (zero heat flux)
    bc_manager.add_bc_to_all_boundaries("neumann", 0.0)
    
    # Specify temperature only at inlet and outlet
    bc_manager.add_bc_by_function(
        region_func=lambda x, y: abs(x) < 1e-10,
        bc_type="dirichlet",
        value_func=100.0,  # Hot inlet
        name="hot_inlet"
    )
    
    bc_manager.add_bc_by_function(
        region_func=lambda x, y: abs(x - 1.0) < 1e-10,
        bc_type="dirichlet",
        value_func=20.0,  # Cold outlet
        name="cold_outlet"
    )
    
    # Top and bottom are insulated (use global Neumann BC)
    
    solver = P1DGPoissonSolverWithBC(mesh, bc_manager)
    f = lambda x, y: 0.0  # No internal heat source
    u_dofs = solver.solve(f)
    
    return u_dofs


def example_priority_demonstration():
    """
    Demonstrate the priority system:
    1. MED groups (highest priority)
    2. Analytical functions
    3. Global boundary BC
    4. Default (lowest priority)
    """
    from med_reader import load_med_mesh_mc
    from bc_handler import BoundaryConditionManager, P1DGPoissonSolverWithBC
    
    mesh = load_med_mesh_mc("./mesh/mesh.med")
    edge_groups = extract_edge_groups_from_med("./mesh/mesh.med")
    
    bc_manager = BoundaryConditionManager(mesh, edge_groups)
    
    # Priority 4 (lowest): Default is homogeneous Dirichlet
    # This is built-in, no need to set
    
    # Priority 3: Set global BC
    bc_manager.add_bc_to_all_boundaries("neumann", 0.5)
    
    # Priority 2: Analytical function
    bc_manager.add_bc_by_function(
        region_func=lambda x, y: abs(y - 1.0) < 1e-10,
        bc_type="dirichlet",
        value_func=10.0,
        name="top_wall"
    )
    
    # Priority 1 (highest): MED group
    bc_manager.add_bc_by_group("inlet", "dirichlet", 100.0)
    
    # Result:
    # - "inlet" group edges: u=100 (Dirichlet) [Priority 1]
    # - Top wall (y=1): u=10 (Dirichlet) [Priority 2]
    # - All other boundaries: ∂u/∂n=0.5 (Neumann) [Priority 3]
    
    solver = P1DGPoissonSolverWithBC(mesh, bc_manager)
    f = lambda x, y: 0.0
    u_dofs = solver.solve(f)
    
    return u_dofs
```

## EXAMPLE USAGE - MED GROUPS

```python
def example_med_groups():
    """
    Example using boundary conditions from MED file groups
    """
    from med_reader import load_med_mesh_mc
    from bc_handler import BoundaryConditionManager, P1DGPoissonSolverWithBC
    
    # 1. Load mesh
    mesh = load_med_mesh_mc("./mesh/mesh.med")
    
    # 2. Extract edge groups from MED file
    edge_groups = extract_edge_groups_from_med("./mesh/mesh.med")
    
    # 3. Create boundary condition manager
    bc_manager = BoundaryConditionManager(mesh, edge_groups)
    
    # 4. Define boundary conditions for each group using add_bc_by_group
    bc_manager.add_bc_by_group("left", "dirichlet", 1.0)
    bc_manager.add_bc_by_group("right", "dirichlet", 0.0)
    bc_manager.add_bc_by_group("top", "neumann", 0.5)
    bc_manager.add_bc_by_group("bottom", "dirichlet", lambda x, y: np.sin(np.pi * x))
    
    # 5. Create solver and solve
    solver = P1DGPoissonSolverWithBC(mesh, bc_manager, penalty_param=10.0)
    
    def source_term(x, y):
        return 2.0 * np.pi**2 * np.sin(np.pi * x) * np.sin(np.pi * y)
    
    u_dofs = solver.solve(source_term)
    
    # Evaluate solution
    for cell_id in range(min(5, mesh.n_cells)):
        centroid = mesh.cell_centroid(cell_id)
        u_val = solver.evaluate_solution(u_dofs, centroid, cell_id)
        print(f"Cell {cell_id}: u({centroid[0]:.3f}, {centroid[1]:.3f}) = {u_val:.6f}")
    
    return u_dofs
```

## EXAMPLE USAGE - ANALYTICAL FUNCTIONS

```python
def example_analytical_boundaries():
    """
    Example using analytical functions to define boundaries
    No MED groups required!
    """
    from med_reader import load_med_mesh_mc
    from bc_handler import BoundaryConditionManager, P1DGPoissonSolverWithBC
    
    # Load mesh (no need for edge groups)
    mesh = load_med_mesh_mc("./mesh/mesh.med")
    
    # Create BC manager without edge groups
    bc_manager = BoundaryConditionManager(mesh)
    
    # Define boundaries analytically
    # Assume domain is [0,1] x [0,1]
    
    # Left boundary: x ≈ 0
    bc_manager.add_bc_by_function(
        region_func=lambda x, y: abs(x) < 1e-10,
        bc_type="dirichlet",
        value_func=1.0,
        name="left_wall"
    )
    
    # Right boundary: x ≈ 1
    bc_manager.add_bc_by_function(
        region_func=lambda x, y: abs(x - 1.0) < 1e-10,
        bc_type="dirichlet",
        value_func=0.0,
        name="right_wall"
    )
    
    # Bottom boundary: y ≈ 0
    bc_manager.add_bc_by_function(
        region_func=lambda x, y: abs(y) < 1e-10,
        bc_type="dirichlet",
        value_func=0.0,
        name="bottom_wall"
    )
    
    # Top boundary: y ≈ 1
    bc_manager.add_bc_by_function(
        region_func=lambda x, y: abs(y - 1.0) < 1e-10,
        bc_type="neumann",
        value_func=1.0,
        name="top_wall"
    )
    
    # Solve
    solver = P1DGPoissonSolverWithBC(mesh, bc_manager, penalty_param=10.0)
    
    f = lambda x, y: -1.0
    u_dofs = solver.solve(f)
    
    return u_dofs


def example_circular_domain():
    """
    Example with circular boundary detection
    """
    from med_reader import load_med_mesh_mc
    from bc_handler import BoundaryConditionManager, P1DGPoissonSolverWithBC
    
    mesh = load_med_mesh_mc("./mesh/circle.med")
    bc_manager = BoundaryConditionManager(mesh)
    
    # Circular boundary: r ≈ 1 (centered at origin)
    def is_on_circle(x, y, radius=1.0, center=(0.0, 0.0), tol=0.05):
        r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
        return abs(r - radius) < tol
    
    bc_manager.add_bc_by_function(
        region_func=is_on_circle,
        bc_type="dirichlet",
        value_func=lambda x, y: x**2 + y**2,
        name="circle_boundary"
    )
    
    solver = P1DGPoissonSolverWithBC(mesh, bc_manager, penalty_param=10.0)
    f = lambda x, y: -4.0  # Laplacian of x^2 + y^2
    u_dofs = solver.solve(f)
    
    return u_dofs
```

## EXAMPLE USAGE - MIXED MODE (MED GROUPS + ANALYTICAL)

```python
def example_mixed_mode():
    """
    Example combining MED groups and analytical boundaries
    Priority: MED groups are checked first, then analytical functions
    """
    from med_reader import load_med_mesh_mc
    from bc_handler import BoundaryConditionManager, P1DGPoissonSolverWithBC
    
    # Load mesh and groups
    mesh = load_med_mesh_mc("./mesh/mesh.med")
    edge_groups = extract_edge_groups_from_med("./mesh/mesh.med")
    
    bc_manager = BoundaryConditionManager(mesh, edge_groups)
    
    # Use MED groups for some boundaries
    bc_manager.add_bc_by_group("inlet", "dirichlet", lambda x, y: 4*y*(1-y))
    bc_manager.add_bc_by_group("outlet", "neumann", 0.0)
    
    # Use analytical functions for other boundaries
    # These will only apply to edges NOT in the MED groups
    bc_manager.add_bc_by_function(
        region_func=lambda x, y: abs(y) < 1e-10,
        bc_type="dirichlet",
        value_func=0.0,
        name="bottom"
    )
    
    bc_manager.add_bc_by_function(
        region_func=lambda x, y: abs(y - 1.0) < 1e-10,
        bc_type="dirichlet",
        value_func=0.0,
        name="top"
    )
    
    # Solve
    solver = P1DGPoissonSolverWithBC(mesh, bc_manager, penalty_param=10.0)
    f = lambda x, y: 0.0
    u_dofs = solver.solve(f)
    
    return u_dofs
```

## ADVANCED EXAMPLES

```python
def example_complex_analytical():
    """
    Complex analytical boundary detection examples
    """
    from med_reader import load_med_mesh_mc
    from bc_handler import BoundaryConditionManager, P1DGPoissonSolverWithBC
    
    mesh = load_med_mesh_mc("./mesh/mesh.med")
    bc_manager = BoundaryConditionManager(mesh)
    
    # Example 1: Parabolic boundary
    def parabolic_boundary(x, y):
        return abs(y - x**2) < 0.01
    
    bc_manager.add_bc_by_function(
        region_func=parabolic_boundary,
        bc_type="dirichlet",
        value_func=100.0,
        name="parabola"
    )
    
    # Example 2: Multiple line segments
    def left_or_bottom(x, y):
        return (abs(x) < 1e-10) or (abs(y) < 1e-10)
    
    bc_manager.add_bc_by_function(
        region_func=left_or_bottom,
        bc_type="dirichlet",
        value_func=0.0,
        name="left_or_bottom"
    )
    
    # Example 3: Angular sector
    def angular_sector(x, y):
        angle = np.arctan2(y, x)
        return (angle > 0) and (angle < np.pi/4)
    
    bc_manager.add_bc_by_function(
        region_func=angular_sector,
        bc_type="neumann",
        value_func=lambda x, y: np.sin(5*x),
        name="sector"
    )
    
    # Example 4: Distance-based
    def near_point(x, y, px=0.5, py=0.5, radius=0.1):
        dist = np.sqrt((x-px)**2 + (y-py)**2)
        return abs(dist - radius) < 0.01
    
    bc_manager.add_bc_by_function(
        region_func=near_point,
        bc_type="dirichlet",
        value_func=50.0,
        name="circle_around_point"
    )
    
    solver = P1DGPoissonSolverWithBC(mesh, bc_manager, penalty_param=10.0)
    f = lambda x, y: 1.0
    u_dofs = solver.solve(f)
    
    return u_dofs
```

## EXAMPLES WITH HELPER FUNCTIONS FOR COMMON BOUNDARIES

```python
def create_rectangle_bc(xmin, xmax, ymin, ymax, tol=1e-10):
    """
    Helper to create boundary detection functions for a rectangle
    
    Returns:
    --------
    dict with keys 'left', 'right', 'bottom', 'top'
    """
    return {
        'left': lambda x, y: abs(x - xmin) < tol,
        'right': lambda x, y: abs(x - xmax) < tol,
        'bottom': lambda x, y: abs(y - ymin) < tol,
        'top': lambda x, y: abs(y - ymax) < tol,
    }

def create_circle_bc(center, radius, tol=0.01):
    """Helper to create circular boundary detection function"""
    cx, cy = center
    return lambda x, y: abs(np.sqrt((x-cx)**2 + (y-cy)**2) - radius) < tol


def example_using_helpers():
    """Example using helper functions"""
    from med_reader import load_med_mesh_mc
    from bc_handler import BoundaryConditionManager, P1DGPoissonSolverWithBC
    
    mesh = load_med_mesh_mc("./mesh/mesh.med")
    bc_manager = BoundaryConditionManager(mesh)
    
    # Create rectangle boundaries
    rect_funcs = create_rectangle_bc(0.0, 1.0, 0.0, 1.0)
    
    bc_manager.add_bc_by_function(rect_funcs['left'], "dirichlet", 1.0, "left")
    bc_manager.add_bc_by_function(rect_funcs['right'], "dirichlet", 0.0, "right")
    bc_manager.add_bc_by_function(rect_funcs['bottom'], "dirichlet", 0.0, "bottom")
    bc_manager.add_bc_by_function(rect_funcs['top'], "neumann", 0.0, "top")
    
    solver = P1DGPoissonSolverWithBC(mesh, bc_manager)
    f = lambda x, y: 1.0
    u_dofs = solver.solve(f)
    
    return u_dofs
```