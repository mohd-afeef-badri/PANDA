"""Boundary condition management utilities."""


class BoundaryCondition:
    """Base class for boundary conditions"""
    def __init__(self, bc_type, value_func):
        """
        Parameters:
        -----------
        bc_type : str
            'dirichlet' or 'neumann'
        value_func : callable
            Function f(x, y) that returns the BC value at point (x, y)
        """
        self.bc_type = bc_type.lower()
        self.value_func = value_func
    
    def evaluate(self, x, y):
        """Evaluate the boundary condition at point (x, y)"""
        return self.value_func(x, y)


class BoundaryConditionManager:
    """Manages boundary conditions for different edge groups or analytical regions"""
    def __init__(self, mesh, edge_groups=None):
        """
        Parameters:
        -----------
        mesh : PolygonalMesh
            The mesh object
        edge_groups : dict, optional
            Dictionary mapping group names to arrays of boundary edge indices
            Format: {'group_name': np.array([edge_indices...])}
            If None, only analytical boundary detection will be used
        """
        self.mesh = mesh
        self.edge_groups = edge_groups if edge_groups is not None else {}
        self.conditions = {}
        self.analytical_conditions = []
        self.global_boundary_condition = None  # For all boundaries
        self.default_dirichlet = BoundaryCondition('dirichlet', lambda x, y: 0.0)
        
        # Map each boundary edge to its group
        self.edge_to_group = {}
        for group_name, edge_indices in self.edge_groups.items():
            for edge_idx in edge_indices:
                self.edge_to_group[edge_idx] = group_name
    
    def add_bc_by_group(self, group_name, bc_type, value_func):
        """
        Add a boundary condition to a specific edge group from MED file
        
        Parameters:
        -----------
        group_name : str
            Name of the edge group
        bc_type : str
            'dirichlet' or 'neumann'
        value_func : callable or float
            Function f(x, y) or constant value for the BC
        """
        if isinstance(value_func, (int, float)):
            const_val = value_func
            value_func = lambda x, y, v=const_val: v
        
        self.conditions[group_name] = BoundaryCondition(bc_type, value_func)
    
    def add_bc_by_function(self, region_func, bc_type, value_func, name=None, tolerance=1e-10):
        """
        Add a boundary condition defined by an analytical function
        
        Parameters:
        -----------
        region_func : callable
            Function f(x, y) that returns True if point (x, y) is on this boundary
            Example: lambda x, y: abs(x - 1.0) < 1e-10  (right boundary at x=1)
        bc_type : str
            'dirichlet' or 'neumann'
        value_func : callable or float
            Function f(x, y) or constant value for the BC
        name : str, optional
            Name for this boundary region (for debugging)
        tolerance : float, default=1e-10
            Tolerance for region function evaluation
        """
        if isinstance(value_func, (int, float)):
            const_val = value_func
            value_func = lambda x, y, v=const_val: v
        
        bc = BoundaryCondition(bc_type, value_func)
        
        # Store as tuple: (region_func, bc, name)
        self.analytical_conditions.append((region_func, bc, name, tolerance))
        
        if name:
            print(f"Added analytical BC '{name}' ({bc_type})")
    
    def add_bc_to_all_boundaries(self, bc_type, value_func):
        """
        Add a boundary condition to ALL boundary edges (global fallback)
        This will apply to any boundary edge that doesn't match a specific group or analytical condition
        
        Parameters:
        -----------
        bc_type : str
            'dirichlet' or 'neumann'
        value_func : callable or float
            Function f(x, y) or constant value for the BC
        """
        if isinstance(value_func, (int, float)):
            const_val = value_func
            value_func = lambda x, y, v=const_val: v
        
        self.global_boundary_condition = BoundaryCondition(bc_type, value_func)
        print(f"Set global boundary condition: {bc_type}")
    
    def set_default_bc(self, bc_type, value_func):
        """
        Alias for add_bc_to_all_boundaries for clarity
        Sets the default boundary condition for unspecified boundaries
        """
        self.add_bc_to_all_boundaries(bc_type, value_func)
    
    def get_bc(self, edge_id):
        """
        Get boundary condition for a specific edge
        Priority: 1) MED group, 2) Analytical function, 3) Global boundary BC, 4) Default
        
        Returns:
        --------
        BoundaryCondition or None
        """
        # First check if edge belongs to a MED group
        group_name = self.edge_to_group.get(edge_id)
        if group_name is not None:
            bc = self.conditions.get(group_name)
            if bc is not None:
                return bc
        
        # Check analytical conditions
        edge_mid = self.mesh.edge_midpoint(edge_id)
        x, y = edge_mid[0], edge_mid[1]
        
        for region_func, bc, name, tol in self.analytical_conditions:
            try:
                if region_func(x, y):
                    return bc
            except:
                # If region_func fails, skip this condition
                continue
        
        # Check if there's a global boundary condition
        if self.global_boundary_condition is not None:
            return self.global_boundary_condition
        
        # Default to homogeneous Dirichlet
        return self.default_dirichlet


__all__ = ["BoundaryCondition", "BoundaryConditionManager"]
