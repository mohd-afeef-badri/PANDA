import numpy as np
from DG_P1 import *
import manufactured_solutions as manufactured_solutions
from med_reader import *

def test_p1dg_poisson_accuracy():
    print("Testing P1 DG Poisson Solver Accuracy on Boundary Layer Problem")
    mesh = create_square_mesh(n=51)
    u_exact, f, g, _ = manufactured_solutions.boundary_layer()

    bc = BoundaryConditionManager(mesh)
    bc.add_bc_to_all_boundaries("dirichlet", g)

    solver = P1DGPoissonSolver(mesh, bc, penalty_param=10.0)
    u_dofs = solver.solve(f)

    errors = []
    for cell_id in range(mesh.n_cells):
        x, y = mesh.cell_centroid(cell_id)
        u_num = solver.evaluate_solution(u_dofs, (x, y), cell_id)
        errors.append(abs(u_num - u_exact(x, y)))

    max_error = max(errors)
    l2_error = np.sqrt(np.mean(np.array(errors)**2))

    assert max_error < 2e-2
    assert l2_error < 1e-2

def test_convergence_rate():
    print("Testing P1 DG Poisson Solver Convergence Rate on Smooth Problem")
    hs = []
    errors = []

    for n in [4, 8, 16]:
        mesh = create_square_mesh(n=n)
        u_exact, f, g, _ = manufactured_solutions.smooth_sin_cos()

        bc = BoundaryConditionManager(mesh)
        bc.add_bc_to_all_boundaries("dirichlet", g)

        solver = P1DGPoissonSolver(mesh, bc, penalty_param=10)
        u = solver.solve(f)

        err = []
        for cid in range(mesh.n_cells):
            x, y = mesh.cell_centroid(cid)
            err.append((solver.evaluate_solution(u, (x,y), cid)
                        - u_exact(x,y))**2)

        errors.append(np.sqrt(np.mean(err)))
        hs.append(1.0 / n)

    rate = np.log(errors[0]/errors[-1]) / np.log(hs[0]/hs[-1])
    assert rate > 1.7

def test_penalty_stability():
    print("Testing P1 DG Poisson Solver Stability with Varying Penalty Parameters")
    mesh = create_square_mesh(n=5)
    u_exact, f, g, _ = manufactured_solutions.smooth_sin_cos()

    bc = BoundaryConditionManager(mesh)
    bc.add_bc_to_all_boundaries("dirichlet", g)

    penalties = [2.0, 5.0, 10.0]
    errors = []

    for γ in penalties:
        solver = P1DGPoissonSolver(mesh, bc, penalty_param=γ)
        u = solver.solve(f)

        e = []
        for cid in range(mesh.n_cells):
            x, y = mesh.cell_centroid(cid)
            e.append(abs(solver.evaluate_solution(u, (x,y), cid)
                         - u_exact(x,y)))
        errors.append(max(e))

    # Regression condition: no penalty blows up
    assert max(errors) / min(errors) < 3.0

