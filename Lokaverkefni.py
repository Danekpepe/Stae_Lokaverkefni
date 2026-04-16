import numpy as np
from scipy.sparse import diags, csr_matrix
from scipy.sparse.linalg import spsolve


def heatwave(L, T, N, M, psi, phi, f, kappa=1.0):
    """
    Solve the 1D heat equation using finite differences.
    
    The heat equation: ∂ₜu = κ∂ₓ²u
    
    Boundary conditions:
        u(0, t) = psi(t)
        u(L, t) = phi(t)
    Initial condition:
        u(x, 0) = f(x)
    
    Parameters:
    -----------
    L : float
        Length of the spatial domain [0, L]
    T : float
        Total time duration [0, T]
    N : int
        Number of spatial grid points (including boundaries)
    M : int
        Number of time steps
    psi : callable or float
        Left boundary condition function psi(t) or constant value
    phi : callable or float
        Right boundary condition function phi(t) or constant value
    f : callable or array
        Initial condition function f(x) or array of values on the spatial grid
    kappa : float
        Thermal diffusivity coefficient (default: 1.0)
    
    Returns:
    --------
    HW : ndarray
        Solution matrix of shape (N+1, M+1) where N is the number of spatial points
    """
    
    # Grid setup
    h = L / N
    tau = T / M
    sigma = kappa * tau / (h ** 2)
    
    # Spatial grid
    x = np.linspace(0, L, N + 1)
    
    # Initial profile
    u = f(x) if callable(f) else np.asarray(f)
    
    # Storage for solution at all time steps
    HW = np.zeros((M + 1, N + 1))
    HW[0, :] = u
    
    # Construct the system matrix for interior nodes only
    n_interior = N + 1
    main_diag = np.ones(n_interior) * (1 + 2 * sigma)
    off_diag = np.ones(n_interior - 1) * (-sigma)
    A = diags([off_diag, main_diag, off_diag], [-1, 0, 1], shape=(n_interior, n_interior), format='csr')
    
    # Time stepping
    for m in range(M):
        t_next = (m + 1) * tau
        
        left_bc = psi(t_next) if callable(psi) else psi
        right_bc = phi(t_next) if callable(phi) else phi
        
        if callable(f):
            f_val = f(x)
        else:
            f_val = np.asarray(f)[1:-1]
        
        b = u.copy() + tau * f_val
        b[0] = sigma * left_bc
        b[-1] = sigma * right_bc
        
        u_interior = spsolve(A, b)
        u[1:-1] = u_interior[1:-1]
        u[0] = left_bc
        u[-1] = right_bc
        HW[m+1, :] = u
    
    return HW


def PoissonEq(L1, L2, h, k, ub, ut, vl, vr, f):
    """
    Solve the 2D Poisson equation using Finite Element Method (FEM).
    
    The Poisson equation: -Δφ = f on domain [0, L1] × [0, L2]
    
    Parameters:
    -----------
    L1 : float
        Length of domain in x-direction
    L2 : float
        Length of domain in y-direction
    h : float
        Grid spacing in x-direction
    k : float
        Grid spacing in y-direction
    ub : float or callable
        Boundary condition at y = 0 (bottom)
    ut : float or callable
        Boundary condition at y = L2 (top)
    vl : float or callable
        Boundary condition at x = 0 (left)
    vr : float or callable
        Boundary condition at x = L1 (right)
    f : callable or ndarray
        Source term f(x, y)
    
    Returns:
    --------
    PE : ndarray
        Solution array with shape corresponding to the triangular mesh nodes
    """
    
    # Create grid
    x = np.arange(0, L1 + h, h)
    y = np.arange(0, L2 + k, k)
    nx = len(x)
    ny = len(y)
    
    # Total number of interior nodes (excluding boundaries)
    # For a rectangular mesh divided into triangles
    n_interior = (nx - 2) * (ny - 2)
    n_total = nx * ny
    
    # Create node numbering map
    nodes = np.arange(n_total).reshape(ny, nx)
    interior_nodes = nodes[1:-1, 1:-1].flatten()
    
    # Assemble stiffness matrix A and load vector b
    # Using triangular elements with linear basis functions
    
    A = np.zeros((n_interior, n_interior))
    b = np.zeros(n_interior)
    
    # Compute element contributions
    for i, node_idx in enumerate(interior_nodes):
        node_y, node_x = np.unravel_index(node_idx, (ny, nx))
        x_coord = x[node_x]
        y_coord = y[node_y]
        
        # Assemble local stiffness contributions
        # Coefficient from finite difference: (1/h² + 1/k²)
        diag_coeff = 2 * (1 / (h ** 2) + 1 / (k ** 2))
        A[i, i] += diag_coeff
        
        # Neighbors
        neighbors = []
        neighbor_coeff = []
        
        if node_x > 0:  # left
            neighbors.append(nodes[node_y, node_x - 1])
            neighbor_coeff.append(-1 / (h ** 2))
        if node_x < nx - 1:  # right
            neighbors.append(nodes[node_y, node_x + 1])
            neighbor_coeff.append(-1 / (h ** 2))
        if node_y > 0:  # down
            neighbors.append(nodes[node_y - 1, node_x])
            neighbor_coeff.append(-1 / (k ** 2))
        if node_y < ny - 1:  # up
            neighbors.append(nodes[node_y + 1, node_x])
            neighbor_coeff.append(-1 / (k ** 2))
        
        # Add neighbor contributions
        for neighbor, coeff in zip(neighbors, neighbor_coeff):
            if neighbor in interior_nodes:
                j = np.where(interior_nodes == neighbor)[0][0]
                A[i, j] += coeff
        
        # Load term
        if callable(f):
            f_val = f(x_coord, y_coord)
        else:
            f_val = f
        
        b[i] = f_val
    
    # Boundary contributions
    for i, node_idx in enumerate(interior_nodes):
        node_y, node_x = np.unravel_index(node_idx, (ny, nx))
        
        # Check and apply boundary conditions
        if node_x == 1:  # near left boundary
            if callable(vl):
                b[i] += vl(y[node_y]) / (h ** 2)
            else:
                b[i] += vl / (h ** 2)
        
        if node_x == nx - 2:  # near right boundary
            if callable(vr):
                b[i] += vr(y[node_y]) / (h ** 2)
            else:
                b[i] += vr / (h ** 2)
        
        if node_y == 1:  # near bottom boundary
            if callable(ub):
                b[i] += ub(x[node_x]) / (k ** 2)
            else:
                b[i] += ub / (k ** 2)
        
        if node_y == ny - 2:  # near top boundary
            if callable(ut):
                b[i] += ut(x[node_x]) / (k ** 2)
            else:
                b[i] += ut / (k ** 2)
    
    # Solve the system
    u_interior = np.linalg.solve(A, b)
    
    # Construct full solution with boundary conditions
    PE = np.zeros((ny, nx))
    
    # Set interior values
    for i, node_idx in enumerate(interior_nodes):
        node_y, node_x = np.unravel_index(node_idx, (ny, nx))
        PE[node_y, node_x] = u_interior[i]
    
    # Set boundary values
    for j in range(nx):
        if callable(ub):
            PE[0, j] = ub(x[j])
        else:
            PE[0, j] = ub
        
        if callable(ut):
            PE[-1, j] = ut(x[j])
        else:
            PE[-1, j] = ut
    
    for i in range(ny):
        if callable(vl):
            PE[i, 0] = vl(y[i])
        else:
            PE[i, 0] = vl
        
        if callable(vr):
            PE[i, -1] = vr(y[i])
        else:
            PE[i, -1] = vr
    
    return PE

