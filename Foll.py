import numpy as np
from scipy.sparse import diags
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
import pandas as pd
import matplotlib.pyplot as plt
import math

def heatwave2(L, T, N, M, psi, phi, f, kappa=1.0):
    """
    Solves 1D Heat Equation by solving the global system Ac = b.
    P = (N-1) * M total internal nodes.
    """
    h = L / N
    tau = T / M
    sigma = (tau * kappa) / (h**2)
    print(f"Sigma: {sigma}")
    
    # Number of internal spatial nodes per time step
    num_x = N - 1
    # Total internal nodes in the global system
    P = num_x * M
    
    A = lil_matrix((P, P))
    b = np.zeros(P)
    
    # Helper to map (j, alpha) to global index p
    # j: spatial index (1 to N-1), alpha: time index (1 to M)
    def get_p(j, alpha):
        return (alpha - 1) * num_x + (j - 1)

    for alpha in range(1, M + 1):
        for j in range(1, N):
            p = get_p(j, alpha)
            
            # The equation: c_{j, alpha} - sigma*c_{j-1, alpha-1} 
            # - (1-2sigma)*c_{j, alpha-1} - sigma*c_{j+1, alpha-1} = 0
            # Note: This is re-arranged from the explicit update to fit Ac = b
            
            # Current node coefficient
            A[p, p] = 1.0
            
            # Previous time step components (alpha - 1)
            if alpha == 1:
                # These terms involve the initial condition f(x)
                term = (sigma * f((j+1)*h) + 
                        (1 - 2*sigma) * f(j*h) + 
                        sigma * f((j-1)*h))
                b[p] = term
            else:
                # These terms involve variables from the previous time row
                # Middle
                A[p, get_p(j, alpha-1)] = -(1 - 2*sigma)
                # Left
                if j > 1:
                    A[p, get_p(j-1, alpha-1)] = -sigma
                else:
                    # Bound by psi at alpha-1
                    b[p] += sigma * psi((alpha-1)*tau)
                # Right
                if j < N - 1:
                    A[p, get_p(j+1, alpha-1)] = -sigma
                else:
                    # Bound by phi at alpha-1
                    b[p] += sigma * phi((alpha-1)*tau)

    # Solve the system
    c_internal = spsolve(A.tocsr(), b)
    
    # Reconstruct the full HW matrix (M+1 x N+1)
    HW = np.zeros((M + 1, N + 1))
    
    # Fill Initial Conditions (Row 0)
    x_coords = np.linspace(0, L, N + 1)
    HW[0, :] = [f(x) for x in x_coords]
    
    # Fill Boundary Conditions and Internal Nodes
    for alpha in range(1, M + 1):
        t = alpha * tau
        HW[alpha, 0] = psi(t)
        HW[alpha, N] = phi(t)
        # Map internal solution vector back to matrix
        start = (alpha - 1) * num_x
        end = alpha * num_x
        HW[alpha, 1:N] = c_internal[start:end]
            
    return HW


def PoissonEq3(L1, L2, h, k, ub, ut, vl, vr, f):
    """
    Solves the 2D Poisson equation on a rectangular domain using a triangular
    mesh of linear finite elements. The rectangular grid is divided into two
    triangles per cell.
    """
    N = int(round(L1 / h))
    M = int(round(L2 / k))
    num_nodes = (N + 1) * (M + 1)

    A = lil_matrix((num_nodes, num_nodes))
    b = np.zeros(num_nodes)

    def get_idx(j, p):
        return p * (N + 1) + j

    def element_matrix(coords):
        # coords is a 3x2 array of triangle vertex coordinates
        x0, y0 = coords[0]
        x1, y1 = coords[1]
        x2, y2 = coords[2]
        det = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
        area = abs(det) * 0.5

        B = np.array([
            [1.0, x0, y0],
            [1.0, x1, y1],
            [1.0, x2, y2],
        ])
        invB = np.linalg.inv(B)
        grads = invB[1:, :].T

        K = np.zeros((3, 3))
        for i in range(3):
            for j in range(3):
                K[i, j] = area * np.dot(grads[i], grads[j])

        return K, area

    for p in range(M):
        for j in range(N):
            n1 = get_idx(j, p)
            n2 = get_idx(j + 1, p)
            n3 = get_idx(j, p + 1)
            n4 = get_idx(j + 1, p + 1)

            coords1 = np.array([
                [j * h, p * k],
                [(j + 1) * h, p * k],
                [j * h, (p + 1) * k],
            ])
            coords2 = np.array([
                [(j + 1) * h, p * k],
                [(j + 1) * h, (p + 1) * k],
                [j * h, (p + 1) * k],
            ])

            K1, area1 = element_matrix(coords1)
            K2, area2 = element_matrix(coords2)

            tri1 = [n1, n2, n3]
            tri2 = [n2, n4, n3]

            for a_local, a_global in enumerate(tri1):
                for b_local, b_global in enumerate(tri1):
                    A[a_global, b_global] += K1[a_local, b_local]

            for a_local, a_global in enumerate(tri2):
                for b_local, b_global in enumerate(tri2):
                    A[a_global, b_global] += K2[a_local, b_local]

            if callable(f):
                f1 = f(coords1[0, 0], coords1[0, 1])
                f2 = f(coords1[1, 0], coords1[1, 1])
                f3 = f(coords1[2, 0], coords1[2, 1])
            else:
                f1 = f2 = f3 = f
            f_avg = (f1 + f2 + f3) * (1.0 / 3.0)
            for a_local, a_global in enumerate(tri1):
                b[a_global] += area1 * f_avg / 3.0

            if callable(f):
                f1 = f(coords2[0, 0], coords2[0, 1])
                f2 = f(coords2[1, 0], coords2[1, 1])
                f3 = f(coords2[2, 0], coords2[2, 1])
            else:
                f1 = f2 = f3 = f
            f_avg = (f1 + f2 + f3) * (1.0 / 3.0)
            for a_local, a_global in enumerate(tri2):
                b[a_global] += area2 * f_avg / 3.0

    for p in range(M + 1):
        for j in range(N + 1):
            idx = get_idx(j, p)
            x, y = j * h, p * k

            if p == 0:
                A[idx, :] = 0
                A[idx, idx] = 1.0
                b[idx] = ub(x) if callable(ub) else ub
            elif p == M:
                A[idx, :] = 0
                A[idx, idx] = 1.0
                b[idx] = ut(x) if callable(ut) else ut
            elif j == 0:
                A[idx, :] = 0
                A[idx, idx] = 1.0
                b[idx] = vl(y) if callable(vl) else vl
            elif j == N:
                A[idx, :] = 0
                A[idx, idx] = 1.0
                b[idx] = vr(y) if callable(vr) else vr

    sol = spsolve(A.tocsr(), b)
    return sol.reshape((M + 1, N + 1))


def solve_ua(x, t, L, N=100):
    omega = np.pi / L
    ua = 0
    
    for n in range(1, N + 1):
        # Calculate the components of the sum
        coeff = ((-1)**(n + 1)) / (2 * n - 1)**2
        exponent = -((2 * n - 1)**2) * (omega**2) * t
        trig = np.sin((2 * n - 1) * omega * x)
        
        # Add to the total sum
        ua += coeff * np.exp(exponent) * trig
        
    return (8 / (np.pi**2)) * ua



def plot_comparison_table(array_top, array_bottom, x_labels, t_labels, title="Comparison Table"):

    # 1. Validation
    if array_top.shape != array_bottom.shape:
        raise ValueError("The two input arrays must have the same dimensions.")
    
    rows, cols = array_top.shape
    if len(x_labels) != cols or len(t_labels) != rows:
        raise ValueError("Label lengths must match the array dimensions.")

    # 2. Combine data into string format with newlines
    combined_data = []
    for i in range(rows):
        row_content = []
        for j in range(cols):
            # Format: Top value on one line, Bottom value on the next
            cell_text = f"{array_top[i, j]}\n{array_bottom[i, j]}"
            row_content.append(cell_text)
        combined_data.append(row_content)

    # 3. Create Visualization
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('off')
    ax.axis('tight')

    # Create the table
    table = ax.table(
        cellText=combined_data,
        colLabels=x_labels,
        rowLabels=t_labels,
        cellLoc='center',
        loc='center'
    )

    # 4. Styling
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    # Increase height (the '3' parameter) to fit the two lines of text
    table.scale(1.2, 3) 

    plt.title(title, fontsize=14, pad=20)
    plt.show()


def theoretical_energy(t, L, num_terms=100, tolerance=1e-12):
    """
    Computes E(t) based on the infinite series:
    E(t) = (16L / pi^4) * sum_{n=1}^inf [ exp(-(2 * pi^2 * (2n-1)^2 / L^2) * t) / (2n-1)^4 ]
    
    Args:
        t (float): Time variable.
        L (float): Length/System constant.
        num_terms (int): Maximum number of terms to sum.
        tolerance (float): Stopping criterion if terms become smaller than this value.
        
    Returns:
        float: The calculated energy E(t).
    """
    if t < 0:
        raise ValueError("Time t should be non-negative.")
    
    constant_factor = (16 * L) / (np.pi**4)
    total_sum = 0.0
    
    for n in range(1, num_terms + 1):
        # Calculate the odd integer term (2n - 1)
        odd_term = 2 * n - 1
        
        # Calculate the exponent part
        exponent = - (2 * (np.pi**2) * (odd_term**2) * t) / (L**2)
        
        # Calculate the specific term in the series
        term = np.exp(exponent) / (odd_term**4)
        
        total_sum += term
        
        # Optimization: break if the term is smaller than the precision limit
        if term < tolerance:
            break
            
    return constant_factor * total_sum

def energychange_theoretical(E,dt):
    dE = []
    for i in range(len(E)-1):
        dE.append((E[i+1] - E[i]) / dt)
    return dE

def compute_energy(hw,dt):
    hw_2 = hw**2
    energy = np.sum(hw_2, axis=1)
    return 1/2 * energy*dt

def compute_energychange(hw,dx, kappa=1.0):
    du = np.diff(hw, axis=1) / dx
    integral = np.sum(du**2, axis=1)
    return - kappa * integral*dx