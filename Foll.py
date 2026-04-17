import numpy as np
from scipy.sparse import diags
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
import pandas as pd
import matplotlib.pyplot as plt
import math

def heatwave(L, T, N, M, psi, phi, f, kappa=1.0):
    """
    Heatwave fallið eins og lýst er í verkefninu
    L: Lengd í rúmi
    T: Lengd í tíma
    N: Fjöldi hluta í rúmi (N+1 punkta)
    M: Fjöldi hluta í tíma (M+1 punkta)
    psi: Gildi í x=0
    phi: Gildi í x=L
    f: Upphafsskilyrði (gildi í t=0)
    kappa: Hitaleiðni, alltaf 1 í þessu verkefni
    """

    #Byrjum á því að reikna fastana
    h = L / N
    tau = T / M
    sigma = (tau * kappa) / (h**2)

    #Látum fallið alltaf prenta út sigma svo við getum séð hvort það sé á milli 0 og 0.5
    print(f"Sigma: {sigma}")

    # Fjöldi óþekktra staka í HW fylkinu (innri punktar).
    P = (N-1) * M
    
    #Skilgreinum A fylkið og b vigurinn.
    A = lil_matrix((P, P)) #Býr til tómt sparse matrix í réttri stærð
    b = np.zeros(P)
    
    # Fall sem hjálpar okkur að varpa (j, alpha) yfir í 'global' kerfið okkar.
    def get_p(j, alpha):
        return (alpha - 1) * (N-1) + (j - 1)


    #Hérna skilgreinum við A og b í samræmi við hitajöfnuna og skilyrðin.
    for alpha in range(1, M + 1):
        for j in range(1, N):
            p = get_p(j, alpha)
            

            A[p, p] = 1.0
            
            if alpha == 1:
                # These terms involve the initial condition f(x)
                b[p] = (sigma * f((j+1)*h) + 
                        (1 - 2*sigma) * f(j*h) + 
                        sigma * f((j-1)*h))
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

    # Leysum fyrir kerfið og nýtum okkur 'sparse' eiginleika A og b.
    c_internal = spsolve(A.tocsr(), b)
    
    # Skilgreinum allt HW fylkið með 0 gildum.
    HW = np.zeros((M + 1, N + 1))
    
    # Fyllum inn í fyrstu línu HW með upphafsskilyrðinu f(x)
    x_coords = np.linspace(0, L, N + 1)
    HW[0, :] = [f(x) for x in x_coords]
    
    # Fylumm inn í restina af HW
    for alpha in range(1, M + 1):
        t = alpha * tau
        HW[alpha, 0] = psi(t)
        HW[alpha, N] = phi(t)
        
        start = (alpha - 1) * (N-1)
        end = alpha * (N-1)
        HW[alpha, 1:N] = c_internal[start:end]
            
    return HW


def PoissonEq(L1, L2, h, k, ub, ut, vl, vr, f):
    """
    Poisson fallið eins og lýst er í verkefninu
    L1: Lengd í x-stefnu
    L2: Lengd í y-stefnu
    h: Skref í x-stefnu
    k: Skref í y-stefnu
    ub: Gildi í y=0
    ut: Gildi í y=L2
    vl: Gildi í x=0
    vr: Gildi í x=L1
    f: Upphafsskilyrði (gildi í t=0)
    """

    #Byrjum á því að reikna fastana
    N = int(round(L1 / h))
    M = int(round(L2 / k))
    num_nodes = (N + 1) * (M + 1)

    #Skilgreinum A fylkið og b vigurinn.
    A = lil_matrix((num_nodes, num_nodes))
    b = np.zeros(num_nodes)

    # Fall sem hjálpar okkur að varpa (j, p) yfir í 'global' kerfið okkar.
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

    # Leysum fyrir kerfið og nýtum okkur 'sparse' eiginleika A og b.
    sol = spsolve(A.tocsr(), b)

    return sol.reshape((M + 1, N + 1))




def ua_theoretical(x, t, L, N=100):
    """
    Reiknar u_a með gefinni nálgunarformúlu
    x: Staðsetning í rúmi
    t: Tími
    L: Lengd kerfisins
    N: Fjöldi liða í summunni
    """

    #Skilgreinum fasta
    omega = np.pi / L
    ua = 0
    
    #Reiknum summuna
    for n in range(1, N + 1):
        coeff = ((-1)**(n + 1)) / (2 * n - 1)**2
        exponent = -((2 * n - 1)**2) * (omega**2) * t
        trig = np.sin((2 * n - 1) * omega * x)
        
        ua += coeff * np.exp(exponent) * trig
        
    return (8 / (np.pi**2)) * ua



def plot_comparison_table(array_top, array_bottom, x_labels, t_labels, title="Samanburðartafla"):

    """
    Hjálparfall til að búa til samanburðartöflu
    array_top: Fylki1
    array_bottom: Fylki sem á að bera saman við Fylki1
    x_labels: x gildin
    t_labels: t gildin
    """


    if array_top.shape != array_bottom.shape:
        raise ValueError("The two input arrays must have the same dimensions.")
    
    rows, cols = array_top.shape
    if len(x_labels) != cols or len(t_labels) != rows:
        raise ValueError("Label lengths must match the array dimensions.")

  
    combined_data = []
    for i in range(rows):
        row_content = []
        for j in range(cols):
            cell_text = f"{array_top[i, j]}\n{array_bottom[i, j]}"
            row_content.append(cell_text)
        combined_data.append(row_content)


    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('off')
    ax.axis('tight')


    table = ax.table(
        cellText=combined_data,
        colLabels=x_labels,
        rowLabels=t_labels,
        cellLoc='center',
        loc='center'
    )


    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 3) 

    plt.title(title, fontsize=14, pad=20)
    plt.savefig("comparison_table.png", bbox_inches='tight')
    plt.show()




def E_theoretical(t, L, N=100):
    """
    Reiknar E með gefinni nálgunarformúlu
    t: Tími
    L: Lengd kerfisins
    N: Fjöldi liða í summunni
    """

    if t < 0:
        raise ValueError("Time t should be non-negative.")
    
    constant_factor = (16 * L) / (np.pi**4)
    total_sum = 0.0
    
    for n in range(1, N + 1):
        odd_term = 2 * n - 1
        
        exponent = - (2 * (np.pi**2) * (odd_term**2) * t) / (L**2)
        
        term = np.exp(exponent) / (odd_term**4)
        
        total_sum += term
            
    return constant_factor * total_sum


def dE_theoretical(E,dt):
    """
    Reiknar dE/dt út frá E_theoretical.
    dt: Tími á milli E gilda
    """
    dE = []
    for i in range(len(E)-1):
        dE.append((E[i+1] - E[i]) / dt)
    return dE




def E_computed(hw,dt):
    """
    Reiknar orku út frá hw.
    dt: Tímaskref innan hw
    """
    hw_2 = hw**2
    energy = np.sum(hw_2, axis=1)
    return 1/2 * energy*dt

def dE_computed(hw,dt, kappa=1.0):
    """
    Reiknar dE/dt út frá hw.
    dt: Tímaskref innan hw
    """
    du = np.diff(hw, axis=1) / dt
    integral = np.sum(du**2, axis=1)
    return - kappa * integral*dt




def Flux(phi, L1, L2):
    """
    Reiknar flæði út frá phi í gegnum jaðar rétthyrnings

    phi: Rafmættið (fylki)
    L1: Lengd á jaðar í x-stefnu
    L2: Lengd á jaðar í y-stefnu
    """
    Ny, Nx = phi.shape
    dx = L1 / (Nx - 1)
    dy = L2 / (Ny - 1)

    #(x = L1):
    grad_x_right = (phi[:, -1] - phi[:, -2]) / dx
    
    #(x = 0):
    grad_x_left = (phi[:, 1] - phi[:, 0]) / dx
    
    #(y = L2):
    grad_y_top = (phi[-1, :] - phi[-2, :]) / dy
    
    #(y = 0):
    grad_y_bottom = (phi[1, :] - phi[0, :]) / dy

    #Heildun
    int_right  = np.trapezoid(grad_x_right, dx=dy)
    int_left   = np.trapezoid(grad_x_left, dx=dy)
    int_top    = np.trapezoid(grad_y_top, dx=dx)
    int_bottom = np.trapezoid(grad_y_bottom, dx=dx)

    #Leggjum saman og reiknum flæðið
    flux = int_right - int_left + int_top - int_bottom
    
    return flux
