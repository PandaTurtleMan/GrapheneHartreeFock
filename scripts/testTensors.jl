include("../src/Tensors.jl")

function run_verification_test()
    println("--- Starting Verification Test ---")
    harmonicRange = 0
    # --- 1. Define physical parameters ---
    N_n = 1      # Number of Landau levels
    N_λ = 2      # Number of sublattice/pseudospin states
    N_s = 2      # Number of spin states
    N_K = 2      # Number of valley states
    N_l = 3    # Number of superlattice bands
    kxRadius = 1
    kyRadius = 1
    N_kx = 2*kxRadius
    N_ky = 2*kyRadius
    N_G = 2*harmonicRange + 1      # Reciprocal lattice vector grid size (NG x NG)
    L = 10.0     # System size
    l_B = 1.0    # Magnetic length
    ε = 1.0      # Dielectric constant
    p_supercell = 1
    q_supercell = 1
    Ky = 2π/L

    # --- 2. Define indices ---
    i_n = Index(2*N_n+1, "n")
    #i_λ = Index(N_λ, "λ")
    i_s = Index(N_s, "s")
    i_K = Index(N_K, "K")
    i_l = Index(N_l, "l")
    orbital_indices = (i_n, i_s, i_K, i_l)

    ikx = Index(N_kx, "kx")
    iky = Index(N_ky, "ky")
    momentum_indices = (ikx, iky)

    iqx = Index(N_kx, "qx")
    iqy = Index(N_ky, "qy")
    q_indices = (iqx, iqy)

    iGx = Index(N_G, "Gx")
    iGy = Index(N_G, "Gy")
    G_indices = (iGx, iGy)

    # Fixed momentum transfer for direct term and shift for exchange term
    Q_val = (1,1) # Corresponds to the first element of the k-grid, bottom left corner
    iQx = iqx(Q_val[1])
    iQy = iqy(Q_val[2])
    Q_indices = (iQx, iQy)

    # --- 3. Define grids ---

    k_step_x = 2π/(L * (2*kxRadius))
    k_step_y = 2π/(q_supercell * L * (2*kyRadius))
    k_grid_vals_x = [n * k_step_x for n in -kxRadius:(kxRadius-1)]
    k_grid_vals_y = [n * k_step_y for n in -kyRadius:(kyRadius-1)]

    q_grid = (k_grid_vals_x, k_grid_vals_y)

    g_max = div(N_G - 1, 2)
    G_grid_vals = range(-g_max, g_max, length=N_G)
    G_vectors = (G_grid_vals, G_grid_vals)

    # --- 4. Pre-compute tensors ---
    println("\n--- Pre-computation Step ---")
    V_full = precomputePotentialTensor(q_indices, G_indices, q_grid, G_vectors, tanh, ε, L)
    S_core_full = precomputeFormFactorTensorCore((i_n), q_indices, G_indices, q_grid, G_vectors, L, l_B)
    S_neg_q_core_full = precomputeFormFactorSnegQ((i_n), q_indices, G_indices, q_grid, G_vectors, L, l_B)

    Phase_D_full = precomputeDirectPhaseTensor(orbital_indices, momentum_indices, q_indices, G_indices, q_grid, G_vectors, L, l_B, p_supercell, q_supercell, Ky)
    Phase_X_full = precomputeExchangePhaseTensor(orbital_indices, momentum_indices, q_indices, Q_val, G_indices, q_grid, G_vectors, L, l_B, p_supercell, q_supercell, Ky)
    Shift, ikx_p, iky_p = precomputeConvolutionTensor(momentum_indices, q_indices, Q_val,kxRadius,kyRadius)

    # --- 5. Create a dummy density matrix Δ ---
    # Δ is diagonal in all orbital indices and momentum
    println("\n--- Building Dummy Density Matrix Δ ---")
    Δ = ITensor(orbital_indices'..., orbital_indices..., ikx', iky')
    for idx_k_x in 1:N_kx, idx_k_y in 1:N_ky
        for idx_n in 1:(2*N_n+1), idx_s in 1:N_s, idx_K in 1:N_K, idx_l in 1:N_l
            # Let's say only the lowest LL is occupied
            if idx_n == 1
                Δ[i_n'(idx_n),i_s'(idx_s), i_K'(idx_K), i_l'(idx_l),
                    i_n(idx_n), i_s(idx_s), i_K(idx_K), i_l(idx_l),
                    ikx'(idx_k_x), iky'(idx_k_y)] = 1.0
            end
        end
    end
    println("Norm of Δ: ", norm(Δ))

    println("\n--- Building Hamiltonian Terms ---")
    println("Building Direct Term (Tensor Network)...")
    H_direct = buildDirectTerm(Δ, V_full, S_core_full, S_neg_q_core_full, Phase_D_full,
                                Q_indices, orbital_indices, momentum_indices)
    println("Building Exchange Term (Tensor Network)...")
    H_exchange = buildExchangeTerm(Δ, V_full, S_core_full, S_neg_q_core_full, Phase_X_full,
                                    Shift, ikx_p, iky_p, orbital_indices, momentum_indices)
    print("Norm of H_exchange,",norm(H_exchange))

    println("\n--- Building Naive Terms ---")
    println("Building Direct Term (Naive)...")
    H_direct_naive = naiveDirectTerm(Δ, V_full, S_core_full, S_neg_q_core_full, Phase_D_full,
                                        orbital_indices, momentum_indices, q_grid, G_vectors, Q_val,
                                        q_indices, G_indices)
    println("Building Exchange Term (Naive)...")
    H_exchange_naive = naiveExchangeTerm(Δ, V_full, S_core_full, S_neg_q_core_full, Phase_X_full,
                                            orbital_indices, momentum_indices, q_grid, G_vectors, Q_val,
                                            q_indices, G_indices, L, l_B, p_supercell, q_supercell, Ky)

    println("\n--- Comparing Results ---")
    # Convert tensor network results to arrays
    H_direct_arr = permutedims(array(H_direct), [1,2,3,4,5, 6,7,8,9,10])
    H_exchange_arr = permutedims(array(H_exchange), [1,2,3,4,5, 6,7,8,9,10])

    # Calculate differences
    diff_direct = norm(H_direct_arr - H_direct_naive)
    diff_exchange = norm(H_exchange_arr - H_exchange_naive)

    println("Norm of Difference (Direct Term):   ", diff_direct)
    println("Norm of Difference (Exchange Term): ", diff_exchange)

    if diff_direct < 1e-10 && diff_exchange < 1e-10
        println("SUCCESS: Tensor network matches naive implementation!")
    else
        println("WARNING: Discrepancy between tensor network and naive implementation")
    end
end



#------------------------------------------------------------------------------
# SCRIPT EXECUTION
#------------------------------------------------------------------------------

# Run the verification test
run_verification_test()