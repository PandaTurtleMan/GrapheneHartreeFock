using ITensors
using LinearAlgebra
using ProgressMeter
using Random
using Parameters


ITensors.disable_warn_order()
#------------------------------------------------------------------------------
# DUMMY FUNCTION (for testing)
#------------------------------------------------------------------------------
# Note: This script assumes the existence of a function:
# grapheneLandauFourierMatrixElement(qx, qy, n1, n2, l_B, λ1, λ2)
# which is not defined here but is required by the script.
# For testing, we will create a dummy version of it.
function grapheneLandauFourierMatrixElement(qx, qy, n1, n2, l_B, λ1, λ2)
    # Dummy implementation for testing purposes.
    # In a real scenario, this would compute the actual form factor.
    return 1.0
end

#------------------------------------------------------------------------------
# TENSOR PRE-COMPUTATION
#------------------------------------------------------------------------------

"""
Pre-computes the Coulomb potential tensor V(q+G).
"""
function precomputePotentialTensor(q_indices, G_indices, q_grid, G_vectors, screening_fn, ε, L)
    #V = ITensor(q_indices..., G_indices...)
    K = 2 * π / L

    iqx, iqy = q_indices'
        iGx, iGy = G_indices
    qx_vals, qy_vals = q_grid
    Gx_vals, Gy_vals = G_vectors
    V = ITensor(iqx,iqy,iGx,iGy)

    @showprogress "Pre-computing Potential V(q+G)..." for qx_idx in 1:dim(iqx), qy_idx in 1:dim(iqy), Gx_idx in 1:dim(iGx), Gy_idx in 1:dim(iGy)
        q = [qx_vals[qx_idx], qy_vals[qy_idx]]
        G = [Gx_vals[Gx_idx], Gy_vals[Gy_idx]]

        q_total_norm = norm(q + K * G)

        if q_total_norm < 1e-9
            Vval = 2*π/ε
        else
            # A dummy screening function for testing
            screening_val = tanh(q_total_norm)
            Vval = (2 * π / q_total_norm) * screening_val / ε
        end

        V[iqx(qx_idx), iqy(qy_idx), iGx(Gx_idx), iGy(Gy_idx)] = Vval
    end
    return V
end

"""
Pre-computes the core form factor tensor S(n1,λ1; n2,λ2; q,G).
The indices are (n1_out, λ1_out, n2_in, λ2_in, qx, qy, Gx, Gy).
"""
function precomputeFormFactorTensorCore(indices, q_indices, G_indices, q_grid, G_vectors, L, l_B)
    i_n = indices
    iqx, iqy = q_indices'
    iGx, iGy = G_indices'
    S_core = ITensor(i_n, i_n'', iqx,iqy,iGx,iGy)


    qx_vals, qy_vals = q_grid
    Gx_vals, Gy_vals = G_vectors
    K = 2 * π / L

    @showprogress "Pre-computing Form Factor Core S(n,λ,q,G)..." for qx_idx in 1:dim(iqx), qy_idx in 1:dim(iqy), Gx_idx in 1:dim(iGx), Gy_idx in 1:dim(iGy)
        q = [qx_vals[qx_idx], qy_vals[qy_idx]]
        G = [Gx_vals[Gx_idx], Gy_vals[Gy_idx]]
        q_total = q + K * G

        for n1_idx in 1:dim(i_n), n2_idx in 1:dim(i_n)
            n1 = n1_idx - div(dim(i_n) - 1,2) -1
            λ1 =
            (if (n1 < 0)
                -1
            else
                1
            end)
            n2 = n2_idx - div(dim(i_n) - 1,2) -1
            λ2 =
            (if (n2 < 0)
                        -1
            else
                1
            end)
            val = grapheneLandauFourierMatrixElement(q_total[1], q_total[2], n1 - 1, n2 - 1, l_B, (λ1 == 1 ? 1 : -1), (λ2 == 1 ? 1 : -1))
            S_core[i_n(n1_idx), i_n''(n2_idx), iqx(qx_idx), iqy(qy_idx), iGx(Gx_idx), iGy(Gy_idx)] = val
        end
    end
    return S_core
end

"""
Pre-computes the S(-q) form factor tensor using S(-q)_ab = (-1)^(na-nb) * S(q)_ba.
"""
function precomputeFormFactorSnegQ(indices, q_indices, G_indices, q_grid, G_vectors, L, l_B)
    i_n = indices
    iqx, iqy = q_indices''
        iGx, iGy = G_indices
    S_neg_q = ITensor(i_n', i_n''', iqx,iqy,iGx,iGy)


    qx_vals, qy_vals = q_grid
    Gx_vals, Gy_vals = G_vectors
    K = 2 * π / L

    @showprogress "Pre-computing Form Factor S(-q)..." for qx_idx in 1:dim(iqx), qy_idx in 1:dim(iqy), Gx_idx in 1:dim(iGx), Gy_idx in 1:dim(iGy)
        q = [qx_vals[qx_idx], qy_vals[qy_idx]]
        G = [Gx_vals[Gx_idx], Gy_vals[Gy_idx]]
        q_total = q + K * G

        for na_idx in 1:dim(i_n), nb_idx in 1:dim(i_n)
            na = na_idx - div(dim(i_n) - 1,2) -1
            λa =
            (if na < 0
                -1
            else
                1
            end)
            nb = nb_idx - div(dim(i_n) - 1,2) -1
            λb =  (if nb < 0
            -1
            else
                1
            end)
            # Calculate S(q)_ba
            val_S_ba = grapheneLandauFourierMatrixElement(-q_total[1], -q_total[2], nb - 1, na - 1, l_B, (λb == 1 ? 1 : -1), (λa == 1 ? 1 : -1))
            # S(-q)_ab = (-1)^(na-nb) * S(q)_ba
            final_val = val_S_ba
            S_neg_q[i_n'(na_idx), i_n'''(nb_idx), iqx(qx_idx), iqy(qy_idx), iGx(Gx_idx), iGy(Gy_idx)] = final_val
        end
    end
    return S_neg_q
end

"""
Pre-computes the combined phase factor for the Direct Term.
"""
function precomputeDirectPhaseTensor(orbital_indices, momentum_indices, q_indices, G_indices,
                                        q_grid, G_vectors, L, l_B, p_supercell, q_supercell, Ky)
    # Unpack indices
    i_l = orbital_indices[4]
    ikx, iky = momentum_indices
    iqx, _ = q_indices
    iGx, iGy = G_indices

    # Create indices for all 4 states
    l1, l2, l3, l4 = i_l, i_l', i_l'', i_l'''
        kx3, ky3 = ikx, iky
    kx4, ky4 = prime(ikx), prime(iky)

    # Get grid values
    qx_vals, _ = q_grid
    kx_vals, ky_vals = q_grid
    Gx_vals, Gy_vals = G_vectors
    K_sys = 2π / L
    Kx = K_sys

    Phase_D = ITensor(l1, l2, l3, l4, kx3, ky3, kx4, ky4, iqx, iGx, iGy)

    @showprogress "Pre-computing Combined Direct Phase Tensor..." for l1_idx in 1:dim(l1), l2_idx in 1:dim(l2), l3_idx in 1:dim(l3), l4_idx in 1:dim(l4),
        kx3_idx in 1:dim(kx3), ky3_idx in 1:dim(ky3), kx4_idx in 1:dim(kx4), ky4_idx in 1:dim(ky4),
        qx_idx in 1:dim(iqx), Gx_idx in 1:dim(iGx), Gy_idx in 1:dim(iGy)

        Qx = qx_vals[qx_idx]
        _kx3, _ky3 = kx_vals[kx3_idx], ky_vals[ky3_idx]
        _kx4, _ky4 = kx_vals[kx4_idx], ky_vals[ky4_idx]
        Gx, Gy = Gx_vals[Gx_idx], Gy_vals[Gy_idx]

        # Phase 1
        phase1 = exp(im * (Qx + Gx*K_sys) * l_B^2 * (_ky4 - _ky3 + Ky * (l4_idx - l3_idx) / q_supercell))

        # Phase 2
        phase2 = 0.0 + 0.0im
        if mod(l3_idx - l1_idx + q_supercell * Gy / Ky, p_supercell) == 0
            phase2 = exp(im * (2π / (p_supercell * Kx)) * (_kx3 - Qx) * (l3_idx - l1_idx + q_supercell * Gy / Ky))
        end

        # Phase 3
        phase3 = 0.0 + 0.0im
        if mod(l4_idx - l2_idx - q_supercell * Gy / Ky, p_supercell) == 0
            phase3 = exp(im * (2π / (p_supercell * Kx)) * (_kx4 + Qx) * (l4_idx - l2_idx - q_supercell * Gy / Ky))
        end

        Phase_D[l1(l1_idx), l2(l2_idx), l3(l3_idx), l4(l4_idx), kx3(kx3_idx), ky3(ky3_idx), kx4(kx4_idx), ky4(ky4_idx), iqx(qx_idx), iGx(Gx_idx), iGy(Gy_idx)] = phase1 * phase2 * phase3
    end
    return Phase_D
end

"""
Pre-computes the combined phase factor for the Exchange Term.
"""
function precomputeExchangePhaseTensor(
    orbital_indices,
    momentum_indices,
    q_indices,
    Q_val,
    G_indices,
    q_grid,
    G_vectors,
    L,
    l_B,
    p_supercell,
    q_supercell,
    Ky
)
    # Unpack indices
    i_l = orbital_indices[4]
    ikx, iky = momentum_indices
    iqx, iqy = q_indices'
    iGx, iGy = G_indices'

    # Create indices for all 4 states. In exchange, the pairing is 1-4 and 2-3.
    # The external legs are 1 and 2.
    l1, l2 = i_l, i_l'
    l3, l4 = i_l'', i_l'''
    kx, ky = ikx', iky' # External momentum k_2

    # Get grid values
    qx_vals, qy_vals = q_grid
    kx_vals, ky_vals = q_grid
    Gx_vals, Gy_vals = G_vectors
    # Q is a fixed vector, get its value from the grid using the provided integer indices
    Qx_idx, Qy_idx = Q_val
    Qx_val = qx_vals[Qx_idx]
    Qy_val = qy_vals[Qy_idx]
    K_sys = 2π / L
    Kx = K_sys

    # The phase tensor MUST include all external momentum indices (k2x, k2y) to have the correct final structure
    Phase_X = ITensor(l1, l2, l3, l4, kx, ky, iqx, iqy, iGx, iGy)

    @showprogress "Pre-computing Combined Exchange Phase Tensor..." for l1_idx in 1:dim(l1), l2_idx in 1:dim(l2), l3_idx in 1:dim(l3), l4_idx in 1:dim(l4),
        kx_idx in 1:dim(kx), ky_idx in 1:dim(ky), # Loop over k2y as well
        qx_idx in 1:dim(iqx), qy_idx in 1:dim(iqy), Gx_idx in 1:dim(iGx), Gy_idx in 1:dim(iGy)

        _kx = kx_vals[kx_idx]
        _qx, _qy = qx_vals[qx_idx], qy_vals[qy_idx]
        # FIX: Use distinct names for the numeric values of G to avoid overwriting the Index objects.
        _Gx_val, _Gy_val = Gx_vals[Gx_idx], Gy_vals[Gy_idx]

        # Phase 1
        phase1 = exp(im * (_qx + _Gx_val*K_sys) * l_B^2 * (Qy_val - _qy + Ky * (l4_idx - l3_idx) / q_supercell))

        # Phase 2
        phase2 = 0.0 + 0.0im
        if mod(l3_idx - l1_idx + q_supercell * _Gy_val / Ky, p_supercell) == 0
            phase2 = exp(im * (2π / (p_supercell * Kx)) * (_kx - Qx_val) * (l3_idx - l1_idx + q_supercell * _Gy_val / Ky))
        end

        # Phase 3
        phase3 = 0.0 + 0.0im
        if mod(l4_idx - l2_idx - q_supercell * _Gy_val / Ky, p_supercell) == 0
            phase3 = exp(im * (2π / (p_supercell * Kx)) * (_kx + _qx) * (l4_idx - l2_idx - q_supercell * _Gy_val / Ky))
        end

        # FIX: Use the correct Index objects (iGx, iGy, iqx, iqy) for indexing the tensor.
        # The original code used Gx, Gy, qx, qy. Gx and Gy had been overwritten with Float64 values,
        # causing the error. qx and qy were different (primed) indices not present on the tensor.
        Phase_X[l1(l1_idx), l2(l2_idx), l3(l3_idx), l4(l4_idx), kx(kx_idx), ky(ky_idx), iqx(qx_idx), iqy(qy_idx), iGx(Gx_idx), iGy(Gy_idx)] = phase1 * phase2 * phase3
    end
    return Phase_X
end

"""
This tensor shifts the input to the density matrix by k' = k - q - Q.
It is used to implement the convolution in the exchange term sum.
This function now also returns the primed indices to avoid lookup errors.
"""
#make sure this is right...
function precomputeConvolutionTensor(momentum_indices, q_indices, Q_grid_indices, kxRadius,kyRadius)
    ikx, iky = momentum_indices
    iqx, iqy = q_indices
    iQx_idx, iQy_idx = Q_grid_indices
    ikx_p, iky_p = prime(ikx), prime(iky)
    N_kx = dim(ikx)  # Should be 2*kxRadius
    N_ky = dim(iky)  # should be 2*kyradius

    Shift = ITensor(ikx, iky, iqx, iqy, ikx', iky')

    @showprogress "Building convolution tensor..." for kx_idx in 1:dim(ikx), ky_idx in 1:dim(iky), qx_idx in 1:dim(iqx), qy_idx in 1:dim(iqy)
        # Calculate index shifts relative to gamma point (kRadius+1,kRadius+1)
        q_shift_x = (qx_idx - (kxRadius + 1)) - (iQx_idx - (kxRadius + 1))
        q_shift_y = (qy_idx - (kyRadius + 1)) - (iQy_idx - (kyRadius + 1))

        # Apply shifts with periodic boundary conditions
        k_prime_x_idx = mod1(kx_idx + q_shift_x, N_kx)
        k_prime_y_idx = mod1(ky_idx + q_shift_y, N_ky)

        Shift[ikx(kx_idx), iky(ky_idx), iqx(qx_idx), iqy(qy_idx), ikx'(k_prime_x_idx), iky'(k_prime_y_idx)] = 1.0
    end
    return Shift, ikx_p, iky_p
end

#------------------------------------------------------------------------------
# HAMILTONIAN CONSTRUCTION (TENSOR NETWORK)
#------------------------------------------------------------------------------

function buildDirectTerm(Δ::ITensor, V_full::ITensor, S_core_full::ITensor, S_neg_q_core_full::ITensor, Phase_D_full::ITensor,
                            Q_indices, orbital_indices, momentum_indices)

    # Unpack indices
    i_n, i_s, i_K, i_l = orbital_indices
    ikx, iky = momentum_indices
    iQx, iQy = Q_indices

    # Get G indices from one of the tensors that has them
    iGx = findindex(V_full, "Gx")
    iGy = findindex(V_full, "Gy")

    # Initialize output Hamiltonian tensor with zeros

    # Define composite indices and new momentum indices once, as they are G-independent
    Γ1 = (n=i_n, s=i_s,  K=i_K,  l=i_l)
    Γ2 = (n=i_n', s=i_s',   K=i_K',   l=i_l')
    Γ3 = (n=i_n'', s=i_s'', K=i_K'', l=i_l'')
    Γ4 = (n=i_n''',s=i_s''',K=i_K''',l=i_l''')
    ikx4 = prime(ikx)
    iky4 = prime(iky)

    Deltas = delta(Γ1.s, Γ4.s) * delta(Γ2.s, Γ3.s) * delta(Γ1.K, Γ4.K) * delta(Γ2.K, Γ3.K)

    Δ = replaceinds(Δ, (orbital_indices'..., orbital_indices..., ikx, iky), (Γ4..., Γ2..., ikx4, iky4))

    #contract over k first
    Phase_D_slice = Phase_D_full * setelt(iQx)

    D = Δ*Phase_D_slice

    # --- Slice all q-dependent tensors at q = Q ---

    V_slice = V_full * setelt(iQx') * setelt(iQy')

    S_core_slice = S_core_full * setelt(iQx') * setelt(iQy')

    S_neg_q_core_slice = S_neg_q_core_full * setelt(iQx'') * setelt(iQy'')



    #contact over G, again use deltas to avoid premature contractions

    #first add the V tensor
    D = D * delta(iGx,iGx',iGx'')
    D = D* delta(iGy,iGy',iGy'')

    prime!(V_slice,iGx,iGy)
    D = D * V_slice


    #next is S tensor
    D = D * delta(iGx,iGx',iGx'')
    D = D* delta(iGy,iGy',iGy'')
    D = D * S_core_slice


    #last is S neg tensor, now we just contract
    D = D*S_neg_q_core_slice

    #finally the deltas
    D = D*Deltas
    D = setprime(D,0,ikx)
    D = setprime(D,0,iky)
    D = permute(D, i_n, i_s, i_K, i_l, i_n'', i_s'', i_K'', i_l'', ikx, iky)
    #print(inds(D))
    # The sum over the internal momentum k4 was handled by the tensor contraction.
    return D
end

function naiveDirectTerm(
    Δ,
    V_full,
    S_core_full,
    S_neg_q_core_full,
    Phase_D_full,
    orbital_indices,
    momentum_indices,
    q_grid,
    G_vectors,
    Q_val,
    q_indices,
    G_indices
)
    i_n, i_s, i_K, i_l = orbital_indices
    ikx, iky = momentum_indices
    iqx, iqy = q_indices
    iGx, iGy = G_indices

    # Extract grid values
    kx_vals, ky_vals = q_grid
    Gx_vals, Gy_vals = G_vectors
    Qx_idx, Qy_idx = Q_val
    Qx_val = kx_vals[Qx_idx]
    Qy_val = ky_vals[Qy_idx]

    # Dimensions
    dims = (dim(i_n), dim(i_s), dim(i_K), dim(i_l),
            dim(i_n), dim(i_s), dim(i_K), dim(i_l),
            dim(ikx), dim(iky))
    H_direct_naive = zeros(ComplexF64, dims)

    for n1 in 1:dim(i_n), s1 in 1:dim(i_s), K1 in 1:dim(i_K), l1 in 1:dim(i_l),
        n3 in 1:dim(i_n), s3 in 1:dim(i_s), K3 in 1:dim(i_K), l3 in 1:dim(i_l),
        kx3_idx in 1:dim(ikx), ky3_idx in 1:dim(iky)

        for kx4_idx in 1:dim(ikx), ky4_idx in 1:dim(iky),
            n2 in 1:dim(i_n), s2 in 1:dim(i_s), K2 in 1:dim(i_K), l2 in 1:dim(i_l),
            n4 in 1:dim(i_n), s4 in 1:dim(i_s), K4 in 1:dim(i_K), l4 in 1:dim(i_l),
            Gx_idx in 1:dim(iGx), Gy_idx in 1:dim(iGy)

            # Kronecker deltas
            if !(s1 == s4 && s2 == s3 && K1 == K4 && K2 == K3)
                continue
            end

            # Density matrix value
            Δ_val = Δ[i_n'(n4), i_s'(s4), i_K'(K4), i_l'(l4),
                        i_n(n2), i_s(s2), i_K(K2), i_l(l2),
                        ikx'(kx4_idx), iky'(ky4_idx)]

            # Potential and form factors
            V_val = V_full[iqx'(Qx_idx), iqy'(Qy_idx), iGx(Gx_idx), iGy(Gy_idx)]
            S13_val = S_core_full[i_n(n1), i_n''(n3),
                                    iqx'(Qx_idx), iqy'(Qy_idx), iGx'(Gx_idx), iGy'(Gy_idx)]
            S42_val = S_neg_q_core_full[i_n'(n4),  i_n'''(n2),
                                        iqx''(Qx_idx), iqy''(Qy_idx), iGx(Gx_idx), iGy(Gy_idx)]

            # Phase factor
            phase_val = Phase_D_full[i_l(l1), i_l'(l2), i_l''(l3), i_l'''(l4),
                                        ikx(kx3_idx), iky(ky3_idx),
                                        ikx'(kx4_idx), iky'(ky4_idx),
                                        iqx(Qx_idx), iGx(Gx_idx), iGy(Gy_idx)]

            term = Δ_val * V_val * S13_val * S42_val * phase_val
            H_direct_naive[n1, s1, K1, l1, n3, s3, K3, l3, kx3_idx, ky3_idx] += term
        end
    end
    return H_direct_naive
end

function buildExchangeTerm(
    Δ::ITensor,
    VTensor::ITensor,
    STensor::ITensor,
    SNegTensor::ITensor,
    Phase_X::ITensor,
    Shift::ITensor,
    ikx_p::Index,
    iky_p::Index,
    orbital_indices,
    momentum_indices
)

    # Unpack indices
    i_n, i_s, i_K, i_l = orbital_indices
    ikx, iky = momentum_indices
    iqx = findindex(Shift, "qx")
    iqy = findindex(Shift, "qy")
    iGx = findindex(VTensor, "Gx")
    iGy = findindex(VTensor, "Gy")

    # Initialize output Hamiltonian tensor with zeros
    H_exchange = ITensor(ikx, iky, orbital_indices'..., orbital_indices...)

    # --- Define composite indices for the 4 states in the interaction ---
    Γ1 = (n=i_n, s=i_s,  K=i_K,  l=i_l)
    Γ2 = (n=i_n', s=i_s',   K=i_K',   l=i_l')
    Γ3 = (n=i_n'',  s=i_s'', K=i_K'', l=i_l'')
    Γ4 = (n=i_n''',s=i_s''',K=i_K''',l=i_l''')

    # --- Prepare Tensors with Correct Index Structure ---
    Deltas = delta(Γ1.s, Γ4.s) * delta(Γ2.s, Γ3.s) * delta(Γ1.K, Γ4.K) * delta(Γ2.K, Γ3.K)



    # The density matrix is evaluated at a shifted momentum k' = k - q - Q.
    Δ = replaceinds(Δ, (orbital_indices'..., orbital_indices...), (Γ3..., Γ2...))

    E = Δ * Shift


    E = E*delta(ikx,ikx',ikx'')
    #print("\n Norm of E:", norm(E))

    E=E*δ(iky,iky',iky'')
    #print("\n Norm of E:", norm(E))

    E=E*δ(iqx,iqx',iqx'')
    #print("\n Norm of E:", norm(E))

    E=E*δ(iqy,iqy',iqy'')
    #print("\n Norm of E:", norm(E))

    #now contract with phase tensor, deltas force a pointwise multplication over k,q
    E = E*Phase_X
    #print("\n Norm of E:", norm(E))


    #sequentially pointwise multiply each of the parts of the structure factor together with the term
    E = E*delta(iqx,iqx',iqx'')*delta(iqy,iqy',iqy'')*delta(iGx,iGx',iGx'')*delta(iGy,iGy',iGy'')

    E = E*VTensor


    E = E*delta(iqx,iqx',iqx'')*delta(iqy,iqy',iqy'')*delta(iGx,iGx',iGx'')*delta(iGy,iGy',iGy'')

    E = E*STensor

    #this last guy will sum over q and G!
    E = E*SNegTensor

    #now the deltas
    E = E*Deltas

    E = noprime(E, tags=("kx"))
    H_X = noprime(E,tags=("ky"))
    #print(inds(E))

    H_X = permute(H_X, i_n, i_s, i_K, i_l, i_n''', i_s''', i_K''', i_l''', ikx, iky)

    #print(inds(H_X))
    # the final result is negated
    return -1.0 * H_X
end

@with_kw struct ExchangeTermParams
    Δ::ITensor
    V_full::ITensor
    S_core_full::ITensor
    S_neg_q_core_full::ITensor
    Phase_X_full::ITensor
    orbital_indices::Tuple{Index, Index, Index, Index}
    momentum_indices::Tuple{Index, Index}
    q_grid::Tuple{Vector{Float64}, Vector{Float64}}
    G_vectors::Tuple{Vector{Float64}, Vector{Float64}}
    Q_val::Tuple{Float64, Float64}
    q_indices::Tuple{Index, Index}
    G_indices::Tuple{Index, Index}
    L::Float64
    l_B::Float64
    p_supercell::Int64
    q_supercell::Int64
    Ky::Float64
end
function naiveExchangeTerm(params::ExchangeTermParams)
    Δ,
    V_full,
    S_core_full,
    S_neg_q_core_full,
    Phase_X_full,
    orbital_indices,
    momentum_indices,
    q_grid,
    G_vectors,
    Q_val,
    q_indices,
    G_indices,
    L,
    l_B,
    p_supercell,
    q_supercell,
    Ky = params

    i_n, i_s, i_K, i_l = orbital_indices
    ikx, iky = momentum_indices
    iqx, iqy = q_indices
    iGx, iGy = G_indices

    # Extract grid values
    kx_vals, ky_vals = q_grid
    Gx_vals, Gy_vals = G_vectors
    Qx_idx, Qy_idx = Q_val
    Qx_val = kx_vals[Qx_idx]
    Qy_val = ky_vals[Qy_idx]
    K_sys = 2π / L

    # Dimensions
    dims = (dim(i_n), dim(i_s), dim(i_K), dim(i_l),
            dim(i_n), dim(i_s), dim(i_K), dim(i_l),
            dim(ikx), dim(iky))
    H_exchange_naive = zeros(ComplexF64, dims)

    for n1 in 1:dim(i_n), s1 in 1:dim(i_s), K1 in 1:dim(i_K), l1 in 1:dim(i_l),
        n4 in 1:dim(i_n), s4 in 1:dim(i_s), K4 in 1:dim(i_K), l4 in 1:dim(i_l),
        kx4_idx in 1:dim(ikx), ky4_idx in 1:dim(iky)

        # Kronecker deltas for external states
        if !(s1 == s4 && K1 == K4)
            continue
        end

        for qx_idx in 1:dim(iqx), qy_idx in 1:dim(iqy),
            Gx_idx in 1:dim(iGx), Gy_idx in 1:dim(iGy),
            n2 in 1:dim(i_n), s2 in 1:dim(i_s), K2 in 1:dim(i_K), l2 in 1:dim(i_l),
            n3 in 1:dim(i_n), s3 in 1:dim(i_s), K3 in 1:dim(i_K), l3 in 1:dim(i_l)

            # Kronecker deltas for internal states
            if !(s2 == s3 && K2 == K3)
                continue
            end

            # Shifted momentum calculation
            kx4 = kx_vals[kx4_idx]
            ky4 = ky_vals[ky4_idx]
            qx = kx_vals[qx_idx]
            qy = ky_vals[qy_idx]
            kx_prime = kx4 + qx - Qx_val
            ky_prime = ky4 + qy - Qy_val

            # Find nearest grid indices for shifted momentum
            kx_prime_idx = argmin(abs.(kx_vals .- kx_prime))
            ky_prime_idx = argmin(abs.(ky_vals .- ky_prime))

            # Density matrix value at shifted momentum
            Δ_val = Δ[i_n'(n3), i_s'(s3), i_K'(K3), i_l'(l3),
                        i_n(n2), i_s(s2), i_K(K2), i_l(l2),
                        ikx'(kx_prime_idx), iky'(ky_prime_idx)]

            # Potential and form factors
            V_val = V_full[iqx'(qx_idx), iqy'(qy_idx), iGx(Gx_idx), iGy(Gy_idx)]
            S13_val = S_core_full[i_n(n1),  i_n''(n3),
                                    iqx'(qx_idx), iqy'(qy_idx), iGx'(Gx_idx), iGy'(Gy_idx)]
            S42_val = S_neg_q_core_full[i_n'(n4), i_n'''(n2),
                                        iqx''(qx_idx), iqy''(qy_idx), iGx(Gx_idx), iGy(Gy_idx)]

            # Phase factor
            Gx_int = Gx_vals[Gx_idx]
            Gy_int = Gy_vals[Gy_idx]
            phase_val = Phase_X_full[i_l(l1), i_l'(l2), i_l''(l3), i_l'''(l4),
                                        ikx'(kx4_idx), iky'(ky4_idx),
                                        iqx'(qx_idx), iqy'(qy_idx), iGx'(Gx_idx), iGy'(Gy_idx)]

            term = -Δ_val * V_val * S13_val * S42_val * phase_val
            H_exchange_naive[n1,  s1, K1, l1, n4, s4, K4, l4, kx4_idx, ky4_idx] += term
        end
    end
    return H_exchange_naive
end

function partial_trace(rho::ITensor, inds_to_trace::Vector{<:Index})
    # Start with the original tensor. We will contract it with a series of delta tensors.
    traced_rho = rho

    # Iterate over each site index that we want to trace out.
    for i in inds_to_trace
        # Find the primed version of the index.
        ip = prime(i)

        # Check if both the unprimed and primed indices exist on the tensor.
        # This is a sanity check to ensure the operation is valid.
        if !hasinds(traced_rho, i, ip)
            error("Input ITensor must have both the index $i and its primed version $ip to trace over.")
        end

        # Contract with a delta tensor. This effectively sums over the diagonal
        # elements for the subsystem defined by index `i`, which is the definition of a trace.
        traced_rho *= delta(i, ip)
    end

    return traced_rho
end

function lambdaTensor(p, q, L, kGrid, reciVectorGrid, n_levels, l_levels)
    # Extract grid values
    kx_vals, ky_vals = kGrid
    Gx_vals, Gy_vals = reciVectorGrid

    # Create indices
    ikx = Index(length(kx_vals), "kx")
    iky = Index(length(ky_vals), "ky")
    iGx = Index(length(Gx_vals), "Gx")
    iGy = Index(length(Gy_vals), "Gy")
    i_n = Index(n_levels, "n")
    i_n′ = Index(n_levels, "n′")
    i_l = Index(l_levels, "l")
    i_l′ = Index(l_levels, "l′")
    i_λ = Index(2, "λ")  # sublattice index (values: 1 → +1, 2 → -1)
    i_λ′ = Index(2, "λ′")
    i_s = Index(2, "s")  # spin index
    i_s′ = Index(2, "s′")
    i_K = Index(2, "K")  # valley index
    i_K′ = Index(2, "K′")

    # Create core tensor without spin/valley indices
    Λ_core = ITensor(ikx, iky, iGx, iGy, i_n, i_n′, i_l, i_l′, i_λ, i_λ′)
    Λ_neg_core = ITensor(ikx, iky, iGx, iGy, i_n, i_n′, i_l, i_l′, i_λ, i_λ′)

    # Precompute constants
    K = 2π / L
    l_B = sqrt(q/p) * L

    @showprogress "Computing lambda tensor..." for ikx_idx in 1:dim(ikx), iky_idx in 1:dim(iky),
        iGx_idx in 1:dim(iGx), iGy_idx in 1:dim(iGy),
        i_n_idx in 1:dim(i_n), i_n′_idx in 1:dim(i_n′),
        i_l_idx in 1:dim(i_l), i_l′_idx in 1:dim(i_l′),
        i_λ_idx in 1:dim(i_λ), i_λ′_idx in 1:dim(i_λ′)

        # Get actual values from indices
        k_x = kx_vals[ikx_idx]
        k_y = ky_vals[iky_idx]
        G_x = Gx_vals[iGx_idx]
        G_y = Gy_vals[iGy_idx]
        n1 = i_n_idx - 1  # Convert to 0-based index
        n2 = i_n′_idx - 1
        l1 = i_l_idx
        l2 = i_l′_idx
        λ1 = i_λ_idx == 1 ? 1 : -1
        λ2 = i_λ′_idx == 1 ? 1 : -1

        # Compute matrix elements for both G and -G
        element = fourierMatrixElement(k_x, k_y, n1, n2, l1, l2, λ1, λ2,
                                        1, 1, 1, 1, L, p, q, G_x, G_y)
        element_neg = fourierMatrixElement(k_x, k_y, n1, n2, l1, l2, λ1, λ2,
                                            1, 1, 1, 1, L, p, q, -G_x, -G_y)

        Λ_core[ikx(ikx_idx), iky(iky_idx), iGx(iGx_idx), iGy(iGy_idx),
                i_n(i_n_idx), i_n′(i_n′_idx), i_l(i_l_idx), i_l′(i_l′_idx),
                i_λ(i_λ_idx), i_λ′(i_λ′_idx)] = element
        Λ_neg_core[ikx(ikx_idx), iky(iky_idx), iGx(iGx_idx), iGy(iGy_idx),
                    i_n(i_n_idx), i_n′(i_n′_idx), i_l(i_l_idx), i_l′(i_l′_idx),
                    i_λ(i_λ_idx), i_λ′(i_λ′_idx)] = element_neg
    end

    # Extend to spin and valley indices using delta tensors
    δ_s = delta(i_s, i_s′)
    δ_K = delta(i_K, i_K′)

    Λ_full = Λ_core * δ_s * δ_K
    Λ_neg_full = Λ_neg_core * δ_s * δ_K

    return Λ_full, Λ_neg_full, (ikx, iky, iGx, iGy, i_n, i_n′, i_l, i_l′, i_λ, i_λ′, i_s, i_s′, i_K, i_K′)
end

function ionicCorrectionTensor(Λ, Λ_neg, V, q_indices, kGrid)
    # Extract indices
    iGx = commonind(Λ, V)
    iGy = commonind(Λ, V)
    ikx = commonind(Λ, Λ_neg)
    iky = commonind(Λ, Λ_neg)
    iqx, iqy = q_indices

    # Get grid size for normalization
    n_kpoints = dim(ikx) * dim(iky)

    # Find the middle index for q=0 (assuming odd-sized grids centered at 0)
    qx_mid_idx = div(dim(iqx), 2) + 1
    qy_mid_idx = div(dim(iqy), 2) + 1

    # Set q=0 in potential tensor
    V0 = V * setelt(iqx(qx_mid_idx)) * setelt(iqy(qy_mid_idx))

    # Prime G indices for pointwise multiplication
    V0_primed = prime(V0, (iGx, iGy))
    Λ_primed = prime(Λ, (iGx, iGy))
    Λ_neg_primed = prime(Λ_neg, (iGx, iGy))

    # Create delta tensor for G indices
    δ_G = delta(iGx, iGx'', iGx''') * delta(iGy, iGy'', iGy''')

    # Pointwise multiplication with V tensor
    VΛ = V0_primed * δ_G * Λ_primed
    VΛ_neg = V0_primed * δ_G * Λ_neg_primed

    # Unprime G indices
    VΛ = unprime(VΛ, (iGx, iGy))
    VΛ_neg = unprime(VΛ_neg, (iGx, iGy))

    # Create a tensor of ones for momentum summation
    ones_k = ITensor(ikx, iky)
    for i in 1:dim(ikx), j in 1:dim(iky)
        ones_k[ikx(i), iky(j)] = 1.0
    end

    # Sum over momentum indices by contracting with ones tensor
    VΛ_sum = VΛ * ones_k
    VΛ_neg_sum = VΛ_neg * ones_k

    # Partial trace over orbital indices only (excluding momentum)
    orbital_indices = [i_n, i_n′, i_l, i_l′, i_λ, i_λ′, i_s, i_s′, i_K, i_K′]
    traced_Λ = partial_trace(Λ, orbital_indices) / n_kpoints
    traced_Λ_neg = partial_trace(Λ_neg, orbital_indices) / n_kpoints

    # Contract with traced tensors
    I1 = VΛ_sum * traced_Λ_neg
    I2 = VΛ_neg_sum * traced_Λ

    # Return the sum
    return 0.5 * (I1 + I2)
end


function build_noninteracting_hamiltonian_tensor(harmonics,levels, p, q, L, orbital_indices, momentum_indices, k_grid_vals_x, k_grid_vals_y)
    i_n, i_s, i_K, i_l = orbital_indices
    ikx, iky = momentum_indices

    H0_total = ITensor(orbital_indices..., orbital_indices'..., ikx, iky)

    for kx_idx in eachindex(k_grid_vals_x), ky_idx in eachindex(k_grid_vals_y)
        kx_val = k_grid_vals_x[kx_idx]
        ky_val = k_grid_vals_y[ky_idx]

        H0_k = Hamiltonian(kx_val, ky_val, levels, harmonics, 0.0, p, q, L, [0,0,0], [0,0,0], 0.0)
        H0_mat = Matrix(H0_k)

        # Set the values for this k-point
        for orb1 in 1:dim(i_n)*dim(i_s)*dim(i_K)*dim(i_l),
            orb2 in 1:dim(i_n)*dim(i_s)*dim(i_K)*dim(i_l)

            # Convert flat indices to multi-indices
            idx1 = ind2sub((dim(i_n), dim(i_s), dim(i_K), dim(i_l)), orb1)
            idx2 = ind2sub((dim(i_n), dim(i_s), dim(i_K), dim(i_l)), orb2)

            H0_total[i_n(idx1[1]), i_s(idx1[2]), i_K(idx1[3]), i_l(idx1[4]),
                        i_n'(idx2[1]), i_s'(idx2[2]), i_K'(idx2[3]), i_l'(idx2[4]),
                        ikx(kx_idx), iky(ky_idx)] = H0_mat[orb1, orb2]
        end
    end

    return H0_total
end
