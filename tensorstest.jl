using ITensors
using LinearAlgebra
using ProgressMeter
using Random


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
    i_n, i_λ = indices
    iqx, iqy = q_indices'
        iGx, iGy = G_indices'
            S_core = ITensor(i_n, i_λ, i_n'', i_λ'',iqx,iqy,iGx,iGy)


        qx_vals, qy_vals = q_grid
        Gx_vals, Gy_vals = G_vectors
        K = 2 * π / L

        @showprogress "Pre-computing Form Factor Core S(n,λ,q,G)..." for qx_idx in 1:dim(iqx), qy_idx in 1:dim(iqy), Gx_idx in 1:dim(iGx), Gy_idx in 1:dim(iGy)
            q = [qx_vals[qx_idx], qy_vals[qy_idx]]
            G = [Gx_vals[Gx_idx], Gy_vals[Gy_idx]]
            q_total = q + K * G

            for n1 in 1:dim(i_n), n2 in 1:dim(i_n), λ1 in 1:dim(i_λ), λ2 in 1:dim(i_λ)
                val = grapheneLandauFourierMatrixElement(q_total[1], q_total[2], n1 - 1, n2 - 1, l_B, (λ1 == 1 ? 1 : -1), (λ2 == 1 ? 1 : -1))
                S_core[i_n(n1), i_λ(λ1), i_n''(n2), i_λ''(λ2), iqx(qx_idx), iqy(qy_idx), iGx(Gx_idx), iGy(Gy_idx)] = val
            end
        end
        return S_core
end

"""
Pre-computes the S(-q) form factor tensor using S(-q)_ab = (-1)^(na-nb) * S(q)_ba.
"""
function precomputeFormFactorSnegQ(indices, q_indices, G_indices, q_grid, G_vectors, L, l_B)
    i_n, i_λ = indices
    iqx, iqy = q_indices''
        iGx, iGy = G_indices
    S_neg_q = ITensor(i_n', i_λ', i_n''', i_λ''', iqx,iqy,iGx,iGy)


    qx_vals, qy_vals = q_grid
    Gx_vals, Gy_vals = G_vectors
    K = 2 * π / L

    @showprogress "Pre-computing Form Factor S(-q)..." for qx_idx in 1:dim(iqx), qy_idx in 1:dim(iqy), Gx_idx in 1:dim(iGx), Gy_idx in 1:dim(iGy)
        q = [qx_vals[qx_idx], qy_vals[qy_idx]]
        G = [Gx_vals[Gx_idx], Gy_vals[Gy_idx]]
        q_total = q + K * G

        for na in 1:dim(i_n), nb in 1:dim(i_n), λa in 1:dim(i_λ), λb in 1:dim(i_λ)
            # Calculate S(q)_ba
            val_S_ba = grapheneLandauFourierMatrixElement(-q_total[1], -q_total[2], nb - 1, na - 1, l_B, (λb == 1 ? 1 : -1), (λa == 1 ? 1 : -1))
            # S(-q)_ab = (-1)^(na-nb) * S(q)_ba
            final_val = val_S_ba
            S_neg_q[i_n'(na), i_λ'(λa), i_n'''(nb), i_λ'''(λb), iqx(qx_idx), iqy(qy_idx), iGx(Gx_idx), iGy(Gy_idx)] = final_val
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
        i_l = orbital_indices[5]
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
        function precomputeExchangePhaseTensor(orbital_indices, momentum_indices, q_indices, Q_val, G_indices,
                                               q_grid, G_vectors, L, l_B, p_supercell, q_supercell, Ky)
            # Unpack indices
            i_l = orbital_indices[5]
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
                i_n, i_λ, i_s, i_K, i_l = orbital_indices
                ikx, iky = momentum_indices
                iQx, iQy = Q_indices

                # Get G indices from one of the tensors that has them
                iGx = findindex(V_full, "Gx")
                iGy = findindex(V_full, "Gy")

                # Initialize output Hamiltonian tensor with zeros

                # Define composite indices and new momentum indices once, as they are G-independent
                Γ1 = (n=i_n,  λ=i_λ,  s=i_s,  K=i_K,  l=i_l)
                Γ2 = (n=i_n',   λ=i_λ',   s=i_s',   K=i_K',   l=i_l')
                Γ3 = (n=i_n'', λ=i_λ'', s=i_s'', K=i_K'', l=i_l'')
                Γ4 = (n=i_n''',λ=i_λ''',s=i_s''',K=i_K''',l=i_l''')
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
                D = permute(D, i_n, i_λ, i_s, i_K, i_l, i_n'', i_λ'', i_s'', i_K'', i_l'', ikx, iky)
                #print(inds(D))
                # The sum over the internal momentum k4 was handled by the tensor contraction.
                return D
            end

            function naiveDirectTerm(Δ, V_full, S_core_full, S_neg_q_core_full, Phase_D_full,
                                     orbital_indices, momentum_indices, q_grid, G_vectors, Q_val,
                                     q_indices, G_indices)
                i_n, i_λ, i_s, i_K, i_l = orbital_indices
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
                dims = (dim(i_n), dim(i_λ), dim(i_s), dim(i_K), dim(i_l),
                        dim(i_n), dim(i_λ), dim(i_s), dim(i_K), dim(i_l),
                        dim(ikx), dim(iky))
                H_direct_naive = zeros(ComplexF64, dims)

                for n1 in 1:dim(i_n), λ1 in 1:dim(i_λ), s1 in 1:dim(i_s), K1 in 1:dim(i_K), l1 in 1:dim(i_l),
                    n3 in 1:dim(i_n), λ3 in 1:dim(i_λ), s3 in 1:dim(i_s), K3 in 1:dim(i_K), l3 in 1:dim(i_l),
                    kx3_idx in 1:dim(ikx), ky3_idx in 1:dim(iky)

                    for kx4_idx in 1:dim(ikx), ky4_idx in 1:dim(iky),
                        n2 in 1:dim(i_n), λ2 in 1:dim(i_λ), s2 in 1:dim(i_s), K2 in 1:dim(i_K), l2 in 1:dim(i_l),
                        n4 in 1:dim(i_n), λ4 in 1:dim(i_λ), s4 in 1:dim(i_s), K4 in 1:dim(i_K), l4 in 1:dim(i_l),
                        Gx_idx in 1:dim(iGx), Gy_idx in 1:dim(iGy)

                        # Kronecker deltas
                        if !(s1 == s4 && s2 == s3 && K1 == K4 && K2 == K3)
                            continue
                        end

                        # Density matrix value
                        Δ_val = Δ[i_n'(n4), i_λ'(λ4), i_s'(s4), i_K'(K4), i_l'(l4),
                                  i_n(n2), i_λ(λ2), i_s(s2), i_K(K2), i_l(l2),
                                  ikx'(kx4_idx), iky'(ky4_idx)]

                        # Potential and form factors
                        V_val = V_full[iqx'(Qx_idx), iqy'(Qy_idx), iGx(Gx_idx), iGy(Gy_idx)]
                        S13_val = S_core_full[i_n(n1), i_λ(λ1), i_n''(n3), i_λ''(λ3),
                                              iqx'(Qx_idx), iqy'(Qy_idx), iGx'(Gx_idx), iGy'(Gy_idx)]
                        S42_val = S_neg_q_core_full[i_n'(n4), i_λ'(λ4), i_n'''(n2), i_λ'''(λ2),
                                                    iqx''(Qx_idx), iqy''(Qy_idx), iGx(Gx_idx), iGy(Gy_idx)]

                        # Phase factor
                        phase_val = Phase_D_full[i_l(l1), i_l'(l2), i_l''(l3), i_l'''(l4),
                                                 ikx(kx3_idx), iky(ky3_idx),
                                                 ikx'(kx4_idx), iky'(ky4_idx),
                                                 iqx(Qx_idx), iGx(Gx_idx), iGy(Gy_idx)]

                        term = Δ_val * V_val * S13_val * S42_val * phase_val
                        H_direct_naive[n1, λ1, s1, K1, l1, n3, λ3, s3, K3, l3, kx3_idx, ky3_idx] += term
                    end
                end
                return H_direct_naive
            end

            function buildExchangeTerm(Δ::ITensor, VTensor::ITensor, STensor::ITensor, SNegTensor::ITensor, Phase_X::ITensor,
                                       Shift::ITensor, ikx_p::Index, iky_p::Index, orbital_indices, momentum_indices)

                # Unpack indices
                i_n, i_λ, i_s, i_K, i_l = orbital_indices
                ikx, iky = momentum_indices
                iqx = findindex(Shift, "qx")
                iqy = findindex(Shift, "qy")
                iGx = findindex(VTensor, "Gx")
                iGy = findindex(VTensor, "Gy")

                # Initialize output Hamiltonian tensor with zeros
                H_exchange = ITensor(ikx, iky, orbital_indices'..., orbital_indices...)

                # --- Define composite indices for the 4 states in the interaction ---
                Γ1 = (n=i_n,  λ=i_λ,  s=i_s,  K=i_K,  l=i_l)
                Γ2 = (n=i_n',   λ=i_λ',   s=i_s',   K=i_K',   l=i_l')
                Γ3 = (n=i_n'', λ=i_λ'', s=i_s'', K=i_K'', l=i_l'')
                Γ4 = (n=i_n''',λ=i_λ''',s=i_s''',K=i_K''',l=i_l''')

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

                H_X = permute(H_X, i_n, i_λ, i_s, i_K, i_l, i_n''', i_λ''', i_s''', i_K''', i_l''', ikx, iky)

                #print(inds(H_X))
                # the final result is negated
                return -1.0 * H_X
            end

            function naiveExchangeTerm(Δ, V_full, S_core_full, S_neg_q_core_full, Phase_X_full,
                                       orbital_indices, momentum_indices, q_grid, G_vectors, Q_val,
                                       q_indices, G_indices, L, l_B, p_supercell, q_supercell, Ky)
                i_n, i_λ, i_s, i_K, i_l = orbital_indices
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
                dims = (dim(i_n), dim(i_λ), dim(i_s), dim(i_K), dim(i_l),
                        dim(i_n), dim(i_λ), dim(i_s), dim(i_K), dim(i_l),
                        dim(ikx), dim(iky))
                H_exchange_naive = zeros(ComplexF64, dims)

                for n1 in 1:dim(i_n), λ1 in 1:dim(i_λ), s1 in 1:dim(i_s), K1 in 1:dim(i_K), l1 in 1:dim(i_l),
                    n4 in 1:dim(i_n), λ4 in 1:dim(i_λ), s4 in 1:dim(i_s), K4 in 1:dim(i_K), l4 in 1:dim(i_l),
                    kx4_idx in 1:dim(ikx), ky4_idx in 1:dim(iky)

                    # Kronecker deltas for external states
                    if !(s1 == s4 && K1 == K4)
                        continue
                    end

                    for qx_idx in 1:dim(iqx), qy_idx in 1:dim(iqy),
                        Gx_idx in 1:dim(iGx), Gy_idx in 1:dim(iGy),
                        n2 in 1:dim(i_n), λ2 in 1:dim(i_λ), s2 in 1:dim(i_s), K2 in 1:dim(i_K), l2 in 1:dim(i_l),
                        n3 in 1:dim(i_n), λ3 in 1:dim(i_λ), s3 in 1:dim(i_s), K3 in 1:dim(i_K), l3 in 1:dim(i_l)

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
                        Δ_val = Δ[i_n'(n3), i_λ'(λ3), i_s'(s3), i_K'(K3), i_l'(l3),
                                  i_n(n2), i_λ(λ2), i_s(s2), i_K(K2), i_l(l2),
                                  ikx'(kx_prime_idx), iky'(ky_prime_idx)]

                        # Potential and form factors
                        V_val = V_full[iqx'(qx_idx), iqy'(qy_idx), iGx(Gx_idx), iGy(Gy_idx)]
                        S13_val = S_core_full[i_n(n1), i_λ(λ1), i_n''(n3), i_λ''(λ3),
                                              iqx'(qx_idx), iqy'(qy_idx), iGx'(Gx_idx), iGy'(Gy_idx)]
                        S42_val = S_neg_q_core_full[i_n'(n4), i_λ'(λ4), i_n'''(n2), i_λ'''(λ2),
                                                    iqx''(qx_idx), iqy''(qy_idx), iGx(Gx_idx), iGy(Gy_idx)]

                        # Phase factor
                        Gx_int = Gx_vals[Gx_idx]
                        Gy_int = Gy_vals[Gy_idx]
                        phase_val = Phase_X_full[i_l(l1), i_l'(l2), i_l''(l3), i_l'''(l4),
                                                 ikx'(kx4_idx), iky'(ky4_idx),
                                                 iqx'(qx_idx), iqy'(qy_idx), iGx'(Gx_idx), iGy'(Gy_idx)]

                        term = -Δ_val * V_val * S13_val * S42_val * phase_val
                        H_exchange_naive[n1, λ1, s1, K1, l1, n4, λ4, s4, K4, l4, kx4_idx, ky4_idx] += term
                    end
                end
                return H_exchange_naive
            end


            #------------------------------------------------------------------------------
            # VERIFICATION AND TESTING
            #------------------------------------------------------------------------------
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
                i_n = Index(N_n, "n")
                i_λ = Index(N_λ, "λ")
                i_s = Index(N_s, "s")
                i_K = Index(N_K, "K")
                i_l = Index(N_l, "l")
                orbital_indices = (i_n, i_λ, i_s, i_K, i_l)

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
                Q_val = (1,1) # Corresponds to the first element of the k-grid
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
                S_core_full = precomputeFormFactorTensorCore((i_n, i_λ), q_indices, G_indices, q_grid, G_vectors, L, l_B)
                S_neg_q_core_full = precomputeFormFactorSnegQ((i_n, i_λ), q_indices, G_indices, q_grid, G_vectors, L, l_B)

                Phase_D_full = precomputeDirectPhaseTensor(orbital_indices, momentum_indices, q_indices, G_indices, q_grid, G_vectors, L, l_B, p_supercell, q_supercell, Ky)
                Phase_X_full = precomputeExchangePhaseTensor(orbital_indices, momentum_indices, q_indices, Q_val, G_indices, q_grid, G_vectors, L, l_B, p_supercell, q_supercell, Ky)
                Shift, ikx_p, iky_p = precomputeConvolutionTensor(momentum_indices, q_indices, Q_val,kxRadius,kyRadius)

                # --- 5. Create a dummy density matrix Δ ---
                # Δ is diagonal in all orbital indices and momentum
                println("\n--- Building Dummy Density Matrix Δ ---")
                Δ = ITensor(orbital_indices'..., orbital_indices..., ikx', iky')
                for idx_k_x in 1:N_kx, idx_k_y in 1:N_ky
                    for idx_n in 1:N_n, idx_λ in 1:N_λ, idx_s in 1:N_s, idx_K in 1:N_K, idx_l in 1:N_l
                        # Let's say only the lowest LL is occupied
                        if idx_n == 1
                            Δ[i_n'(idx_n), i_λ'(idx_λ), i_s'(idx_s), i_K'(idx_K), i_l'(idx_l),
                              i_n(idx_n), i_λ(idx_λ), i_s(idx_s), i_K(idx_K), i_l(idx_l),
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
                H_direct_arr = permutedims(array(H_direct), [1,2,3,4,5, 6,7,8,9,10, 11,12])
                H_exchange_arr = permutedims(array(H_exchange), [1,2,3,4,5, 6,7,8,9,10, 11,12])

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
