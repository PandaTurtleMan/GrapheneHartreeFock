# To run this script:
# 1. Save the ENTIRE content of this file as a single Julia file (e.g., "run_verification.jl").
# 2. Make sure you have ITensors, Test, and ProgressMeter installed in your Julia environment.
# 3. To enable multithreading for better performance, run from your terminal:
#    julia -t auto run_verification.jl

using ITensors
using Test
using LinearAlgebra
using Random
using ProgressMeter

###############################################################################
# OPTIMIZED TENSOR-BASED IMPLEMENTATION
###############################################################################

# Note: Placeholder functions for your actual implementations.
function grapheneLandauFourierMatrixElement(qx, qy, n1, n2, lB, lda1, lda2)
    return 1.0
end

function screening_fn(q_norm)
    return 1.0
end

# --- UTILITY FUNCTIONS ---
function get_gamma_indices(i_n, i_λ, i_s, i_k, i_l)
    # Splat the i_k tuple to create flat tuples of indices.
    # This avoids nested tuples which cause errors in `replaceinds`.
    Γ1 = (prime(i_n), prime(i_λ), prime(i_s), prime(i_k)..., prime(i_l))
    Γ3 = (i_n, i_λ, i_s, i_k..., i_l)
    Γ2 = (prime(i_n,2), prime(i_λ,2), prime(i_s,2), prime(i_k, 2)..., prime(i_l,2))
    Γ4 = (prime(i_n,3), prime(i_λ,3), prime(i_s,3), prime(i_k, 3)..., prime(i_l,3))
    return Γ1, Γ2, Γ3, Γ4
end

# --- TENSOR PRECOMPUTATION (OPTIMIZED) ---
function precomputePotentialTensor(q_indices, G_indices, q_grid, G_vectors, ε, L)
    V = ITensor(q_indices..., G_indices...)
    K = 2 * π / L
    iqx, iqy = q_indices
    iGx, iGy = G_indices
    qx_vals, qy_vals = q_grid
    Gx_vals, Gy_vals = G_vectors

    # Use a manual Progress object for thread-safe progress tracking.
    p = Progress(dim(iqx), "Pre-computing Potential V(q+G)...")
    Base.Threads.@threads for qx_idx in 1:dim(iqx)
        for qy_idx in 1:dim(iqy), Gx_idx in 1:dim(iGx), Gy_idx in 1:dim(iGy)
            q = [qx_vals[qx_idx], qy_vals[qy_idx]]
            G = [Gx_vals[Gx_idx], Gy_vals[Gy_idx]]
            q_total = q + K * G
            q_total_norm = norm(q_total)
            val = q_total_norm < 1e-9 ? 0.0 : (2 * π / q_total_norm) * screening_fn(q_total_norm) / ε
            V[iqx(qx_idx), iqy(qy_idx), iGx(Gx_idx), iGy(Gy_idx)] = val
        end
        next!(p)
    end
    return V
end

function precomputeFormFactorTensorCore(i_n::Index, i_λ::Index, q_indices, G_indices, q_grid, G_vectors, L, l_B)
    S_core = ITensor(i_n', i_λ', i_n, i_λ, q_indices..., G_indices...)
    K = 2 * π / L
    iqx, iqy = q_indices
    iGx, iGy = G_indices
    qx_vals, qy_vals = q_grid
    Gx_vals, Gy_vals = G_vectors

    # Use a manual Progress object for thread-safe progress tracking.
    p = Progress(dim(iqx), "Pre-computing Form Factor Core S(n,λ,q,G)...")
    Base.Threads.@threads for qx_idx in 1:dim(iqx)
        for qy_idx in 1:dim(iqy), Gx_idx in 1:dim(iGx), Gy_idx in 1:dim(iGy)
            q = [qx_vals[qx_idx], qy_vals[qy_idx]]
            G = [Gx_vals[Gx_idx], Gy_vals[Gy_idx]]
            q_total = q + K * G
            for n1 in 1:dim(i_n), n2 in 1:dim(i_n), λ1 in 1:dim(i_λ), λ2 in 1:dim(i_λ)
                val = grapheneLandauFourierMatrixElement(q_total[1], q_total[2], n1-1, n2-1, l_B, (λ1==1 ? 1 : -1), (λ2==1 ? 1 : -1))
                S_core[i_n'(n1), i_λ'(λ1), i_n(n2), i_λ(λ2), iqx(qx_idx), iqy(qy_idx), iGx(Gx_idx), iGy(Gy_idx)] = val
            end
        end
        next!(p)
    end
    return S_core
end

function precomputeParityFactor(i_n::Index)
    Parity = ITensor(i_n', i_n)
    @showprogress "Pre-computing Parity Factor..." for n1 in 1:dim(i_n'), n2 in 1:dim(i_n)
        Parity[i_n'(n1), i_n(n2)] = (-1.0)^(n1 - n2)
    end
    return Parity
end

function precomputePhaseTensor(l_indices, k_q_indices, G_indices, k_grid, q_grid, G_vectors, p_supercell, q_supercell, l_B, L, phase_type::Symbol)
    K = 2 * π / L
    Kx = Ky = K
    kx_vals, ky_vals = k_grid
    qx_vals, qy_vals = q_grid
    Gx_vals, Gy_vals = G_vectors

    if phase_type == :P1
        i_l, i_l_p = l_indices
        iqx, iky_p, iky = k_q_indices
        iGx, _ = G_indices
        Phase = ITensor(0.0, i_l_p, i_l, iqx, iky_p, iky, iGx)
        @showprogress "Pre-computing Phase Factor 1..." for l_p_idx in 1:dim(i_l_p), l_idx in 1:dim(i_l), qx_idx in 1:dim(iqx), ky_p_idx in 1:dim(iky_p), ky_idx in 1:dim(iky), Gx_idx in 1:dim(iGx)
            exponent = im * (qx_vals[qx_idx] + Gx_vals[Gx_idx] * K) * l_B^2 * (ky_vals[ky_p_idx] - ky_vals[ky_idx] + (l_p_idx - l_idx) / q_supercell * Ky)
            Phase[i_l_p(l_p_idx), i_l(l_idx), iqx(qx_idx), iky_p(ky_p_idx), iky(ky_idx), iGx(Gx_idx)] = exp(exponent)
        end
        elseif phase_type == :P2
        i_l1, i_l3 = l_indices
        iqx, ikx_p = k_q_indices
        _, iGy = G_indices
        Phase = ITensor(0.0, i_l3, i_l1, ikx_p, iqx, iGy)
        @showprogress "Pre-computing Phase Factor 2..." for l1_idx in 1:dim(i_l1), l3_idx in 1:dim(i_l3), kx_p_idx in 1:dim(ikx_p), qx_idx in 1:dim(iqx), Gy_idx in 1:dim(iGy)
            if mod(l3_idx - l1_idx + q_supercell * Gy_vals[Gy_idx] / Ky, p_supercell) == 0
                exponent = im * (2π / (p_supercell * Kx)) * (kx_vals[kx_p_idx] - qx_vals[qx_idx]) * (l3_idx - l1_idx + q_supercell * Gy_vals[Gy_idx] / Ky)
                Phase[i_l3(l3_idx), i_l1(l1_idx), ikx_p(kx_p_idx), iqx(qx_idx), iGy(Gy_idx)] = exp(exponent)
            end
        end
        elseif phase_type == :P3
        i_l2, i_l4 = l_indices
        iqx, ikx_p = k_q_indices
        _, iGy = G_indices
        Phase = ITensor(0.0, i_l4, i_l2, ikx_p, iqx, iGy)
        @showprogress "Pre-computing Phase Factor 3..." for l2_idx in 1:dim(i_l2), l4_idx in 1:dim(i_l4), kx_p_idx in 1:dim(ikx_p), qx_idx in 1:dim(iqx), Gy_idx in 1:dim(iGy)
            if mod(l4_idx - l2_idx - q_supercell * Gy_vals[Gy_idx] / Ky, p_supercell) == 0
                exponent = im * (2π / (p_supercell * Kx)) * (kx_vals[kx_p_idx] + qx_vals[qx_idx]) * (l4_idx - l2_idx - q_supercell * Gy_vals[Gy_idx] / Ky)
                Phase[i_l4(l4_idx), i_l2(l2_idx), ikx_p(kx_p_idx), iqx(qx_idx), iGy(Gy_idx)] = exp(exponent)
            end
        end
    end
    return Phase
end

# --- HAMILTONIAN CONSTRUCTION (OPTIMIZED) ---
function buildDirectTerm(Δ::ITensor, V_full::ITensor, S_core_full::ITensor, Parity::ITensor, P1_full::ITensor, P2_full::ITensor, P3_full::ITensor,
                         Q_indices, i_n, i_λ, i_s, i_k, i_l::Index)
    ITensors.disable_warn_order()
    try
        iQx, iQy = Q_indices
        (ikx, iky) = i_k

        # --- Slice ALL tensors at q=Q ---
        slice_qx = onehot(iQx => 1)
        slice_qy = onehot(iQy => 1)

        # Slice V and S_core
        V_D = V_full * slice_qx * slice_qy
        S_core_D = S_core_full * slice_qx * slice_qy

        # Slice phase tensors at q=Q
        P1_D = P1_full * slice_qx * slice_qy * slice_qy  # Fix all q-indices
        P2_D = P2_full * slice_qx * slice_qx
        P3_D = P3_full * slice_qx * slice_qx

        Γ1, Γ2, Γ3, Γ4 = get_gamma_indices(i_n, i_λ, i_s, i_k, i_l)
        inds_out_orig = (prime(i_n), prime(i_λ), prime(i_s), prime(i_k)..., prime(i_l))
        inds_in_orig = (i_n, i_λ, i_s, i_k..., i_l)

        # Replace indices to match Γ conventions
        S_D_13 = replaceinds(S_core_D, (prime(i_n), prime(i_λ), i_n, i_λ), (Γ1[1], Γ1[2], Γ3[1], Γ3[2]))
        S_neg_Q_24 = replaceinds(S_core_D, (prime(i_n), prime(i_λ), i_n, i_λ), (Γ2[1], Γ2[2], Γ4[1], Γ4[2])) *
            replaceinds(Parity, (prime(i_n), i_n), (Γ2[1], Γ4[1]))

        # Adjust phase tensors to match Γ indices
        P1_34 = replaceinds(P1_D, (prime(i_l), i_l, prime(iky), iky), (Γ4[6], Γ3[6], Γ4[5], Γ3[5]))
        P2_13 = replaceinds(P2_D, (prime(i_l), i_l, prime(ikx)), (Γ3[6], Γ1[6], Γ3[4]))
        P3_24 = replaceinds(P3_D, (prime(i_l), i_l, prime(ikx)), (Γ4[6], Γ2[6], Γ4[4]))

        # Prepare Δ with Γ2 and Γ4 indices
        Δ_42 = replaceinds(Δ, (inds_out_orig..., inds_in_orig...), (Γ4..., Γ2...))

        # Build Vertex tensor and contract with Δ_42
        Vertex = V_D * S_D_13 * S_neg_Q_24 * P1_34 * P2_13 * P3_24 *
            delta(Γ1[3], Γ4[3]) * delta(Γ2[3], Γ3[3])
        H_direct = Vertex * Δ_42

        # Remove remaining G indices (they should be fully contracted)
        for ind in inds(H_direct)
            if hasind(V_full, ind)  # Remove any lingering G/q indices
                H_direct *= delta(ind)
            end
        end

        # Correct prime levels for final output
        for ind in inds(H_direct)
            p = plev(ind)
            if p == 2  # Γ2 indices should become output (prime=1)
                H_direct = replaceind(H_direct, ind, setprime(ind, 1))
                elseif p == 3  # Γ4 indices should become input (prime=0)
                H_direct = replaceind(H_direct, ind, setprime(ind, 0))
            end
        end

        # Permute to desired order and return
        return permute(H_direct, inds_out_orig..., inds_in_orig...)
    finally
        ITensors.reset_warn_order()
    end
end

###############################################################################
# NAIVE LOOP-BASED IMPLEMENTATION
###############################################################################

function buildDirectTerm_naive(Δ::ITensor, V_full::ITensor, S_core_full::ITensor, Parity::ITensor, P1_full::ITensor, P2_full::ITensor, P3_full::ITensor,
                               Q_indices, i_n, i_λ, i_s, i_k, i_l::Index, q_indices)

    # --- Setup ---
    H_direct_naive = copy(Δ)
    H_direct_naive .= 0.0 + 0.0im

    iQx, iQy = Q_indices
    (ikx, iky) = i_k
    (iqx, iqy) = q_indices

    ikx_p2 = first(inds(P2_full, tags="kx"))
    ikx_p3 = first(inds(P3_full, tags="kx"))

    iGx = only(inds(V_full, tags="Gx"))
    iGy = only(inds(V_full, tags="Gy"))

    dim_n, dim_λ, dim_s = dim(i_n), dim(i_λ), dim(i_s)
    dim_kx, dim_ky = dim(ikx), dim(iky)
    dim_l = dim(i_l)
    dim_Gx, dim_Gy = dim(iGx), dim(iGy)

    # --- Slice precomputed tensors at q=Q ---
    slice_qx = onehot(iQx => 1)
    slice_qy = onehot(iQy => 1)
    V_D = V_full * slice_qx * slice_qy
    S_core_D = S_core_full * slice_qx * slice_qy

    @showprogress "Running naive direct term calculation..." for n1 in 1:dim_n, λ1 in 1:dim_λ, s1 in 1:dim_s, kx1 in 1:dim_kx, ky1 in 1:dim_ky, l1 in 1:dim_l
        for n3 in 1:dim_n, λ3 in 1:dim_λ, s3 in 1:dim_s, kx3 in 1:dim_kx, ky3 in 1:dim_ky, l3 in 1:dim_l
            H_element = 0.0 + 0.0im
            for n2 in 1:dim_n, λ2 in 1:dim_λ, s2 in 1:dim_s, kx2 in 1:dim_kx, ky2 in 1:dim_ky, l2 in 1:dim_l
                for n4 in 1:dim_n, λ4 in 1:dim_λ, s4 in 1:dim_s, kx4 in 1:dim_kx, ky4 in 1:dim_ky, l4 in 1:dim_l
                    for Gx_idx in 1:dim_Gx, Gy_idx in 1:dim_Gy
                        if !(s1 == s4 && s2 == s3) continue end
                        V_val = V_D[iGx(Gx_idx), iGy(Gy_idx)]
                        S_13 = S_core_D[i_n'(n1), i_λ'(λ1), i_n(n3), i_λ(λ3), iGx(Gx_idx), iGy(Gy_idx)]
                        S_24_core = S_core_D[i_n'(n2), i_λ'(λ2), i_n(n4), i_λ(n4), iGx(Gx_idx), iGy(Gy_idx)]
                        Parity_val = Parity[i_n'(n2), i_n(n4)]
                        S_neg_24 = S_24_core * Parity_val

                        P1_val = scalar(P1_full * onehot(prime(i_l)=>l4, i_l=>l3, iqx=>1, prime(iky)=>ky4, iky=>ky3, iGx=>Gx_idx))
                        P2_val = scalar(P2_full * onehot(prime(i_l)=>l3, i_l=>l1, iqx=>1, ikx_p2=>kx3, iGy=>Gy_idx))
                        P3_val = scalar(P3_full * onehot(prime(i_l)=>l4, i_l=>l2, iqx=>1, ikx_p3=>kx4, iGy=>Gy_idx))

                        Δ_val = Δ[i_n'(n4), i_λ'(λ4), i_s'(s4), ikx'(kx4), iky'(ky4), prime(i_l)(l4),
                                  i_n(n2),  i_λ(n2),  i_s(s2),  ikx(kx2),  iky(ky2),  i_l(l2)]
                        term = V_val * S_13 * S_neg_24 * P1_val * P2_val * P3_val * Δ_val
                        H_element += term
                    end
                end
            end
            H_direct_naive[i_n'(n1), i_λ'(λ1), i_s'(s1), ikx'(kx1), iky'(ky1), prime(i_l)(l1),
                           i_n(n3),  i_λ(n3),  i_s(s3),  ikx(kx3),  iky(ky3),  i_l(l3)] = H_element
        end
    end
    return H_direct_naive
end

###############################################################################
# VERIFICATION TEST SUITE
###############################################################################

function random_hermitian_itensor(inds_out, inds_in)
    Δ = randomITensor(ComplexF64, inds_out..., inds_in...)
    Δ_dag = swapprime(dag(Δ), 0, 1)
    Δ = 0.5 * (Δ + Δ_dag)
    return Δ
end

function run_verification_test()
    println("Setting up simulation parameters...")
    L, l_B, ε, p_supercell, q_supercell = 10.0, 1.0, 1.0, 2, 2
    dim_n, dim_λ, dim_s, dim_l = 1, 1, 2, 1
    k_grid_dim_x, k_grid_dim_y = 2, 1
    q_grid_dim_x, q_grid_dim_y = k_grid_dim_x, k_grid_dim_y
    G_grid_dim = 2

    K = 2π / L
    K_MBZ_y = 2π / (q_supercell * L)
    kx_vals = [i * K / k_grid_dim_x for i in 0:k_grid_dim_x-1]
        ky_vals = [i * K_MBZ_y / k_grid_dim_y for i in 0:k_grid_dim_y-1]
            k_grid = (kx_vals, ky_vals)
            q_grid = k_grid
            G_vals = [i for i in -floor(Int, G_grid_dim/2):floor(Int, G_grid_dim/2)-1]
                G_vectors = (G_vals, G_vals)

                i_n, i_λ, i_s = Index(dim_n, "n"), Index(dim_λ, "λ"), Index(dim_s, "s,Spin")
                ikx, iky = Index(k_grid_dim_x, "kx,Momentum"), Index(k_grid_dim_y, "ky,Momentum")
                i_k, i_l = (ikx, iky), Index(dim_l, "l")
                iqx, iqy = Index(q_grid_dim_x, "qx"), Index(q_grid_dim_y, "qy")
                q_indices = (iqx, iqy)
                iGx, iGy = Index(G_grid_dim, "Gx,G"), Index(G_grid_dim, "Gy,G")
                G_indices = (iGx, iGy)
                Q_indices = (iqx, iqy)

                println("\nPre-computing all required tensors...")
                V_full = precomputePotentialTensor(q_indices, G_indices, q_grid, G_vectors, ε, L)
                S_core_full = precomputeFormFactorTensorCore(i_n, i_λ, q_indices, G_indices, q_grid, G_vectors, L, l_B)
                Parity = precomputeParityFactor(i_n)

                P1_full = precomputePhaseTensor((i_l, prime(i_l)), (iqx, prime(iky), iky), (iGx, iGy), k_grid, q_grid, G_vectors, p_supercell, q_supercell, l_B, L, :P1)
                P2_full = precomputePhaseTensor((i_l, prime(i_l)), (iqx, ikx), (iGx, iGy), k_grid, q_grid, G_vectors, p_supercell, q_supercell, l_B, L, :P2)
                P3_full = precomputePhaseTensor((i_l, prime(i_l)), (iqx, ikx), (iGx, iGy), k_grid, q_grid, G_vectors, p_supercell, q_supercell, l_B, L, :P3)

                println("\nGenerating random density matrix for test...")
                    Γ_out_inds = (prime(i_n), prime(i_λ), prime(i_s), prime(i_k)..., prime(i_l))
                    Γ_in_inds = (i_n, i_λ, i_s, i_k..., i_l)
                    Δ = random_hermitian_itensor(Γ_out_inds, Γ_in_inds)

                    println("\n--- Starting Verification ---")
                    @testset "Hartree-Fock Implementation Verification" begin
                        println("Testing Direct Term...")
                        t_tensor = @elapsed H_direct_tensor = buildDirectTerm(Δ, V_full, S_core_full, Parity, P1_full, P2_full, P3_full, Q_indices, i_n, i_λ, i_s, i_k, i_l)
                        println("  Optimized tensor method took: $(round(t_tensor, digits=4))s")

                        t_naive = @elapsed H_direct_naive_result = buildDirectTerm_naive(Δ, V_full, S_core_full, Parity, P1_full, P2_full, P3_full, Q_indices, i_n, i_λ, i_s, i_k, i_l, q_indices)
                        println("  Naive loop method took: $(round(t_naive, digits=4))s")

                        @test H_direct_tensor ≈ H_direct_naive_result
                    end
                    println("\n--- Verification Complete ---")
                end

                # To run the test, simply call this function:
                run_verification_test()
