using LinearAlgebra, ProgressMeter, JLD2
include("Utilities.jl")
include("solveNonInteractingProblem.jl")
include("HFUtilities.jl")
include("Tensors.jl")

function construct_hf_hamiltonian(H0, V, Δ)
    matrixSize = size(Δ, 1)
    # Convert Hermitian matrix to regular dense matrix
    H_dense = Matrix(H0)

    for i in 1:matrixSize
        for j in 1:matrixSize
            hartree_term = zero(ComplexF64)
            exchange_term = zero(ComplexF64)
            for k in 1:matrixSize
                for l in 1:matrixSize
                    hartree_term += V[i, k, l, j] * Δ[l, k]
                    exchange_term += V[i, k, j, l] * Δ[l, k]
                end
            end
            correction = hartree_term - exchange_term

            # For diagonal elements, ensure they remain real
            if i == j
                H_dense[i, j] += real(correction)
            else
                H_dense[i, j] += correction
            end
        end
    end

    # Convert back to Hermitian before returning
    return Hermitian(H_dense)
end

function createNewDelta(nF, U)
    N = size(U, 1)
    nF > N && throw("Too many particles!")
    NF = Diagonal([i <= nF ? 1.0 : 0.0 for i in 1:N])
    return U * NF * U'
end

function constructInitialGuess(H0, nF)
    F = eigen(Hermitian(H0))
    N = size(H0,1)
    U = F.vectors
    return U[:, 1:nF] * U[:, 1:nF]'
end

function hartree_fock_iteration(
    H0::AbstractMatrix, V,nF;
    max_iter::Int=1000,
    tol::Real=1e-6
)
    # Initial setup
    Δ = constructInitialGuess(H0,nF)
    prev_energy = Inf
    # Create progress bar
    previousIters = Matrix{ComplexF64}[]
    errorVecs = Matrix{ComplexF64}[]
    progress = Progress(
        max_iter,
        desc="Hartree-Fock Iterations: ",
        barglyphs=BarGlyphs('|','█', '▁', '|', ' '),
        output=stderr,
        showspeed=true
    )
    for iter in 1:max_iter
        # Construct and diagonalize HF Hamiltonian
        H_hf = construct_hf_hamiltonian(H0, V, Δ)
        F = eigen(Hermitian(H_hf))
        U = F.vectors
        basisSize = size(Δ,1)

        Δ_new_diag = U[:, 1:nF] * U[:, 1:nF]'
        error = Δ_new_diag - Δ

        # Store current state
        push!(errorVecs, error)
        push!(previousIters, Δ)

        # Apply DIIS only if sufficient history exists
        if length(errorVecs) >= 2
            coeffs = DIISErrorCoeffs(errorVecs)
            m = length(coeffs)
            Δ_new = extrapolateFockMatrixWithDIIS(coeffs, previousIters[end-m+1:end], nF)
        else
            Δ_new = Δ_new_diag  # Fallback to diagonal result
        end

        # Convergence check (using δΔ from diagonal output)
        δΔ = norm(Δ_new - Δ)

        # Update progress bar
        energy = real(tr(Δ * H_hf))
        next!(progress; showvalues=[
            (:Iteration, iter),
            (:Energy, energy),
            (:δΔ, δΔ)
        ])

        # Check convergence
        #δΔ_rel = δΔ / norm(Δ)
        if abs(energy - prev_energy) < tol && δΔ < tol
            break
        end

        # Update for next iteration
        prev_energy = energy
        Δ = Δ_new
    end

    return Δ
end

function compute_final_energy(
    H0::AbstractMatrix,
    Δ::AbstractMatrix,
    k_x::Real,
    k_y::Real,
    levels::Int,
    L::Real,
    l_B::Real,
    screening::Function,
    cutoff::Real,
    V
)
    H_hf = construct_hf_hamiltonian(H0, V,Δ)
    return 0.5 * real(tr(Δ * (H0 + H_hf)))
end

function run_hartree_fock_tensor(
    filename::String,
    initial_density_matrices::Vector{Tuple{String, ITensor}},
    levels::Int,
    p::Int,
    q::Int,
    L::Float64,
    nF::Int,
    ε::Float64,
    harmonicRange::Int,
    kxRadius::Int,
    kyRadius::Int,
    screening_fn::Function,
    Q_val::Tuple{Int, Int},
    harmonics
)
    # Grid parameters
    N_kx, N_ky = 2*kxRadius, 2*kxRadius
    N_G = 2*harmonicRange + 1

    # Create indices
    i_n = Index(2*levels+1, "n")
    i_s = Index(2, "s")
    i_K = Index(2, "K")
    i_l = Index(p, "l")  # p supercell bands
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

    # Create grids
    k_step_x = 2π/(L * N_kx)
    k_step_y = 2π/(q * L * N_ky)
    k_grid_vals_x = [n * k_step_x for n in -kxRadius:(kxRadius-1)]
    k_grid_vals_y = [n * k_step_y for n in -kyRadius:(kyRadius-1)]
    q_grid = (k_grid_vals_x, k_grid_vals_y)

    g_max = harmonicRange
    G_grid_vals = range(-g_max, g_max, length=N_G)
    G_vectors = (G_grid_vals, G_grid_vals)
    l_B = sqrt(q/p) * L

    # Precompute tensors (these will be kept in memory)
    println("Precomputing potential tensor...")
    V_full = precomputePotentialTensor(q_indices, G_indices, q_grid, G_vectors, screening_fn, ε, L)

    println("Precomputing form factor tensors...")
    S_core_full = precomputeFormFactorTensorCore(i_n, q_indices, G_indices, q_grid, G_vectors, L, l_B)
    S_neg_q_core_full = precomputeFormFactorSnegQ(i_n, q_indices, G_indices, q_grid, G_vectors, L, l_B)

    println("Precomputing phase tensors...")
    Phase_D_full = precomputeDirectPhaseTensor(
        orbital_indices,
        momentum_indices,
        q_indices,
        G_indices,
        q_grid,
        G_vectors,
        L,
        l_B,
        p,
        q,
        2π/L
    )

    Phase_X_full = precomputeExchangePhaseTensor(
        orbital_indices,
        momentum_indices,
        q_indices,
        Q_val,
        G_indices,
        q_grid,
        G_vectors,
        L,
        l_B,
        p,
        q,
        2π/L
    )

    Shift, ikx_p, iky_p = precomputeConvolutionTensor(momentum_indices, q_indices, Q_val, kxRadius, kyRadius)

    # Build non-interacting Hamiltonian at all k-points using helper function
    println("Building non-interacting Hamiltonian at all k-points...")
    H0_total = build_noninteracting_hamiltonian_tensor(harmonics,levels, p, q, L, orbital_indices, momentum_indices, k_grid_vals_x, k_grid_vals_y)

    # Ionic correction
    println("Computing ionic correction...")
    Λ_full, Λ_neg_full, λ_indices = lambdaTensor(p, q, L, q_grid, G_vectors, 2*levels+1, p)
    ionic_correction = ionicCorrectionTensor(Λ_full, Λ_neg_full, V_full, q_indices, q_grid)

    # Subtract ionic correction from H0 (fast tensor operation)
    H0_total -= ionic_correction

    # HF iteration parameters
    max_iter = 100
    tol = 1e-6
    n_kpoints = N_kx * N_ky
    n_gvectors = N_G * N_G

    # Store results for each initial density matrix
    results = Dict{String, Tuple{Float64, ITensor, Dict}}()

    # Process each initial density matrix
    for (name, Δ_total) in initial_density_matrices
        println("Processing initial density matrix: $name")

        energy_prev = Inf
        diis_error_history = ITensor[]
        diis_delta_history = ITensor[]
        max_diis_history = 6  # Maximum number of iterations to keep in DIIS history

        # Main HF loop
        for iter in 1:max_iter
            println("Iteration $iter")

            # Compute direct term for entire grid
            H_direct_total = buildDirectTerm(Δ_total, V_full, S_core_full, S_neg_q_core_full,
                                                Phase_D_full, Q_val, orbital_indices, momentum_indices)
            H_direct_total ./= (n_kpoints * n_gvectors)

            # Compute exchange term for entire grid
            H_exchange_total = buildExchangeTerm(Δ_total, V_full, S_core_full, S_neg_q_core_full,
                                                    Phase_X_full, Shift, ikx_p, iky_p, orbital_indices, momentum_indices)
            H_exchange_total ./= (n_kpoints * n_gvectors)

            # Total HF Hamiltonian
            H_hf_total = H0_total + H_direct_total - H_exchange_total

            # Store the current density matrix for DIIS
            Δ_old = copy(Δ_total)

            # Diagonalize at each k-point and update density matrix
            for kx_idx in 1:N_kx, ky_idx in 1:N_ky
                # Extract HF Hamiltonian for this k-point
                H_hf_slice = H_hf_total * setelt(ikx(kx_idx)) * setelt(iky(ky_idx))
                H_hf_mat = matrix(H_hf_slice, orbital_indices..., orbital_indices'...)

                # Diagonalize
                F = eigen(Hermitian(H_hf_mat))
                U = F.vectors

                # Update density matrix for this k-point
                Δ_new = U[:, 1:nF] * U[:, 1:nF]'

                # Store in total density matrix
                for orb1 in 1:dim(i_n)*dim(i_s)*dim(i_K)*dim(i_l),
                    orb2 in 1:dim(i_n)*dim(i_s)*dim(i_K)*dim(i_l)

                    idx1 = ind2sub((dim(i_n), dim(i_s), dim(i_K), dim(i_l)), orb1)
                    idx2 = ind2sub((dim(i_n), dim(i_s), dim(i_K), dim(i_l)), orb2)

                    Δ_total[i_n(idx1[1]), i_s(idx1[2]), i_K(idx1[3]), i_l(idx1[4]),
                            i_n'(idx2[1]), i_s'(idx2[2]), i_K'(idx2[3]), i_l'(idx2[4]),
                            ikx(kx_idx), iky(ky_idx)] = Δ_new[orb1, orb2]
                end

                # Clean up
                H_hf_slice = nothing
                H_hf_mat = nothing
                F = nothing
                U = nothing
                Δ_new = nothing
                GC.gc()
            end

            # Diagonalize at each k-point and update density matrix
            for kx_idx in 1:N_kx, ky_idx in 1:N_ky
                # Extract HF Hamiltonian for this k-point
                H_hf_slice = H_hf_total * setelt(ikx(kx_idx)) * setelt(iky(ky_idx))
                H_hf_mat = matrix(H_hf_slice, orbital_indices..., orbital_indices'...)

                # Diagonalize
                F = eigen(Hermitian(H_hf_mat))
                U = F.vectors

                # Update density matrix for this k-point
                Δ_new = U[:, 1:nF] * U[:, 1:nF]'

                # Store in total density matrix
                for orb1 in 1:dim(i_n)*dim(i_s)*dim(i_K)*dim(i_l),
                    orb2 in 1:dim(i_n)*dim(i_s)*dim(i_K)*dim(i_l)

                    idx1 = ind2sub((dim(i_n), dim(i_s), dim(i_K), dim(i_l)), orb1)
                    idx2 = ind2sub((dim(i_n), dim(i_s), dim(i_K), dim(i_l)), orb2)

                    Δ_total[i_n(idx1[1]), i_s(idx1[2]), i_K(idx1[3]), i_l(idx1[4]),
                            i_n'(idx2[1]), i_s'(idx2[2]), i_K'(idx2[3]), i_l'(idx2[4]),
                            ikx(kx_idx), iky(ky_idx)] = Δ_new[orb1, orb2]
                end

                # Clean up intermediate tensors for this k-point
                H_hf_slice = nothing
                H_hf_mat = nothing
                F = nothing
                U = nothing
                Δ_new = nothing
                GC.gc()
            end

            # Compute total energy
            total_energy = 0.0
            for kx_idx in 1:N_kx, ky_idx in 1:N_ky
                H0_slice = H0_total * setelt(ikx(kx_idx)) * setelt(iky(ky_idx))
                H_hf_slice = H_hf_total * setelt(ikx(kx_idx)) * setelt(iky(ky_idx))
                Δ_slice = Δ_total * setelt(ikx(kx_idx)) * setelt(iky(ky_idx))

                H0_mat = matrix(H0_slice, orbital_indices..., orbital_indices'...)
                H_hf_mat = matrix(H_hf_slice, orbital_indices..., orbital_indices'...)
                Δ_mat = matrix(Δ_slice, orbital_indices..., orbital_indices'...)

                total_energy += real(tr(Δ_mat * (H_hf_mat + H0_mat)/2))

                # Clean up intermediate tensors for this k-point
                H0_slice = nothing
                H_hf_slice = nothing
                Δ_slice = nothing
                H0_mat = nothing
                H_hf_mat = nothing
                Δ_mat = nothing
                GC.gc()
            end
            total_energy /= n_kpoints

            println("Energy: $total_energy")

            # Calculate error for DIIS
            error = Δ_total - Δ_old

            # Store in DIIS history
            push!(diis_error_history, error)
            push!(diis_delta_history, Δ_old)

            # Limit the history size
            if length(diis_error_history) > max_diis_history
                popfirst!(diis_error_history)
                popfirst!(diis_delta_history)
            end

            # Apply DIIS if we have enough history
            if length(diis_error_history) >= 2
                Δ_total = diis_extrapolate(diis_error_history, diis_delta_history)
                println("Applied DIIS extrapolation")
            end

            # Check convergence
            if abs(total_energy - energy_prev) < tol
                println("Converged after $iter iterations")

                # Print order parameters for this converged state
                println("Order parameters for converged state '$name':")

                # Compute order parameters by averaging over k-points
                order_params_sum = Dict(
                    "FM_Sx" => 0.0im, "FM_Sy" => 0.0im, "FM_Sz" => 0.0im,
                    "AFM_Sx" => 0.0im, "AFM_Sy" => 0.0im, "AFM_Sz" => 0.0im,
                    "VFM_Tx" => 0.0im, "VFM_Ty" => 0.0im, "VFM_Tz" => 0.0im,
                    "VAFM_Tx" => 0.0im, "VAFM_Ty" => 0.0im, "VAFM_Tz" => 0.0im,
                    "Sublattice_Pol" => 0.0im
                )

                for kx_idx in 1:N_kx, ky_idx in 1:N_ky
                    Δ_slice = Δ_total * setelt(ikx(kx_idx)) * setelt(iky(ky_idx))
                    Δ_mat = matrix(Δ_slice, orbital_indices..., orbital_indices'...)

                    # Compute order parameters for this k-point
                    k_order_params = compute_order_parameters(Δ_mat, levels, p)

                    # Add to sum
                    for key in keys(order_params_sum)
                        order_params_sum[key] += k_order_params[key]
                    end

                    # Clean up intermediate tensors for this k-point
                    Δ_slice = nothing
                    Δ_mat = nothing
                    k_order_params = nothing
                    GC.gc()
                end

                # Average over k-points
                order_params = Dict(key => value / n_kpoints for (key, value) in order_params_sum)

                for (key, value) in order_params
                    println("  $key: $value")
                end

                # Store results and clean up
                results[name] = (total_energy, Δ_total, order_params)

                # Clean up intermediate tensors for this HF run
                H_hf_total = nothing
                GC.gc()

                break
            end
            energy_prev = total_energy

            # Clean up intermediate tensors for this iteration
            H_hf_total = nothing
            GC.gc()
        end
    end

    # Find the result with the lowest energy
    min_energy = Inf
    min_name = ""
    min_Δ = nothing
    min_order_params = nothing

    for (name, (energy, Δ, order_params)) in results
        println("Result for $name: Energy = $energy")
        if energy < min_energy
            min_energy = energy
            min_name = name
            min_Δ = Δ
            min_order_params = order_params
        end
    end

    println("\nLOWEST ENERGY RESULT")
    println("====================")
    println("Initial state: $min_name")
    println("Energy: $min_energy")
    println("\nOrder parameters:")
    for (key, value) in min_order_params
        println("  $key: $value")
    end
    println("====================")

    # Save only the lowest energy result
    @save filename min_Δ min_energy min_order_params orbital_indices momentum_indices k_grid_vals_x k_grid_vals_y

    println("Calculation complete. Lowest energy result saved to $filename")

    return min_Δ, min_energy, min_order_params
end

# Helper function to compute order parameters for a single k-point
function compute_order_parameters(Δ::AbstractMatrix, levels::Int, p::Int)
    matrix_size = size(Δ, 1)
    nF = round(Int, real(tr(Δ)))
    if nF == 0
        return Dict(
            "FM_Sx" => 0.0im, "FM_Sy" => 0.0im, "FM_Sz" => 0.0im,
            "AFM_Sx" => 0.0im, "AFM_Sy" => 0.0im, "AFM_Sz" => 0.0im,
            "VFM_Tx" => 0.0im, "VFM_Ty" => 0.0im, "VFM_Tz" => 0.0im,
            "VAFM_Tx" => 0.0im, "VAFM_Ty" => 0.0im, "VAFM_Tz" => 0.0im,
            "Sublattice_Pol" => 0.0im
        )
    end
    norm = 1.0 / nF

    # Initialize expectation values
    ops = Dict(
        "FM_Sx" => 0.0im, "FM_Sy" => 0.0im, "FM_Sz" => 0.0im,
        "AFM_Sx" => 0.0im, "AFM_Sy" => 0.0im, "AFM_Sz" => 0.0im,
        "VFM_Tx" => 0.0im, "VFM_Ty" => 0.0im, "VFM_Tz" => 0.0im,
        "VAFM_Tx" => 0.0im, "VAFM_Ty" => 0.0im, "VAFM_Tz" => 0.0im,
        "Sublattice_Pol" => 0.0im
    )

    for i in 1:matrix_size
        v1, l1, s1, n1, S1 = decompose_index_valleyful(i, p, levels)
        lambda1 = (n1 == 0) ? 0 : S1

        # Diagonal operators (Sz, Tz, Sublattice)
        ops["FM_Sz"] += Δ[i, i] * s1
        ops["VFM_Tz"] += Δ[i, i] * v1
        # For n>0, S1 is sublattice sign. For n=0, S1=v1, so we must be careful.
        sublattice_sign = (n1 > 0) ? S1 : 0 # No sublattice DoF for n=0
        ops["Sublattice_Pol"] += Δ[i, i] * sublattice_sign
        ops["AFM_Sz"] += Δ[i, i] * s1 * sublattice_sign
        ops["VAFM_Tz"] += Δ[i, i] * v1 * sublattice_sign

        # Off-diagonal operators (Sx, Sy, Tx, Ty)
        # Find partner states with one flipped quantum number
        j_spin = find_index_valleyful(v1, l1, -s1, n1, S1, p, levels)
        if j_spin != -1
            ops["FM_Sx"] += Δ[j_spin, i] # <i|c_j^+ c_i|i> ~ <S_x>
            ops["FM_Sy"] += Δ[j_spin, i] * im * s1 # <i|c_j^+ c_i|i> ~ <S_y>
            ops["AFM_Sx"] += Δ[j_spin, i] * sublattice_sign
            ops["AFM_Sy"] += Δ[j_spin, i] * im * s1 * sublattice_sign
        end

        j_valley = find_index_valleyful(-v1, l1, s1, n1, S1, p, levels)
        if j_valley != -1
            ops["VFM_Tx"] += Δ[j_valley, i]
            ops["VFM_Ty"] += Δ[j_valley, i] * im * v1
            ops["VAFM_Tx"] += Δ[j_valley, i] * sublattice_sign
            ops["VAFM_Ty"] += Δ[j_valley, i] * im * v1 * sublattice_sign
        end
    end

    # Normalize
    for key in keys(ops)
        ops[key] *= norm
    end

    return ops
end

# Helper function to convert flat index to multi-index
function ind2sub(dims, idx)
    sub = []
    for d in dims
        push!(sub, mod1(idx, d))
        idx = div(idx - 1, d) + 1
    end
    return tuple(sub...)
end
