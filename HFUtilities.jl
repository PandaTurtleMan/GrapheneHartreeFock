
using LinearAlgebra
using Random

#diagonalize matrix utility
function extractMatrixOfEigenvectors(HHF)
    F = eigen(Hermitian(HHF))
    return F.vectors
end

#once you get the final delta matrix, extract the evectors with evalue 1 (i.e the filled states). These will be the highest eigenvalue eigenvectors.
function extractFilledOrbitals(nF,Δ)
    orbitals = extractMatrixOfEigenvectors(Δ)
    N = size(orbitals,1)
    filledOrbitals = orbitals[:,N-nF:N]
    return filledOrbitals
end

    function DIISErrorCoeffs(errorvecs, maxErrors=8)
        m = min(length(errorvecs), maxErrors)
        # DIIS requires at least two error vectors to be meaningful.
        m < 2 && return Float64[]

        # Take the last m error vectors from the history.
        evecs = errorvecs[end-m+1:end]

        # Build the (m+1)x(m+1) DIIS matrix.
        B = zeros(m + 1, m + 1)
        for i in 1:m
            for j in i:m # Only compute upper triangle, since B is symmetric
                dot_product = real(tr(evecs[i]' * evecs[j]))
                B[i, j] = dot_product
                B[j, i] = dot_product
            end
        end

        # Add constraints for the Lagrange multiplier.
        B[1:m, m + 1] .= 1.0
        B[m + 1, 1:m] .= 1.0
        # B[m+1, m+1] is already 0.0

        # Define the right-hand side of the equation B*c = b
        b = zeros(m + 1)
        b[m + 1] = 1.0

        # --- FIXES ---
        # 1. Solve B*c = b using the pseudoinverse for numerical stability.
        # 2. This replaces the incorrect "c = B / b'"
        c = pinv(B) * b

        # Return the first m coefficients, excluding the Lagrange multiplier.
        return c[1:m]
    end

function extrapolateFockMatrixWithDIIS(errorCoeffs, previousDensityMatrices, nF)
    result = zero(first(previousDensityMatrices))
    for (c, Δ) in zip(errorCoeffs, previousDensityMatrices)
        result += c * Δ
    end

    # Purify to ensure idempotency
    F = eigen(result)
    U = F.vectors
    Λ = F.values
    # Sort descending, keep nF largest
    idx = sortperm(real.(Λ), rev=true)[1:nF]
    Δ_new = U[:, idx] * U[:, idx]'
    return Δ_new
end

function random_binary_density_matrix(N::Int, trace_val::Int)
    # --- Input Validation ---
    if N <= 0
        throw(ArgumentError("Matrix dimension N must be positive."))
    end
    if trace_val < 0
        throw(ArgumentError("Trace must be non-negative."))
    end
    if trace_val > N
        throw(ArgumentError("Trace cannot exceed matrix dimension N."))
    end

    # --- Step 1: Generate a random orthonormal basis (eigenvectors) ---
    # Start with a random complex matrix from a Gaussian distribution.
    A = randn(ComplexF64, N, N)
    # Construct a random Hermitian matrix from A. This ensures the eigenvectors
    # form a random basis.
    H_rand = A + A'
        # The eigenvectors of any Hermitian matrix form a complete orthonormal basis.
    F = eigen(H_rand)
    U = F.vectors # This is our random unitary matrix (basis)

    # --- Step 2: Create binary eigenvalues ---
    # Create eigenvalue vector with exactly trace_val ones and (N - trace_val) zeros
    binary_eigs = zeros(Float64, N)
    binary_eigs[1:trace_val] .= 1.0

    # Randomly shuffle the eigenvalues to avoid bias in which eigenvectors
    # correspond to the non-zero eigenvalues
    shuffle!(binary_eigs)

    # --- Step 3: Construct the density matrix ---
    # Combine the random basis (U) with the binary eigenvalues.
    # The resulting matrix Δ = U * Λ * U' is guaranteed to have exactly
    # trace_val eigenvalues equal to 1 and (N - trace_val) eigenvalues equal to 0.
    Δ = U * Diagonal(binary_eigs) * U'

# Ensure the output is explicitly Hermitian to maintain type stability and
# leverage optimized methods for Hermitian matrices.
return Hermitian(Δ)
end

function compute_and_print_order_parameters(Δ::AbstractMatrix, levels::Int, p::Int)
    matrix_size = size(Δ, 1)
    nF = round(Int, real(tr(Δ)))
    if nF == 0
        println("No particles, all order parameters are zero.")
        return
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

    println("\n--- Hartree-Fock Order Parameters ---")
    println("Normalization Factor (1/nF): $norm")
    println("\nSpin Polarization:")
    println("  - Ferromagnetic (FM):   Sx=%.4f, Sy=%.4f, Sz=%.4f\n", real(norm*ops["FM_Sx"]), real(norm*ops["FM_Sy"]), real(0.5*norm*ops["FM_Sz"]))
    println( "  - Antiferro (AFM):      Sx=%.4f, Sy=%.4f, Sz=%.4f\n", real(norm*ops["AFM_Sx"]), real(norm*ops["AFM_Sy"]), real(0.5*norm*ops["AFM_Sz"]))

    println("\nValley Polarization:")
    println("  - Valley-FM (VFM):      Tx=%.4f, Ty=%.4f, Tz=%.4f\n", real(norm*ops["VFM_Tx"]), real(norm*ops["VFM_Ty"]), real(0.5*norm*ops["VFM_Tz"]))
    println("  - Valley-AFM (VAFM):    Tx=%.4f, Ty=%.4f, Tz=%.4f\n", real(norm*ops["VAFM_Tx"]), real(norm*ops["VAFM_Ty"]), real(0.5*norm*ops["VAFM_Tz"]))

    println("\nSublattice Polarization:")
    println( "  - <Σ_z>: %.4f\n", real(norm*ops["Sublattice_Pol"]))
    println("-------------------------------------\n")
end

# DIIS implementation for Hartree-Fock convergence acceleration
function diis_extrapolate(error_history::Vector{ITensor}, delta_history::Vector{ITensor})
    n = length(error_history)

    # Build the B matrix
    B = zeros(n+1, n+1)
    for i in 1:n
        for j in 1:i
            # Compute the inner product between error vectors
            B[i,j] = real(scalar(dag(error_history[i]) * error_history[j]))
            B[j,i] = B[i,j]  # Symmetric
        end
        B[i, n+1] = -1
        B[n+1, i] = -1
    end
    B[n+1, n+1] = 0

    # Build the right-hand side vector
    b = zeros(n+1)
    b[n+1] = -1

    # Solve the linear system
    c = B \ b

    # Extrapolate the new density matrix
    Δ_new = zero(delta_history[1])
    for i in 1:n
        Δ_new += c[i] * delta_history[i]
    end

    return Δ_new
end

function build_zeeman_density_matrix(levels, harmonics, p, q, L, nF, orbital_indices, momentum_indices, k_grid_vals_x, k_grid_vals_y, sz_value)
    i_n, i_s, i_K, i_l = orbital_indices
    ikx, iky = momentum_indices
    N_kx, N_ky = length(k_grid_vals_x), length(k_grid_vals_y)

    Δ = ITensor(orbital_indices'..., orbital_indices..., ikx, iky)

    # Set Zeeman parameters
    zeeman_vector = [0.0, 0.0, sz_value]  # Sz term only
    valley_zeeman_vector = [0.0, 0.0, 0.0]
    m = 0.0

    for kx_idx in 1:N_kx, ky_idx in 1:N_ky
        kx_val = k_grid_vals_x[kx_idx]
        ky_val = k_grid_vals_y[ky_idx]

        # Build Hamiltonian with Zeeman term
        H_zeeman = Hamiltonian(kx_val, ky_val, levels, harmonics, 0.0, p, q, L, zeeman_vector, valley_zeeman_vector, m)
        H_mat = Matrix(H_zeeman)

        # Diagonalize and create density matrix
        F = eigen(Hermitian(H_mat))
        U = F.vectors
        Δ_mat = U[:, 1:nF] * U[:, 1:nF]'

        # Store in ITensor
        for orb1 in 1:size(H_mat, 1), orb2 in 1:size(H_mat, 1)
            # Convert flat indices to quantum numbers
            valley1, l1, spin1, n1, S1 = decompose_index_valleyful(orb1, p, levels)
            valley2, l2, spin2, n2, S2 = decompose_index_valleyful(orb2, p, levels)

            # Map to ITensor indices
            Δ[i_n(n1+1), i_s(spin1 == 1 ? 1 : 2), i_K(valley1+1), i_l(l1),
              i_n'(n2+1), i_s'(spin2 == 1 ? 1 : 2), i_K'(valley2+1), i_l'(l2),
              ikx(kx_idx), iky(ky_idx)] = Δ_mat[orb1, orb2]
        end
    end

    return Δ
end

function build_valley_zeeman_density_matrix(levels, harmonics, p, q, L, nF, orbital_indices, momentum_indices, k_grid_vals_x, k_grid_vals_y, tz_value)
    i_n, i_s, i_K, i_l = orbital_indices
    ikx, iky = momentum_indices
    N_kx, N_ky = length(k_grid_vals_x), length(k_grid_vals_y)

    Δ = ITensor(orbital_indices'..., orbital_indices..., ikx, iky)

    # Set valley Zeeman parameters
    zeeman_vector = [0.0, 0.0, 0.0]
    valley_zeeman_vector = [0.0, 0.0, tz_value]  # Valley Sz term only
    m = 0.0

    for kx_idx in 1:N_kx, ky_idx in 1:N_ky
        kx_val = k_grid_vals_x[kx_idx]
        ky_val = k_grid_vals_y[ky_idx]

        # Build Hamiltonian with valley Zeeman term
        H_valley_zeeman = Hamiltonian(kx_val, ky_val, levels, harmonics, 0.0, p, q, L, zeeman_vector, valley_zeeman_vector, m)
        H_mat = Matrix(H_valley_zeeman)

        # Diagonalize and create density matrix
        F = eigen(Hermitian(H_mat))
        U = F.vectors
        Δ_mat = U[:, 1:nF] * U[:, 1:nF]'

        # Store in ITensor
        for orb1 in 1:size(H_mat, 1), orb2 in 1:size(H_mat, 1)
            # Convert flat indices to quantum numbers
            valley1, l1, spin1, n1, S1 = decompose_index_valleyful(orb1, p, levels)
            valley2, l2, spin2, n2, S2 = decompose_index_valleyful(orb2, p, levels)

            # Map to ITensor indices
            Δ[i_n(n1+1), i_s(spin1 == 1 ? 1 : 2), i_K(valley1+1), i_l(l1),
              i_n'(n2+1), i_s'(spin2 == 1 ? 1 : 2), i_K'(valley2+1), i_l'(l2),
              ikx(kx_idx), iky(ky_idx)] = Δ_mat[orb1, orb2]
        end
    end

    return Δ
end

function build_mass_term_density_matrix(levels, harmonics, p, q, L, nF, orbital_indices, momentum_indices, k_grid_vals_x, k_grid_vals_y, mass_value)
    i_n, i_s, i_K, i_l = orbital_indices
    ikx, iky = momentum_indices
    N_kx, N_ny = length(k_grid_vals_x), length(k_grid_vals_y)

    Δ = ITensor(orbital_indices'..., orbital_indices..., ikx, iky)

    # Set mass term parameters
    zeeman_vector = [0.0, 0.0, 0.0]
    valley_zeeman_vector = [0.0, 0.0, 0.0]
    m = mass_value  # Mass term

    for kx_idx in 1:N_kx, ky_idx in 1:N_ky
        kx_val = k_grid_vals_x[kx_idx]
        ky_val = k_grid_vals_y[ky_idx]

        # Build Hamiltonian with mass term
        H_mass = Hamiltonian(kx_val, ky_val, levels, harmonics, 0.0, p, q, L, zeeman_vector, valley_zeeman_vector, m)
        H_mat = Matrix(H_mass)

        # Diagonalize and create density matrix
        F = eigen(Hermitian(H_mat))
        U = F.vectors
        Δ_mat = U[:, 1:nF] * U[:, 1:nF]'

        # Store in ITensor
        for orb1 in 1:size(H_mat, 1), orb2 in 1:size(H_mat, 1)
            # Convert flat indices to quantum numbers
            valley1, l1, spin1, n1, S1 = decompose_index_valleyful(orb1, p, levels)
            valley2, l2, spin2, n2, S2 = decompose_index_valleyful(orb2, p, levels)

            # Map to ITensor indices
            Δ[i_n(n1+1), i_s(spin1 == 1 ? 1 : 2), i_K(valley1+1), i_l(l1),
              i_n'(n2+1), i_s'(spin2 == 1 ? 1 : 2), i_K'(valley2+1), i_l'(l2),
              ikx(kx_idx), iky(ky_idx)] = Δ_mat[orb1, orb2]
        end
    end

    return Δ
end

# Helper function to build random density matrix
function build_random_density_matrix(levels, p, nF, orbital_indices, momentum_indices, k_grid_vals_x, k_grid_vals_y; seed=nothing)
    i_n, i_s, i_K, i_l = orbital_indices
    ikx, iky = momentum_indices
    N_kx, N_ky = length(k_grid_vals_x), length(k_grid_vals_y)

    Δ = ITensor(orbital_indices'..., orbital_indices..., ikx, iky)

    # The total dimension of the orbital space at one k-point
    N_orb = dim(i_n) * dim(i_s) * dim(i_K) * dim(i_l)

    for kx_idx in 1:N_kx, ky_idx in 1:N_ky
        # Generate a random density matrix for this k-point
        Δ_mat = random_hermitian_density_matrix(N_orb, nF; seed=seed)

        # Store in ITensor
        for orb1 in 1:N_orb, orb2 in 1:N_orb
            # Convert flat indices to multi-indices
            idx1 = ind2sub((dim(i_n), dim(i_s), dim(i_K), dim(i_l)), orb1)
            idx2 = ind2sub((dim(i_n), dim(i_s), dim(i_K), dim(i_l)), orb2)

            Δ[i_n(idx1[1]), i_s(idx1[2]), i_K(idx1[3]), i_l(idx1[4]),
              i_n'(idx2[1]), i_s'(idx2[2]), i_K'(idx2[3]), i_l'(idx2[4]),
              ikx(kx_idx), iky(ky_idx)] = Δ_mat[orb1, orb2]
        end
    end

    return Δ
end

# Helper function to build non-interacting density matrix
function build_noninteracting_density_matrix(levels, harmonics, p, q, L, nF, orbital_indices, momentum_indices, k_grid_vals_x, k_grid_vals_y)
    # Build non-interacting Hamiltonian and diagonalize at each k-point
    i_n, i_s, i_K, i_l = orbital_indices
    ikx, iky = momentum_indices
    N_kx, N_ky = length(k_grid_vals_x), length(k_grid_vals_y)

    Δ = ITensor(orbital_indices'..., orbital_indices..., ikx, iky)

    for kx_idx in 1:N_kx, ky_idx in 1:N_ky
        kx_val = k_grid_vals_x[kx_idx]
        ky_val = k_grid_vals_y[ky_idx]

        # Build non-interacting Hamiltonian
        H0 = Hamiltonian(kx_val, ky_val, levels, harmonics, 0.0, p, q, L, [0,0,0], [0,0,0], 0.0)
        H_mat = Matrix(H0)

        # Diagonalize and create density matrix
        F = eigen(Hermitian(H_mat))
        U = F.vectors
        Δ_mat = U[:, 1:nF] * U[:, 1:nF]'

        # Store in ITensor
        for orb1 in 1:size(H_mat, 1), orb2 in 1:size(H_mat, 1)
            # Convert flat indices to quantum numbers
            valley1, l1, spin1, n1, S1 = decompose_index_valleyful(orb1, p, levels)
            valley2, l2, spin2, n2, S2 = decompose_index_valleyful(orb2, p, levels)

            # Map to ITensor indices
            Δ[i_n(n1+1), i_s(spin1 == 1 ? 1 : 2), i_K(valley1+1), i_l(l1),
              i_n'(n2+1), i_s'(spin2 == 1 ? 1 : 2), i_K'(valley2+1), i_l'(l2),
              ikx(kx_idx), iky(ky_idx)] = Δ_mat[orb1, orb2]
        end
    end

    return Δ
end






