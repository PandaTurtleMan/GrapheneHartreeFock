using LinearAlgebra, ProgressMeter
include("Utilities.jl")
include("solveNonInteractingProblem.jl")
include("HFUtilities.jl")

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
    progress = Progress(max_iter, desc="Hartree-Fock Iterations: ",
                        barglyphs=BarGlyphs('|','█', '▁', '|', ' '),
                        output=stderr, showspeed=true)
    for iter in 1:max_iter
        # Construct and diagonalize HF Hamiltonian
        H_hf = construct_hf_hamiltonian(
            H0, V, Δ,
            )
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
    H0::AbstractMatrix, Δ::AbstractMatrix,
    k_x::Real, k_y::Real,
    levels::Int, L::Real, l_B::Real,
    screening::Function, cutoff::Real,V
    )
    H_hf = construct_hf_hamiltonian(H0, V,Δ)
    return 0.5 * real(tr(Δ * (H0 + H_hf)))
end

