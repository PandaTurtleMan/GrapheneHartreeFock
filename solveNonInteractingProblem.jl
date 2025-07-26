
using Distributed

using SpecialFunctions
using Combinatorics
using CUDA
using SpecialFunctions
using LinearAlgebra
using SpecialPolynomials
using Polynomials
using FastGaussQuadrature
using SparseArrays
using ProgressMeter
using PyPlot
using Combinatorics

include("Utilities.jl")
include("LandauLevels.jl")
include("matrixElements.jl")


export Hamiltonian


function Hamiltonian(k_x, k_y, levels, harmonics, phi, p, q, L, zeeman_vector, valley_zeeman_vector, m)
    g = gcd(p, q)
    p_eff, q_eff = div(p, g), div(q, g)
    matrix_size = 8 * levels * p + 4 * p
    hamiltonian = zeros(ComplexF64, matrix_size, matrix_size)
    B = (p / q) / L^2

    for i in 1:matrix_size
        valley1, l1, spin1, n1, S1 = decompose_index_valleyful(i, p, levels)
        lambda1 = (n1 == 0) ? 0 : S1 # Sublattice component for wavefunction

        for j in 1:matrix_size
            valley2, l2, spin2, n2, S2 = decompose_index_valleyful(j, p, levels)
            lambda2 = (n2 == 0) ? 0 : S2

            # 1. External Potential Term (from C4 harmonics)
            potential_term = matrixElement(k_x, k_y, n1, n2, l1, l2, lambda1, lambda2, spin1, spin2, valley1, valley2, L, p, q, harmonics)
            hamiltonian[i, j] += potential_term

            # 2. On-site terms (diagonal in all quantum numbers)
            if i == j
                # Kinetic Energy (Landau Levels)
                hamiltonian[i, j] += evaluateLandauLevelMatrixElement(n1, B, lambda1, m)

                # S_z and T_z Zeeman terms
                hamiltonian[i, j] += 0.5 * zeeman_vector[3] * spin1
                hamiltonian[i, j] += 0.5 * valley_zeeman_vector[3] * valley1
            end

            # 3. Off-diagonal Zeeman terms (spin-flipping)
            if valley1 == valley2 && l1 == l2 && n1 == n2 && S1 == S2 && spin1 != spin2
                # S_x term
                hamiltonian[i, j] += 0.5 * zeeman_vector[1]
                # S_y term (s1=1, s2=-1 means <up|H|down>)
                if spin1 == 1 && spin2 == -1
                    hamiltonian[i, j] -= 0.5im * zeeman_vector[2]
                else # (s1=-1, s2=1 means <down|H|up>)
                    hamiltonian[i, j] += 0.5im * zeeman_vector[2]
                end
            end

            # 4. Off-diagonal Valley Zeeman terms (valley-flipping)
            if spin1 == spin2 && l1 == l2 && n1 == n2 && S1 == S2 && valley1 != valley2
                # T_x term
                hamiltonian[i, j] += 0.5 * valley_zeeman_vector[1]
                # T_y term (v1=1, v2=-1 means <K|H|K'>)
                if valley1 == 1 && valley2 == -1
                    hamiltonian[i, j] -= 0.5im * valley_zeeman_vector[2]
                else # (v1=-1, v2=1 means <K'|H|K>)
                    hamiltonian[i, j] += 0.5im * valley_zeeman_vector[2]
                end
            end
        end
    end
    return Hermitian(hamiltonian)
end

"""
ITensorHamiltonian(indices, params; bias_terms=Dict())

Constructs the non-interacting Hamiltonian H0 as a BlockSparse ITensor.
Can optionally add bias terms for generating initial states.
    """
    function ITensorHamiltonian(indices, p, q, L, m_eff; bias_params=Dict())
        s, k, λ, n, l = indices["spin"], indices["valley"], indices["sublattice"], indices["ll"], indices["subband"]

        H0 = ITensor(s', k', λ', n', l', s, k, λ, n, l)

        B = (p / q) / L^2
        zeeman_vec = get(bias_params, "zeeman", [0.0, 0.0, 0.0])
        valley_zeeman_vec = get(bias_params, "valley_zeeman", [0.0, 0.0, 0.0])
        mass_term = get(bias_params, "mass", 0.0)

        for s_val in 1:dim(s), k_val in 1:dim(k), λ_val in 1:dim(λ), n_val in 1:dim(n), l_val in 1:dim(l)
            # Diagonal terms
            spin_sign = (s_val == 1) ? 1 : -1
            valley_sign = (k_val == 1) ? 1 : -1
            sublattice_sign = (λ_val == 1) ? 1 : -1
            ll_idx = n_val - 1 # 0-indexed

            # 1. Kinetic Energy
            diag_val = v_F * sqrt(2 * e * B * ll_idx) # Simplified Landau Level energy

            # 2. Biasing terms (for initial state generation)
            diag_val += 0.5 * zeeman_vec[3] * spin_sign
            diag_val += 0.5 * valley_zeeman_vec[3] * valley_sign
            diag_val += mass_term * sublattice_sign # Onsite mass term

            H0[s'(s_val), k'(k_val), λ'(λ_val), n'(n_val), l'(l_val),
               s(s_val), k(k_val), λ(λ_val), n(n_val), l(l_val)] = diag_val
        end

        # Placeholder for off-diagonal bias terms (Sx, Sy, etc.) if needed

        return H0
    end

    # Placeholder for ionic correction
    function ionic_background_correction(indices, params)
        # This function would compute the ionic potential contribution
        # and return it as a rank-2 ITensor with the same indices as H0.
        s, k, λ, n, l = indices["spin"], indices["valley"], indices["sublattice"], indices["ll"], indices["subband"]
        return ITensor(s', k', λ', n', l', s, k, λ, n, l) # Return zero tensor for now
    end


function spectrum(k_x, k_y, levels, projs, phi, p, q, L)
    hamiltonian = Hamiltonian(k_x, k_y, levels,projs, phi, p, q, L)
    return eigvals(hamiltonian)
end

#plots energy as a function of magnetic field at a given point in momentum space
function plotBandEnergies(k_x, k_y, levels, harmonics, phi, L, mnrange)
    projs = []
    for i in 1:length(harmonics)
        push!(projs,dotproduct(harmonics,compute_fourier_coefficients(i)))
    end

    num_points = 239
    x = range(0, 1, length=num_points)
    ys = Vector{Vector{Float64}}(undef, num_points)


    progress = Progress(num_points, barglyphs=BarGlyphs('|','█', '▁', '|', ' '),output=stderr, showspeed=true)
    lock_obj = ReentrantLock()


    @sync begin
        channel = Channel{Int}(num_points)

    @async begin
        for i in 1:num_points
            put!(channel, i)
        end
        close(channel)
    end

    for _ in 1:Threads.nthreads()
        @async begin
            while true
                try
                i = take!(channel)

                B_num = i
                B_den = 239


                evals = spectrum(k_x, k_y, levels, projs, phi, B_num, B_den, L)
                ys[i] = evals
                GC.gc()


                lock(lock_obj) do
                next!(progress; showvalues=[(:Current_B, "$B_num/$B_den")])
                end
                catch e
                isa(e, InvalidStateException) && break
                rethrow(e)
                    end
                end
            end
        end
    end

    figure(figsize=(12, 8))


    for i in 1:num_points
        if !isempty(ys[i])

            scatter([x[i] for _ in 1:length(ys[i])], ys[i],
            s=0.5, c="blue", alpha=0.5)
        end
    end

    # Adjust plot appearance
    xlabel("Magnetic Flux (ϕ/ϕ₀)")
    ylabel("Energy")
    title("Hofstadter Butterfly - Energy Spectrum")
    grid(true, alpha=0.3)
    tight_layout()

    # Save and show the plot
    savefig("hofstadter_butterfly.png", dpi=300)
    show()
end

    function plot_spectrum_along_ky(k_x, ky_range, levels, harmonics, phi, p, q, L)
        """
        Plots the non-interacting spectrum along k_x at fixed k_y.

        Parameters:
        k_y (float): Fixed k_y value
        kx_range (LinRange): Range of k_x values to evaluate
        levels (int): Number of Landau levels
        harmonics (Array): Fourier coefficients for external potential
            phi (float): Magnetic flux per unit cell
            p, q (int): Magnetic field parameters (B = p/(q*L^2))
            L (float): System size
            """
            energies = []

            # Precompute Fourier projection coefficients
            projs = []
            for i in 1:length(harmonics)
                push!(projs, dotproduct(harmonics, compute_fourier_coefficients(i)))
            end

            # Calculate energies for each k_x
            @showprogress "Computing spectrum..." for k_y in ky_range
                H = Hamiltonian(k_x, k_y, levels, projs, phi, p, q, L)
                #H = Hamiltonian(k_x, k_y, levels, projs, phi, p, q, L)
                evals = sort(real.(eigvals(H)))
                push!(energies, evals)
            end

            # Plot results
            fig = figure(figsize=(10, 6))
            for band in 1:length(energies[1])
                band_energies = [e[band] for e in energies]
                    plot(collect(ky_range), band_energies, lw=1.5)
                end

                title("Energy Spectrum at k_x = $k_x")
                xlabel(L"$k_y$")
                       ylabel(L"$E$")
                       grid(true, alpha=0.3)
                       tight_layout()
                       show()

                       return fig
            end

            function test_kx_periodicity()
                # Parameters for the test
                k_y = 0.0
                levels = 5
                harmonics = [1.0]  # Constant potential for simplicity
                phi = 0
                p = 10
                q = 11
                L = 1.0
                k_x0 = 0.1  # Arbitrary starting k_x

                # Compute Fourier projection coefficients
                projs = [dotproduct(harmonics, compute_fourier_coefficients(i)) for i in 1:length(harmonics)]

                    # Calculate expected period in k_x
                    period = 1

                    # Compute eigenvalues at k_x0 and k_x0 + period
                    H1 = spinfulValleyfulHamiltonian(k_x0, k_y, levels, projs, phi, p, q, L,0)
                    H2 = spinfulValleyfulHamiltonian(k_x0 + period, k_y, levels, projs, phi, p, q, L,0)
                    evals1 = sort(real.(eigvals(H1)))
                    evals2 = sort(real.(eigvals(H2)))

                    # Check if eigenvalues are identical within tolerance
                    diff = maximum(abs.(evals1 - evals2))
                    tolerance = 1e-8
                    if diff < tolerance
                        println("Test passed: Hamiltonian is periodic in k_x with period 2π/(qL)")
                    else
                        println("Test failed: Eigenvalues differ by up to $diff")
                    end
                    return diff
                end




