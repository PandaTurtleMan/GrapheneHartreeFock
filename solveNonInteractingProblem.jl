
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

function Hamiltonian(k_x, k_y, levels, projs, phi, p, q, L)
    g = gcd(p, q)
    p = div(p, g)
    q = div(q, g)

    matrix_size = 2*levels*p
    hamiltonian = zeros(ComplexF64, matrix_size, matrix_size)

    # Precompute common values
    #fourierdict = fourier_dict(mnrange)
    B = (p/q)/L^2

    # Optimized matrix construction
    @inbounds Threads.@threads for i in 1:matrix_size
        S1 = iseven(i) ? -1 : 1
        l1 = div(i-1, 2*levels) + 1
        m = div(mod(i-1, 2*levels), 2)

        for j in 1:matrix_size
            S2 = iseven(j) ? -1 : 1
            l2 = div(j-1, 2*levels) + 1
            n = div(mod(j-1, 2*levels), 2)
            for z in 1:length(projs)
                #dummy valley indices!
                hamiltonian[i,j] += projs[z]*matrixElement(k_x,k_y,S1,S2,l1,l2,m,n,z,B,L,0,0)
            end
            if m == n && S1 == S2 && l1 == l2
                hamiltonian[i,j] += evaluateLandauLevelMatrixElements(k_x,k_y,m,n,B,S1,S2)
            end

        end
    end
    return Hermitian(hamiltonian)
end

#hamiltonian with spin, can turn on Zeeman shift
function spinfulHamiltonian(k_x,k_y,levels,projs,phi,p,q,L,BohrMagneton)
    g = gcd(p, q)
    p = div(p, g)
    q = div(q, g)

    matrix_size = 4*levels*p
    hamiltonian = zeros(ComplexF64, matrix_size, matrix_size)

    # Precompute common values
    #fourierdict = fourier_dict(mnrange)
    B = (p/q)/L^2

    # Optimized matrix construction
    @inbounds Threads.@threads for i in 1:matrix_size
        S1 = iseven(i) ? -1 : 1
        spin1 = isOddMod4(i) ? -1 : 1
        l1 = div(i-1, 4*levels) + 1
        m = div(mod(i-1, 4*levels), 4)

        for j in 1:matrix_size
            S2 = iseven(j) ? -1 : 1
            spin2 = isOddMod4(j) ? -1 : 1
            l2 = div(j-1, 4*levels) + 1
            n = div(mod(j-1, 4*levels), 4)
            if spin1 != spin2
                hamiltonian[i,j] = 0
                continue
            end

            for z in 1:length(projs)
                hamiltonian[i,j] += projs[z]*matrixElement(k_x,k_y,S1,S2,l1,l2,m,n,z,B,L,0,0)
            end
            if m == n && S1 == S2 && l1 == l2 && spin1 == spin2
                hamiltonian[i,j] += evaluateLandauLevelMatrixElements(k_x,k_y,m,n,B,S1,S2)
            end
            if spin1 == spin2
               hamiltonian[i,j] += spin1*0.5*BohrMagneton*B
            end
            if spin1 != spin2
                hamiltonian[i,j] = 0
            end

        end
    end
    return Hermitian(hamiltonian)
end

#hamiltonian with valley, block diagonal in valley index (for now)
function spinfulValleyfulHamiltonian(k_x, k_y, levels, projs, phi, p, q, L, BohrMagneton)
    g = gcd(p, q)
    p = div(p, g)
    q = div(q, g)
    matrix_size = 8 * levels * p + 4 * p
    hamiltonian = zeros(ComplexF64, matrix_size, matrix_size)
    B = (p / q) / L^2

    for i in 1:matrix_size
        valley1, l1, spin_sign1, n1, S1 = decompose_index_valleyful(i, p, levels)
        for j in 1:matrix_size
            valley2, l2, spin_sign2, n2, S2 = decompose_index_valleyful(j, p, levels)
            if valley1 != valley2 || spin_sign1 != spin_sign2
                hamiltonian[i, j] = 0
                continue
            end
            for z in 1:length(projs)
                hamiltonian[i, j] += projs[z] * matrixElement(k_x, k_y, S1, S2, l1, l2, n1, n2, z, B, L, valley1, valley2)
            end
            if n1 == n2 && S1 == S2 && l1 == l2
                hamiltonian[i, j] += evaluateLandauLevelMatrixElements(k_x, k_y, n1, n2, B, S1, S2)
            end
            if i == j
                hamiltonian[i, j] += spin_sign1 * 0.5 * BohrMagneton * B
            end
        end
    end
    return Hermitian(hamiltonian)
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

    function plot_spectrum_along_kx(k_x, ky_range, levels, harmonics, phi, p, q, L)
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
                #H = spinfulValleyfulHamiltonian(k_x, k_y, levels, projs, phi, p, q, L,0)
                H = Hamiltonian(k_x, k_y, levels, projs, phi, p, q, L)
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




