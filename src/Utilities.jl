using SpecialFunctions
using SpecialPolynomials
using Polynomials
using Combinatorics
using FFTW
using PyPlot
using ProgressMeter
using Hungarian

function safe_div(num::Real, den::Real; ϵ::Real=1e-12)
    abs(den) < ϵ ? num/ϵ : num/den
end

function theta(n)
   if n >= 0  return 1 else return 0 end
end

function isOddMod4(n)
    if mod(n,4) == 0 || mod(n,4) == 1
        return true
    else
        return false
    end
end

function integrationInPolarCoordinates(polarFunction, rCutoff)
    inner_integral(r) = QuadGK.quadgk(theta -> polarFunction(r, theta), 0, 2*pi)[1]
    outer_integral = QuadGK.quadgk(inner_integral, 0, rCutoff)
    return outer_integral[1]
end

function hermite_poly(n, z)
    if n == 0
        return 1.0
    end
    x = variable(Polynomial{Rational{Int}})
    coefficient_list = zeros(Int, n+1)
    coefficient_list[end] = 1
    poly = Hermite(coefficient_list)
    value = poly(z)
    return float(value)
end

function laguerre_poly(superscript, subscript, z)
    if subscript == 0
        return 1.0
    end
    x = variable(Polynomial{Rational{Int}})
    coefficient_list = zeros(Int, subscript+1)
    coefficient_list[end] = 1
    poly = Laguerre{superscript}(coefficient_list)
    value = poly(z)
    return float(value)
end

function laguerrePart(n,m,l_B,qx,qy)
    cPlus = im*l_B*(qx - im*qy)/sqrt(2)
    cMinus = im*l_B*(qx + im*qy)/sqrt(2)
    if n >= m
        return exp(0.5*cPlus*cMinus)*sqrt(factorial(m)/factorial(n))*(cPlus)^(n-m)*laguerre_poly(n-m,m,(-cPlus*cMinus))
    else
        return exp(0.5*cPlus*cMinus)*sqrt(factorial(n)/factorial(m))*(cMinus)^(m-n)*laguerre_poly(m-n,n,(-cPlus*cMinus))
    end

end

function integrateBesselFunctionFrom0ToA(nu,a,cutoff=10)
    result = 0
    for i in 1:cutoff
        result += besselj(nu + 2*i + 1,a)
    end
    return 2*result
end

function orderfunction(k1::Tuple{Int,Int}, k2::Tuple{Int,Int})
    return k1[1]^2 + k1[2]^2 >= k2[1]^2 + k2[2]^2
end

# Is this necessary?
function sortingalgo(mylist)
    if length(mylist) <= 1
        return mylist
    end
    pivot = mylist[1]
    T = eltype(mylist)
    left = T[]
    right = T[]
    for i in 2:length(mylist)
        if !orderfunction(mylist[i], pivot)
            push!(left, mylist[i])
        else
            push!(right, mylist[i])
        end
    end
    return vcat(sortingalgo(left), [pivot], sortingalgo(right))
end

function equivalentModC4(vec1, vec2)
    m1, n1 = vec1
    m2, n2 = vec2
    return (m1 == m2 && n1 == n2) ||
        (m1 == n2 && n1 == -m2) ||
        (m1 == -n2 && n1 == m2) ||
        (m1 == -m2 && n1 == -n2)
end

function orderedmnsequence(mnrange)
    return sortingalgo([(i, j) for i in -mnrange:mnrange, j in -mnrange:mnrange][:])
end

function get_key(dict, val)
    for (key, value) in dict
        val in value && return key
    end
    return "key doesn't exist"
end

function fourier_dict(mnrange)
    myrecivectors = orderedmnsequence(mnrange)
    fourierdict = Dict{Int, Vector{Tuple{Int,Int}}}()
    indexcounter = 0

    for vec in myrecivectors
        sorted = false
        for j in 0:(indexcounter - 1)
            if haskey(fourierdict, j) && equivalentModC4(vec, fourierdict[j][1])
                push!(fourierdict[j], vec)
                sorted = true
                break
            end
        end
        if !sorted
            fourierdict[indexcounter] = [vec]
            indexcounter += 1
        end
    end
    return fourierdict
end


#this should be changed to be more general, but for now leaving it like this
function get_c4_harmonic_vectors(index::Int)
    if index == 1
        return [(1, 0), (0, 1), (-1, 0), (0, -1)]
        elseif index == 2
        return [(1, 1), (-1, 1), (-1, -1), (1, -1)]
        elseif index == 3
        return [(2, 0), (0, 2), (-2, 0), (0, -2)]
        elseif index == 4
        # Note: (2,1) and (1,2) generate distinct C4 orbits
        return [(2, 1), (-1, 2), (-2, -1), (1, -2), (1, 2), (-2, 1), (-1, -2), (2, -1)]
    else
        return [] # Return empty for unsupported indices
    end
end


function dotproduct(vec1,vec2)
    result = 0
    len = min(length(vec1),length(vec2))
    for i in 1:len
        result += vec1[i]*vec2[i]
    end
    return result
end

function compute_2d_fft(f, xmin, xmax, ymin, ymax; Nx=256, Ny=256)
    dx = (xmax - xmin) / Nx
    dy = (ymax - ymin) / Ny

    x = range(xmin, xmax - dx, length=Nx)
    y = range(ymin, ymax - dy, length=Ny)

    # Initialize progress bar
    progress = Progress(Nx*Ny, desc="Computing FFT samples: ",
                        barglyphs=BarGlyphs('|','█','▁','|',' '),
                        output=stderr)

    samples = Matrix{ComplexF64}(undef, Nx, Ny)

    # Populate samples with progress updates
    for j in 1:Ny
        for i in 1:Nx
            samples[i, j] = f(x[i], y[j])
            next!(progress)
        end
    end

    F = fft(samples)
    F_shifted = fftshift(F)

    fx = fftfreq(Nx, 1/dx) |> fftshift
    fy = fftfreq(Ny, 1/dy) |> fftshift

    return F_shifted, fx, fy
end

function plot_fft_modulus_pyplot(F_shifted, fx, fy; window=nothing, title="Log Plot for Fourier Transform of Correlation Function", aspect_equal=true,log_scale=true,Colorbar=true,figsize=(8, 6))
    # Create figure
    fig = figure(figsize=figsize)

    # Compute modulus (absolute value)
    Z = abs.(F_shifted)

    # Apply logarithmic scaling if requested
    if log_scale
        Z = log10.(Z .+ 1e-15)  # Add small value to avoid log(0)
        cbar_label = "log10(|F| + 1e-15)"
    else
        cbar_label = "|F|"
    end

    # Create plot using pcolormesh for accurate axis positioning
    ax = gca()
    mesh = ax.pcolormesh(fx, fy, Z', shading="auto")

    # Set window limits if specified
    if window !== nothing
        fx_min, fx_max, fy_min, fy_max = window
        xlim(fx_min, fx_max)
        ylim(fy_min, fy_max)
    end

    # Set labels and title
    xlabel("f_x")
    ylabel("f_y")
    PyPlot.title(title)  # Fixed: Explicitly call PyPlot.title

    # Set aspect ratio if requested
    if aspect_equal
        ax.set_aspect("equal")
    end

    # Add colorbar
    if Colorbar
        cbar = colorbar(mesh)
        cbar.set_label(cbar_label)
    end

    return fig
end


function fix_phase!(v::AbstractVector)
    idx = argmax(abs.(v))
    c = v[idx]
    if abs(c) < 1e-12
        return v
    end
    phase_factor = conj(c) / abs(c)
    v .= v .* phase_factor
    return v
end

function fix_phase!(M::AbstractMatrix)
    for j in 1:size(M, 2)
        fix_phase!(view(M, :, j))
    end
    return M
end

function decompose_index_valleyful(i::Int, p::Int, levels::Int)
    states_per_valley = p * (2 + 4 * levels)
    valley = div(i - 1, states_per_valley)
    rem1 = mod(i - 1, states_per_valley)
    l = div(rem1, (2 + 4 * levels)) + 1
    rem2 = mod(rem1, (2 + 4 * levels))
    spin_index = div(rem2, (1 + 2 * levels))
    orb_index = mod(rem2, (1 + 2 * levels))

    if orb_index == 0
        n = 0
        #S = valley == 0 ? 1 : -1
        S = valley == 0 ? -1 : 1
    else
        idx = orb_index - 1
        n = div(idx, 2) + 1
        sublattice = mod(idx, 2)
        S = sublattice == 0 ? 1 : -1
    end

    spin_sign = spin_index == 0 ? 1 : -1
    return valley, l, spin_sign, n, S
end

function groupBandsByLandauLevel(; levels, p, nF, spin, valley)
    # Calculate degeneracies
    if valley
        degeneracy_zeroth = 4 * p
        degeneracy_per_level = 4 * p  # For each n>0 (both positive and negative branches)
        elseif spin
        degeneracy_zeroth = 4 * p
        degeneracy_per_level =  2*p
    else
        degeneracy_zeroth = 2*p
        degeneracy_per_level = p
    end

    groups = Vector{Vector{Int}}()
    total_states = valley ? (8 * levels * p + 4 * p) :
        spin ? (4 * levels * p) : (2 * levels * p)
    nF = min(nF, total_states)
    band_counter = 1

    # Negative energy levels (n = levels to 1, descending order)
    for n in levels:-1:1
        if band_counter > nF
            break
        end
        group_size = min(degeneracy_per_level, nF - band_counter + 1)
        group = collect(band_counter:(band_counter + group_size - 1))
        push!(groups, group)
        band_counter += group_size
    end

    # Zeroth Landau level
    if band_counter <= nF
        group_size = min(degeneracy_zeroth, nF - band_counter + 1)
        group = collect(band_counter:(band_counter + group_size - 1))
        push!(groups, group)
        band_counter += group_size
    end

    # Positive energy levels (n = 1 to levels, ascending order)
    for n in 1:levels
        if band_counter > nF
            break
        end
        group_size = min(degeneracy_per_level, nF - band_counter + 1)
        group = collect(band_counter:(band_counter + group_size - 1))
        push!(groups, group)
        band_counter += group_size
    end

    return groups
end

function fix_gauge!(vecs_current::Matrix{ComplexF64}, vecs_previous::Matrix{ComplexF64})
    n = size(vecs_current, 2)

    # Compute overlap matrix and get absolute values for Hungarian algorithm
    # M[i, j] = |<previous_i | current_j>|
    M = abs.(vecs_previous' * vecs_current)

    # Use Hungarian algorithm to find optimal assignment (maximize total overlap)
    # We negate M to convert to a minimization problem, as hungarian solves minimization
    assignment, _ = hungarian(-M)

    # Reorder bands according to optimal assignment
    new_vecs = similar(vecs_current)
    for (prev_band_idx, current_band_idx) in enumerate(assignment)
        # The hungarian output 'assignment' maps column indices of the cost matrix
        # (which correspond to 'current' bands) to row indices (which correspond to 'previous' bands).
        # So, assignment[current_band_idx] = prev_band_idx means current_band_idx should be
        # mapped to prev_band_idx.
        # However, the hungarian package returns `assignment` such that `assignment[i]` is the
        # column matched with row `i`. If `M` is `U'*V`, row `i` corresponds to `U_i`,
        # and column `j` to `V_j`. So `assignment[i]` is the index `j` that `U_i` is best matched with.
        # Therefore, we want `new_vecs[:, i]` to be `vecs_current[:, assignment[i]]`.
        new_vecs[:, prev_band_idx] = vecs_current[:, assignment[prev_band_idx]]
    end

    # Fix phases to make overlaps real and positive
    # We now compute the overlap with the reordered vectors
    overlap_matrix = vecs_previous' * new_vecs
    for band in 1:n
        c = overlap_matrix[band, band] # This is <vecs_previous[:, band] | new_vecs[:, band]>
        if abs(c) > 1e-12
            phase_factor = conj(c) / abs(c) # Rotate new_vecs[:, band] by -angle(c)
            new_vecs[:, band] .*= phase_factor
        else
            @warn "Very small overlap for band $band (|overlap| = $(abs(c))) when fixing gauge."
        end
    end

    return new_vecs
end

"""
setup_indices(params)

Creates all necessary ITensor Index objects for the simulation.
"""
function setup_indices(p, levels, kx_grid, ky_grid)
    # Quantum number indices with BlockSparse structure (QN conservation)
    # Each state is labeled by a set of quantum numbers.
    # We define the space for each quantum number.
    spin_space = [QN("Sz", sz) => 1 for sz in (1, -1)]
    valley_space = [QN("Kz", K) => 1 for K in (1, -1)]
    sublattice_space = [QN("Lz", λ) => 1 for λ in (1, -1)]
    ll_space = [QN("N", n) => 1 for n in 0:levels-1]
    subband_space = [QN("l", l) => 1 for l in 0:p-1]

    # Create the ITensor indices
    s = Index(spin_space..., name="spin")
    k = Index(valley_space..., name="valley")
    λ = Index(sublattice_space..., name="sublattice")
    n = Index(ll_space..., name="ll")
    l = Index(subband_space..., name="subband")

    # Brillouin Zone momentum indices (regular indices, no QNs)
    ikx = Index(length(kx_grid), "kx")
    iky = Index(length(ky_grid), "ky")

    return Dict(
        "spin" => s, "valley" => k, "sublattice" => λ,
        "ll" => n, "subband" => l, "kx" => ikx, "ky" => iky
    )
end

function random_hermitian_density_matrix(N::Int, nF::Int; seed::Union{Int, Nothing}=nothing)
    """
    Generates a random Hermitian density matrix of size N×N with trace nF.

    Parameters:
    - N: Size of the matrix (N×N)
    - nF: Desired trace of the matrix
    - seed: Optional random seed for reproducibility

    Returns:
    - Δ: A Hermitian matrix with trace nF
    """

    # Set random seed if provided
    if seed !== nothing
        Random.seed!(seed)
    end

    # Input validation
    if N <= 0
        throw(ArgumentError("Matrix dimension N must be positive."))
    end
    if nF < 0
        throw(ArgumentError("Trace must be non-negative."))
    end
    if nF > N
        throw(ArgumentError("Trace cannot exceed matrix dimension N."))
    end

    # Method 1: Generate a random unitary matrix and create a projector
    # This ensures the matrix is idempotent (Δ² = Δ) which is a property of density matrices

    # Generate a random complex matrix
    A = randn(ComplexF64, N, N)

    # Make it Hermitian (A + A† is always Hermitian)
    H_rand = A + A'

    # Diagonalize to get a random unitary matrix
    F = eigen(Hermitian(H_rand))
    U = F.vectors

    # Create eigenvalues: nF ones and (N-nF) zeros
    eigs = zeros(Float64, N)
    eigs[1:nF] .= 1.0

    # Randomly shuffle the eigenvalues
    #shuffle!(eigs)

    # Construct the density matrix
    Δ = U * Diagonal(eigs) * U'

    # Ensure the output is explicitly Hermitian
    return Hermitian(Δ)
end

