# nonInteractingCompositeChern.jl
using LinearAlgebra, ProgressMeter
include("Utilities.jl")
include("matrixElements.jl")
include("LandauLevels.jl")
include("solveNonInteractingProblem.jl")

function compositeLinkVariable(i,j,states1, states2, group)
    M = states1' * states2
    detM = det(M)
    if abs(detM) < 1e-8
        @warn "Small overlap determinant: |det| = $(abs(detM)) for group $group at point k_x = $i, k_y = $j"
    end

    return detM / abs(detM)
end

function compute_kx_column(i, X, Y, build_H, prev_col, grid_size; periodic_ref=nothing)
    col = Vector{Matrix{ComplexF64}}(undef, grid_size)
    kx = X[i]

    # Compute first point in column
    ky = Y[1]
    H = build_H(kx, ky)
    eig = eigen(Hermitian(H))
    vecs = eig.vectors

    if !isnothing(periodic_ref)
        # Force gauge consistency with periodic_ref (left column)
        vecs = fix_gauge!(vecs, periodic_ref[1])
        elseif isnothing(prev_col)
        fix_phase!(vecs)
    else
        vecs = fix_gauge!(vecs, prev_col[1])
    end
    col[1] = vecs

    # Compute remaining points in column
    for j in 2:grid_size
        ky = Y[j]
        H = build_H(kx, ky)
        eig = eigen(Hermitian(H))
        vecs = eig.vectors
        vecs = fix_gauge!(vecs, col[j-1])  # Fix gauge relative to previous point in same column
        col[j] = vecs
    end
    return col
end

function nonInteractingCompositeChernNumber(
    build_H::Function,
    xstart, xend, ystart, yend,
    grid_size, orbitalsGroups;
    ϵ=1e-12
)
    @info "Starting column-based Chern number calculation..."
    X = LinRange(xstart, xend, grid_size)
    Y = LinRange(ystart, yend, grid_size)
    chern_phases = zeros(length(orbitalsGroups))

    # Compute and store col0 (first column)
    @info "Computing first column..."
    col0 = compute_kx_column(1, X, Y, build_H, nothing, grid_size)
    col_left = col0

    # Compute next column if needed
    if grid_size > 1
        @info "Computing second column..."
        col_right = compute_kx_column(2, X, Y, build_H, col_left, grid_size)
    else
        col_right = col0
    end

    progress = Progress(grid_size, desc="Processing columns: ",
                        barglyphs=BarGlyphs('|','█','▁','|',' '),
                        output=stderr)

    nthreads = Threads.nthreads()

    for i in 1:grid_size
        i_next = i == grid_size ? 1 : i+1

        # For last column, use col0 as right column
        if i == grid_size
            col_right = col0
        end

        # Allocate thread-local storage
        thread_phases = [zeros(length(orbitalsGroups)) for _ in 1:nthreads]

        # Process current column pair
        Threads.@threads for j in 1:grid_size
            tid = Threads.threadid()
            j_next = j == grid_size ? 1 : j+1

            for (group_idx, group) in enumerate(orbitalsGroups)
                # Get states for the plaquette
                statesA = @view col_left[j][:, group]       # (i, j)
                statesB = @view col_left[j_next][:, group]   # (i, j_next)
                statesC = @view col_right[j_next][:, group]  # (i_next, j_next)
                statesD = @view col_right[j][:, group]       # (i_next, j)

                # Calculate link variables
                U12 = compositeLinkVariable(i, j, statesA, statesB, group)
                U23 = compositeLinkVariable(i, j, statesB, statesC, group)
                U34 = compositeLinkVariable(i, j, statesC, statesD, group)
                U41 = compositeLinkVariable(i, j, statesD, statesA, group)

                # Calculate Berry phase
                product = U12 * U23 * U34 * U41
                phase = angle(product)
                thread_phases[tid][group_idx] += phase
            end
        end

        # Accumulate thread results
        for phases in thread_phases
            chern_phases .+= phases
        end

        # Prepare for next iteration
        if i < grid_size
            col_left = col_right
            if i < grid_size - 1
                col_right = compute_kx_column(i + 2, X, Y, build_H, col_left, grid_size)
            end
        end

        next!(progress)
    end

    chern_numbers = round.(chern_phases / (2π), digits=5)
    return chern_numbers
end

function build_hamiltonian(kx, ky; levels, harmonics, phi, p, q, L, bohrMagneton, valley)
    if valley
        return spinfulValleyfulHamiltonian(kx, ky, levels, harmonics, phi, p, q, L, bohrMagneton)
    else
        return Hamiltonian(kx, ky, levels, harmonics, phi, p, q, L) # TODO: fix call signature with zeeman terms
    end
end

function groupBandsByLandauLevel(; levels, p, nF, spin, valley)
    @info "Calculating groups"
    # Calculate degeneracies
    if valley
        degeneracy_zeroth = 4 * p
        degeneracy_per_level = 4 * p  # For each n>0 (both branches)
    elseif spin
        degeneracy_zeroth = 4 * p
        degeneracy_per_level = 2 * p
    else
        degeneracy_zeroth = 2* p
        degeneracy_per_level = p
    end

    groups = Vector{Vector{Int}}()
    total_states = valley ? (8 * levels * p + 4 * p) :
        spin ? (4 * levels * p) : (2 * levels * p)
    nF = min(nF, total_states)
    band_counter = 1

    # Negative energy levels (n = levels to 1, descending)
    levels = valley ? levels : levels -1
    for n in levels:-1:1
        band_counter + degeneracy_per_level > nF && break
        group = band_counter:(band_counter + degeneracy_per_level - 1)
        push!(groups, collect(group))
        band_counter += degeneracy_per_level
    end

    # Zeroth Landau level
    if band_counter <= nF
        group_size = min(degeneracy_zeroth, nF - band_counter + 1)
        group = band_counter:(band_counter + group_size - 1)
        push!(groups, collect(group))
        band_counter += group_size
    end

    # Positive energy levels (n = 1 to levels, ascending)
    for n in 1:levels
        band_counter + degeneracy_per_level > nF+1 && break
        group = band_counter:(band_counter + degeneracy_per_level - 1)
        push!(groups, collect(group))
        band_counter += degeneracy_per_level
    end
    @info "Grouping complete!"
    return groups
end
