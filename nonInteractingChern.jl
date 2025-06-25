# nonInteractingCompositeChern.jl
using LinearAlgebra, ProgressMeter
include("Utilities.jl")
include("matrixElements.jl")
include("LandauLevels.jl")
include("solveNonInteractingProblem.jl")

function compositeLinkVariable(filledBandsGrid, k1, k2, orbitalsGroup)
    states_k1 = filledBandsGrid[k1...][:, orbitalsGroup]
    states_k2 = filledBandsGrid[k2...][:, orbitalsGroup]
    M = states_k1' * states_k2  # Overlap matrix
    #matrixSize
    #if norm(M) < sqrt(50)
    #    print("uh oh,$k1 and $k2 did a booboo for group $orbitalsGroup[1]")
    #end
    return det(M)/abs(det(M))
end

function nonInteractingCompositeChernNumber(
    xstart, xend, ystart, yend,
    grid_size, levels, p, q, harmonics,
    bohrMagneton, phi, nF, L, orbitalsGroups;
    spin=false, valley=false, ϵ=1e-12
    )
    X = LinRange(xstart, xend, grid_size)
    Y = LinRange(ystart, yend, grid_size)
    @info "Computing composite Chern numbers for non-interacting bands..."

        # Precompute eigenvectors for all k-points
        eigenvecsGrid = Array{Any}(undef, grid_size, grid_size)
        total_points = grid_size * grid_size
        progress = Progress(total_points, desc="Diagonalizing at k-points: ",
                            barglyphs=BarGlyphs('|','█','▁','|',' '), output=stderr)

        for i in 1:grid_size
            for j in 1:grid_size
                kx = X[i]
                ky = Y[j]
                B = p/(q*L^2)
                l_B = 1/(sqrt(B) + ϵ)

                # Build Hamiltonian
                if valley
                    H = spinfulValleyfulHamiltonian(kx, ky, levels, harmonics, phi, p, q, L, bohrMagneton)
                elseif spin
                    H = spinfulHamiltonian(kx, ky, levels, harmonics, phi, p, q, L, bohrMagneton)
                else
                    H = Hamiltonian(kx, ky, levels, harmonics, phi, p, q, L)
                end

                # Diagonalize and store all eigenvectors
                F = eigen(Hermitian(H))
                eigenvecsGrid[i, j] = F.vectors
                next!(progress)

            end
            GC.gc()
        end
        numberOfGroups = length(orbitalsGroups)
        chern_numbers = zeros(numberOfGroups)
        progress_bands = Progress(numberOfGroups, desc="Computing Chern numbers: ", barglyphs=BarGlyphs('|','█','▁','|',' '), output=stderr)

        for group_idx in 1:numberOfGroups
            chern = 0.0
            group = orbitalsGroups[group_idx]

            for i in 1:(grid_size-1)
                for j in 1:(grid_size -1)
                    # Define plaquette corners (CCW order)
                    k1 = (i, j)
                    k2 = (i+1, j)
                    k3 = (i+1, j+1)
                    k4 = (i, j+1)

                    # Compute link products
                    U12 = compositeLinkVariable(eigenvecsGrid, k1, k2, group)
                    U23 = compositeLinkVariable(eigenvecsGrid, k2, k3, group)
                    U34 = compositeLinkVariable(eigenvecsGrid, k4, k3, group)^(-1)
                    U41 = compositeLinkVariable(eigenvecsGrid, k1, k4, group)^(-1)

                    # Calculate Berry phase for plaquette
                    product = U12 * U23 * U34 * U41
                    phase = angle(product)  # -π to π
                    chern += phase
                end
            end

            chern_numbers[group_idx] = round(chern / (2π), digits=5)
            next!(progress_bands)
        end

        return chern_numbers
    end

    function groupBandsByLandauLevel(; levels, p, nF, spin, valley)
        # Calculate degeneracies
        if valley
            degeneracy_zeroth = 4 * p
            degeneracy_per_level = 4 * p  # For each n>0 (both branches)
            elseif spin
            degeneracy_zeroth = 2 * p
            degeneracy_per_level = 2 * p
        else
            degeneracy_zeroth = p
            degeneracy_per_level = p
        end

        groups = Vector{Vector{Int}}()
        total_states = valley ? (8 * levels * p + 4 * p) :
            spin ? (4 * levels * p) : (2 * levels * p)
        nF = min(nF, total_states)
        band_counter = 1

        # Negative energy levels (n = levels to 1, descending)
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
            band_counter + degeneracy_per_level > nF && break
            group = band_counter:(band_counter + degeneracy_per_level - 1)
            push!(groups, collect(group))
            band_counter += degeneracy_per_level
        end

        return groups
    end

    function main()
        # Physical parameters
        levels= 5
        p, q = 30, 239
        L = 1.0
        nF = 1420
        phi = 0
        fourier_coeffs = [1]  # Fourier coefficients
        bohrMagneton = 0.0

        # Brillouin zone grid
        xstart = -π/L
        xend = π/L
        ystart = -p*π/(q*L)
        yend = p*π/(q*L)
        grid_size = 10

        #Group bands by Landau level
        orbitalsGroups = groupBandsByLandauLevel(
            levels=levels, p=p, nF=nF,
            spin=true, valley=true
            )
        #orbitalsGroups = [1:271]

        # Compute composite Chern numbers
        cherns = nonInteractingCompositeChernNumber(
            xstart, xend, ystart, yend,
            grid_size, levels, p, q, fourier_coeffs,
            bohrMagneton, phi, nF, L, orbitalsGroups;
            spin=true, valley=true
            )

        # Print results
        println("\nComposite Chern Numbers:")
        for (i, chern) in enumerate(cherns)
            bands = orbitalsGroups[i]
            println("Group $i (bands $(first(bands))-$(last(bands)): $(round(chern, digits=3))")
        end
    end

    main()
