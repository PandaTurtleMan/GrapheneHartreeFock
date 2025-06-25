using LinearAlgebra, SpecialFunctions,ProgressMeter
include("Utilities.jl")
include("matrixElements.jl")
include("LandauLevels.jl")
include("solveNonInteractingProblem.jl")
include("hartreeFock.jl")
include("HFUtilities.jl")


function main()
    # Physical parameters
    k_x, k_y = 0.1, 0.5
    levels = 3
    p, q = 5,37
    L = 1.0
    nF = 130
    B = p/(q*L^2)
    phi = 0
    l_B = 1/sqrt(B)
    fourier_coeffs = [1]
    #tunes strength of coloumb interaction, effectively changes dielectric constant
    charge = 1
    screening(u) = tanh(u) #screening
    cutoff = 10.0  # Integration cutoff
    bohrMagneton = 1 #when using the spinful versions, can use this to tune the Zeeman shifts
    U = 0 #also for spinful versions, can use this to tune the Hubbard strength

    # Build non-interacting Hamiltonian (external potential)
    @info "Building non-interacting Hamiltonian..."
    H0 = spinfulValleyfulHamiltonian(k_x, k_y, levels, fourier_coeffs, phi, p, q, L,bohrMagneton)
    @info "Constructing interaction matrix elements..."
    Vtensor = constructValleyfulSpinFullVtensor(k_x,k_y,p,q,L,cutoff,levels,screening,l_B,charge)
    # Run Hartree-Fock calculation
    @info "Starting Hartree-Fock iteration..."
    Δ = hartree_fock_iteration(
        H0,Vtensor,nF;
        max_iter=10000, tol=1e-6
    )

    # Compute final energy
    energy = compute_final_energy(
        H0, Δ, k_x, k_y, levels, L, l_B, screening, cutoff,Vtensor
    )
    println("\nFinal Hartree-Fock energy: $energy")

    @info "Computing order parameters..."

    filledOrbitals = extractFilledOrbitals(nF,Δ)

    @info "Computing ferromagnetic order parameter..."
    FMOrderParameter = ferromagneticOrderParameter(nF,filledOrbitals,levels,p,q)
    println("\nFerromagnetic order parameter: : $FMOrderParameter")

    @info "Computing anti-ferromagnetic order parameter..."
    AFMOrderParameter = antiFerromagneticOrderParameter(nF,filledOrbitals,levels,p,q)
    println("\nAnti-ferromagnetic order parameter: : $AFMOrderParameter")

    #@info "Computing superfluidity order parameter..."
    #superfluidityOrderParameter = superfluidityOrderParameter(filledOrbitals,levels,p,q)
    #println("\nSuperfluidity order parameter: : $superfluidityOrderParameter")

    @info "Computing charge density order parameter..."
    #chargeDensityWaveOrderParameter(filledOrbitals,(k_x,k_y),L,p,q,levels)
    @info "Computing chern number..."
    xstart = -pi/L
    xend = pi/L
    ystart = -p*pi/(q*L)
    yend = p*pi/(q*L)
    length = 10
    @info "Computing Chern numbers with Landau level grouping..."

    # Group bands by Landau level index
    groups = groupBandsByLandauLevel(levels=2, p=2, nF=24, spin=true, valley=true)

    # Compute composite Chern numbers
    composite_cherns = hartreeFockCompositeChernNumber(
        xstart, xend, ystart, yend,
        length, levels, p, q, fourier_coeffs,
        bohrMagneton, charge, screening, cutoff, phi, nF, L,
        groups;  # Pass Landau level groups
        spin=true, valley=true
        )

    # Print results
    for (group_idx, chern) in enumerate(composite_cherns)
        bands_in_group = groups[group_idx]
        ll_index = getLandauLevelIndex(bands_in_group[1], levels, p, spin=true, valley=true)
        println("Composite Chern for LL $ll_index (bands $bands_in_group): $(round(chern, digits=3))")
        end

        # Ensure zeroth Landau level is fully filled
        zero_ll_group = findfirst(group -> getLandauLevelIndex(group[1], levels, p, spin=true, valley=true) == 0, orbitalsGroups)
        if zero_ll_group !== nothing
            zero_ll_bands = orbitalsGroups[zero_ll_group]
            println("\nZeroth Landau level bands (fully filled): $zero_ll_bands")
        end
    end
main()
