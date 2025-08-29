include("../src/nonInteractingChern.jl")

function main()
    # Physical parameters
    levels = 5
    p, q = 10, 11
    L = 1.0
    nF = 1500
    phi = 0
    fourier_coeffs = [1]
    bohrMagneton = 0.0
    spin = true
    valley = true

    # BZ grid
    xstart = -p*π/(q*L)
    xend = p*π/(q*L)
    ystart = -p*π/(q*L)
    yend = p*π/(q*L)
    grid_size = 100  # Reduced for testing

    # Group bands
    orbitalsGroups = groupBandsByLandauLevel(
        levels=levels, p=p, nF=nF,
        spin=spin, valley=valley
        )

    # Build Hamiltonian function with captured parameters
    build_H(kx, ky) = build_hamiltonian(kx, ky;
                                        levels=levels,
                                        harmonics=fourier_coeffs,
                                        phi=phi,
                                        p=p,
                                        q=q,
                                        L=L,
                                        bohrMagneton=bohrMagneton,
                                        valley=valley
                                        )

    # Compute Chern numbers
    cherns = nonInteractingCompositeChernNumber(
        build_H,
        xstart, xend, ystart, yend,
        grid_size, orbitalsGroups
    )

    # Print results
    println("\nComposite Chern Numbers:")
    for (i, chern) in enumerate(cherns)
        bands = orbitalsGroups[i]
        println("Group $i (bands $(first(bands))-$(last(bands))): $(round(chern, digits=3))")
    end
end

# Add parallel processing setup
if Threads.nthreads() == 1
    @info "Running single-threaded. Start Julia with multiple threads (e.g., julia -t auto) for better performance."
else
    @info "Using $(Threads.nthreads()) threads for parallel computation"
end

main()