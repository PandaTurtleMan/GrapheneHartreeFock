include("Utilities.jl")
include("matrixElements.jl")
include("LandauLevels.jl")
include("solveNonInteractingProblem.jl")
include("hartreeFock.jl")

function main()
    # Physical parameters
    k_x, k_y = 0.0, 0.0
    mnrange = 2
    levels = 1
    #phi = 0
    #p, q = 15,37
    #L = 1
    #nF = 8
    #B = p/(q*L^2)
    #l_B = 1/sqrt(B)
    #fourier_coeffs = [1]
    #screening(u) = tanh(u) #screening
    #cutoff = 10.0  # Integration cutoff
    #bohrMagneton = 0

    # Build non-interacting Hamiltonian (external potential)
    #@info "Plotting energy as a function of magnetic field"
    #Initialize CUDA and run
    #CUDA.allowscalar(false)
    #plotBandEnergies(k_x, k_y, levels, fourier_coeffs, phi, L, 2)
    # Set parameters
    #levels = 2
    p, q = 10,11
    L = 1.0
    phi = 0
    zeemanVector = [0,0,0]
    valleyZeemanVector = [0,0,0]
    m = 0


    plotBandEnergies(k_x, k_y, levels, fourier_coeffs, phi, L, zeemanVector, valleyZeemanVector, mnrange)
end

main()
#test_kx_periodicity()
