include("Utilities.jl")
include("matrixElements.jl")
include("LandauLevels.jl")
include("solveNonInteractingProblem.jl")
include("hartreeFock.jl")

function main()
    # Physical parameters
    k_x, k_y = 0.0, 0.0
    mnrange = 2
    levels = 5
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
    length = 100
    ky_range = LinRange(-p*pi/(q*L),p*pi/(q*L),length)
    fourier_coeffs = [1]
    k_x_fixed = -p*pi/(q*L)
    #kx_range
    plot_spectrum_along_ky(k_x_fixed, ky_range, levels, fourier_coeffs, phi, p, q, L)

    #plotBandEnergies(k_x, k_y, levels, fourier_coeffs, phi, L, mnrange)
end

main()
#test_kx_periodicity()
