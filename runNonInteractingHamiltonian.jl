include("Utilities.jl")
include("matrixElements.jl")
include("LandauLevels.jl")
include("solveNonInteractingProblem.jl")
include("hartreeFock.jl")

function main()
    # Physical parameters
    k_x, k_y = 0.0, 0.0
    levels = 3

    m = 0
    phi = 0
    L = 10.0
    zeemanVector = [0,0,0]
    valleyZeemanVector = [0,0,0]

    # Define harmonics as (coefficient, base_vector) tuples
    harmonics = [
        (0.03, (1, 0)),   # First equivalence class
        (0.0, (1, 1))    # Second equivalence class
        ]

    plotBandEnergies(k_x, k_y, levels, harmonics, phi, L, m,
                     zeemanVector, valleyZeemanVector)
end

main()
