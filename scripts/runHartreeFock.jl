using LinearAlgebra, SpecialFunctions, ProgressMeter, JLD2
include("../src/Utilities.jl")
include("../src/matrixElements.jl")
include("../src/LandauLevels.jl")
include("../src/solveNonInteractingProblem.jl")
include("../src/hartreeFock.jl")
include("../src/HFUtilities.jl")

function run_hartree_fock_with_multiple_initial_states()
    # System parameters (example values - adjust as needed)
    levels = 2
    p = 3
    q = 5
    L = 10.0
    nF = 4
    ε = 2.0
    harmonicRange = 2
    kxRadius = 3
    kyRadius = 3
    screening_fn = x -> tanh(x)
    harmonics = [(1.0, (1,0)), (0.5, (1,1))]  # Example harmonics

    # Create indices and grids
    N_kx, N_ky = 2*kxRadius, 2*kyRadius
    i_n = Index(2*levels+1, "n")
    i_s = Index(2, "s")
    i_K = Index(2, "K")
    i_l = Index(p, "l")
    orbital_indices = (i_n, i_s, i_K, i_l)

    ikx = Index(N_kx, "kx")
    iky = Index(N_ky, "ky")
    momentum_indices = (ikx, iky)

    k_step_x = 2π/(L * N_kx)
    k_step_y = 2π/(q * L * N_ky)
    k_grid_vals_x = [n * k_step_x for n in -kxRadius:(kxRadius-1)]
        k_grid_vals_y = [n * k_step_y for n in -kyRadius:(kyRadius-1)]

            # Generate initial density matrices
            initial_density_matrices = []

            # 1. Three random density matrices
            for i in 1:3
                Δ_random = build_random_density_matrix(levels, p, nF, orbital_indices, momentum_indices,
                                                       k_grid_vals_x, k_grid_vals_y, seed=i)
                push!(initial_density_matrices, ("random_$i", Δ_random))
            end

            # 2. Non-interacting ground state
            Δ_non_interacting = build_noninteracting_density_matrix(levels, harmonics, p, q, L, nF,
                                                                    orbital_indices, momentum_indices,
                                                                    k_grid_vals_x, k_grid_vals_y)
            push!(initial_density_matrices, ("non_interacting", Δ_non_interacting))

            # 3. Sz Zeeman ground state
            Δ_sz = build_zeeman_density_matrix(levels, harmonics, p, q, L, nF, orbital_indices,
                                               momentum_indices, k_grid_vals_x, k_grid_vals_y, 0.5)
            push!(initial_density_matrices, ("sz_zeeman", Δ_sz))

            # 4. Valley Zeeman ground state
            Δ_valley = build_valley_zeeman_density_matrix(levels, harmonics, p, q, L, nF, orbital_indices,
                                                          momentum_indices, k_grid_vals_x, k_grid_vals_y, 0.5)
            push!(initial_density_matrices, ("valley_zeeman", Δ_valley))

            # 5. Mass term ground state
            Δ_mass = build_mass_term_density_matrix(levels, harmonics, p, q, L, nF, orbital_indices,
                                                    momentum_indices, k_grid_vals_x, k_grid_vals_y, 0.1)
            push!(initial_density_matrices, ("mass_term", Δ_mass))

            # Define Q points: Gamma point and two random points
            Q_points = [(div(N_kx,2)+1, div(N_ky,2)+1)]  # Gamma point (middle of grid)
            for _ in 1:2
                push!(Q_points, (rand(1:N_kx), rand(1:N_ky)))
            end

            # Run Hartree-Fock for each Q point and collect results
            results = []
            for (i, Q_val) in enumerate(Q_points)
                println("Running Hartree-Fock for Q point $i: $Q_val")

                    filename = "hartree_fock_result_Q$i.jld2"
                    min_Δ, min_energy, min_order_params = run_hartree_fock_tensor(
                        filename, initial_density_matrices, levels, p, q, L, nF, ε,
                        harmonicRange, kxRadius, kyRadius, screening_fn, Q_val, harmonics
                        )

                    push!(results, (Q_val, min_energy, min_Δ, min_order_params))
                end

                # Find the lowest energy result
                min_energy = Inf
                best_result = nothing
                for (Q_val, energy, Δ, order_params) in results
                    if energy < min_energy
                        min_energy = energy
                        best_result = (Q_val, energy, Δ, order_params)
                    end
                end

                # Print the best result
                Q_val, energy, Δ, order_params = best_result
                println("\nLOWEST ENERGY RESULT ACROSS ALL Q POINTS")
                println("========================================")
                println("Q point: $Q_val")
                println("Energy: $energy")
                println("Order parameters:")
                for (key, value) in order_params
                    println("  $key: $value")
                end
                println("========================================\n")

                # Save the best result
                @save "best_hartree_fock_result.jld2" Δ energy order_params Q_val orbital_indices momentum_indices k_grid_vals_x k_grid_vals_y

                return best_result
            end

best_result = run_hartree_fock_with_multiple_initial_states()
