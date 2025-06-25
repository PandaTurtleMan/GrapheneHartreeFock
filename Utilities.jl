using SpecialFunctions
using SpecialPolynomials
using Polynomials
using Combinatorics
using FFTW
using PyPlot
using ProgressMeter

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

function laguerreTilde(n,m,x)
    return sqrt(factorial(n)/factorial(n+m)*x^m *exp(-x))*laguerre_poly(m,n,x)
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
        # Find index of element with largest magnitude
        idx = argmax(abs.(v))
        c = v[idx]
        if abs(c) < 1e-12
            return v  # Avoid division by zero
        end
        phase_factor = conj(c) / abs(c)
        v .= v .* phase_factor
        return v
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


