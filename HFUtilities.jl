
using LinearAlgebra

#diagonalize matrix utility
function extractMatrixOfEigenvectors(HHF)
    F = eigen(Hermitian(HHF))
    return F.vectors
end

#once you get the final delta matrix, extract the evectors with evalue 1 (i.e the filled states). These will be the highest eigenvalue eigenvectors.
function extractFilledOrbitals(nF,Δ)
    orbitals = extractMatrixOfEigenvectors(Δ)
    N = size(orbitals,1)
    filledOrbitals = orbitals[:,N-nF:N]
    return filledOrbitals
end

#can build a slater det explicitly if you want
function SlaterDetFromFilledOrbitals(filledOrbitals,positions,p)
    nF = size(filledOrbitals,1)
    slaterDetMatrix = zeros(nF,nF)
    for i in 1:nF
        for j in 1:nF
            slaterDetMatrix[i,j] = coeffVectorToRealSpace(filledOrbitals(i),positions(j))
        end
    end
    return det(slaterDetMatrix)
end

function calculateCorrelationFunction(filledOrbitals, p, q, levels, L, momentum, position)
    origin = (0.0, 0.0)
    result = 0.0 + 0.0im
    nOrbitals = size(filledOrbitals, 2)  # Number of columns
    for i in 1:nOrbitals
        orb = filledOrbitals[:, i]
        psi_origin = coeffVectorToRealSpace(orb, p, q, momentum, origin, L,levels)
        psi_pos = coeffVectorToRealSpace(orb, p, q, momentum, position, L,levels)
        result += conj(psi_origin) * psi_pos
    end
    return real(result)
end

function chargeDensityWaveOrderParameter(filledOrbitals,momentum,L,p,q,levels)
    function correlationFunction(position)
        return calculateCorrelationFunction(filledOrbitals,p,q,levels,L,momentum,position)
    end
    plotFourierTransformOfGroundStateCorrelationFunction(correlationFunction,p,L)
end

#calculate to see if you're in FM phase
function ferromagneticOrderParameter(nF,filledOrbitals,levels,p,q)
    basisSize = 4*levels*p
    orderParameter = 0
    for i in 1:nF
       for j in 1:basisSize
           if isOddMod4(j)
               orderParameter += abs2(filledOrbitals[j,i])*0.5
           else
               orderParameter -= abs2(filledOrbitals[j,i])*0.5
           end
       end
    end
    return orderParameter
end

#calculate to see if you're in superfluid phase (counts number of Cooper pairs essentially)
#function superfluidityOrderParameter(filledOrbitals,levels,p,q)
#    orderParameter = 0
#    basisSize = 4*levels*p
#    basisSizeUpToSpin = 2*levels*p
#    for i in 1:size(filledOrbitals)
#        for j in 1:basisSizeUpToSpin
#            orderParameter += filledOrbitals[i][2*j-1]*filledOrbitals[i][2*j]
#        end
#    end
#    return orderParameter
#end

#calculate to see if you're in AFM phase
function antiFerromagneticOrderParameter(nF,filledOrbitals,levels,p,q)
    basisSize = 4*levels*p
    orderParameter = 0
    for i in 1:nF
        for j in 1:basisSize
            S = iseven(i) ? -1 : 1
            spin = isOddMod4(i) ? -1 : 1
            orderParameter += S*spin*0.5*abs2(filledOrbitals[j,i])
        end
    end
    return orderParameter
end

#need to finish coding this. Will use to find peaks of FT; this will tell us if state is in CDW phase
function plotFourierTransformOfGroundStateCorrelationFunction(func, p, L; Nx=256, Ny=256)
    xmin, xmax = -pi/L, pi/L
    ymin, ymax = -pi/p*L, pi/p*L

    # Wrap position function for (x,y) input
    f_wrapped(x, y) = func((x, y))

    @info "Computing Fourier transform (this may take a while)..."
        F_shifted, fx, fy = compute_2d_fft(f_wrapped, xmin, xmax, ymin, ymax; Nx, Ny)

        plot_fft_modulus_pyplot(F_shifted, fx, fy)
end

    function DIISErrorCoeffs(errorvecs, maxErrors=8)
        m = min(length(errorvecs), maxErrors)
        m < 2 && return Float64[]  # Need at least 2 vectors

        # Take last m error vectors
        evecs = errorvecs[end-m+1:end]
        B = zeros(m+1, m+1)

        # Build DIIS matrix
        for i in 1:m
            for j in 1:m
                B[i, j] = real(tr(evecs[i] * evecs[j]))
            end
            B[i, m+1] = 1
            B[m+1, i] = 1
        end
        B[m+1, m+1] = 0

        # Solve constraint equation
        b = zeros(m+1)
        b[m+1] = 1

        # First try using regular matrix inverse
        c = B / b'
        return c[1:m]
    end

function extrapolateFockMatrixWithDIIS(errorCoeffs, previousDensityMatrices, nF)
    result = zero(first(previousDensityMatrices))
    for (c, Δ) in zip(errorCoeffs, previousDensityMatrices)
        result += c * Δ
    end

    # Purify to ensure idempotency
    F = eigen(result)
    U = F.vectors
    Λ = F.values
    # Sort descending, keep nF largest
    idx = sortperm(real.(Λ), rev=true)[1:nF]
    Δ_new = U[:, idx] * U[:, idx]'
    return Δ_new
end

#need to finish implementing Chern number order parameter; most of the coding is done just need to adapt it slightly for this usecase
function oldChernNumber(xstart,xend,ystart,yend,length,bandindex,levels,B,phi,L,harmonic)
    X = range(xstart,xend,length)
    Y = range(ystart,yend,length)
    chernNumber = 0

    for i in 1:length-1
        for j in 1:length-1
            u1 = oldLV(X[i],Y[j],X[i+1],Y[j],levels,harmonic,phi,B,L,bandindex)
            u2 = oldLV(X[i+1],Y[j],X[i+1],Y[j+1],levels,harmonic,phi,B,L,bandindex)
            u3 = oldLV(X[i],Y[j+1],X[i+1],Y[j+1],levels,harmonic,phi,B,L,bandindex)
            u4 = oldLV(X[i],Y[j],X[i],Y[j+1],levels,harmonic,phi,B,L,bandindex)

            plaquette = (u1*u2)/(u3*u4)
            phase = angle(plaquette)

            chernNumber += phase
        end
    end
    return chernNumber
end


function oldLV(i1,j1,i2,j2,levels,harmonic,phi,B,L,bandindex)
    u1 = eigenstate(i1,j1,levels,harmonic,phi,B,L,bandindex)
    u2 = eigenstate(i2,j2,levels,harmonic,phi,B,L,bandindex)

    if u1 == nothing || u2 == nothing
        return 0.0
    end

    overlap = dot(u1,u2)/abs(dot(u1,u2))

    return overlap
end

function hartreeFockChernNumber(xstart, xend, ystart, yend, length, levels, p, q, harmonics, bohrMagneton, charge, screening, cutoff, phi, nF, L; spin=false, valley=false, ϵ=1e-12)
    X = LinRange(xstart, xend, length)
    Y = LinRange(ystart, yend, length)
    Spin = spin
    Valley = valley
    @info "Running Chern number calculation for all filled bands..."

    # Precompute filled orbitals for all points
    filledBandsGrid = Array{Any}(undef, length, length)
    total_points = length * length
    progress_points = Progress(total_points, desc="Computing HF at k-points: ",barglyphs=BarGlyphs('|','█','▁','|',' '), output=stderr)

    for i in 1:length
        for j in 1:length
            kx = X[i]
            ky = Y[j]
            B=  (p/q)/(L^2)
            l_B = 1/(sqrt(B) + ϵ)

            if Valley
                H0 = spinfulValleyfulHamiltonian(kx, ky, levels, harmonics, phi, p, q, L, bohrMagneton)
                Vtensor = constructValleyfulSpinFullVtensor(kx, ky, p, q, L, cutoff, levels, screening, l_B, charge)
            elseif Spin
                H0 = spinfulHamiltonian(kx, ky, levels, harmonics, phi, p, q, L, bohrMagneton)
                Vtensor = constructSpinFullVtensor(kx, ky, p, q, L, cutoff, levels, screening, l_B, charge)
            else
                H0 = Hamiltonian(kx, ky, levels, harmonics, phi, p, q, L)
                Vtensor = constructFullVTensor(kx, ky, p, q, L, cutoff, levels, screening, l_B, charge)
            end

            Δ = hartree_fock_iteration(H0, Vtensor, nF; max_iter=200, tol=1e-6)
            filledBands = extractFilledOrbitals(nF, Δ)
            for band in 1:nF
                fix_phase!(@view filledBands[:, band])
            end
            filledBandsGrid[i,j] = filledBands
                next!(progress_points)
        end
    end

    # Compute Chern number for each band
    chern_numbers = zeros(nF)
    progress_bands = Progress(nF, desc="Computing Chern numbers: ",barglyphs=BarGlyphs('|','█','▁','|',' '),output=stderr)

    for bandindex in 1:nF
        chern = 0.0
        # Inside the bandindex loop:
        for i in 1:length-1
            for j in 1:length-1
                u1 = linkVariable(i, j, i+1, j, filledBandsGrid, bandindex)
                u2 = linkVariable(i+1, j, i+1, j+1, filledBandsGrid, bandindex)
                u3 = linkVariable(i+1, j+1, i, j+1, filledBandsGrid, bandindex)
                u4 = linkVariable(i, j+1, i, j, filledBandsGrid, bandindex)

                plaquette = u1 * u2*u3 * u4
                phase = angle(plaquette)
                chern += phase
            end
        end
        chern_numbers[bandindex] = chern/(2*pi)
        next!(progress_bands)
    end

    return chern_numbers
end

    function hartreeFockCompositeChernNumber(xstart, xend, ystart, yend, length, levels, p, q, harmonics, bohrMagneton, charge, screening, cutoff, phi, nF, L,orbitalsGroups; spin=false, valley=false, ϵ=1e-12)
        X = LinRange(xstart, xend, length)
        Y = LinRange(ystart, yend, length)
        Spin = spin
        Valley = valley
        @info "Running Chern number calculation for all filled band groups..."

            # Precompute filled orbitals for all points
            filledBandsGrid = Array{Any}(undef, length, length)
            total_points = length * length
            progress_points = Progress(total_points, desc="Computing HF at k-points: ",barglyphs=BarGlyphs('|','█','▁','|',' '), output=stderr)

            for i in 1:length
                for j in 1:length
                    kx = X[i]
                    ky = Y[j]
                    B=  (p/q)/(L^2)
                    l_B = 1/(sqrt(B) + ϵ)

                    if Valley
                        H0 = spinfulValleyfulHamiltonian(kx, ky, levels, harmonics, phi, p, q, L, bohrMagneton)
                        Vtensor = constructValleyfulSpinFullVtensor(kx, ky, p, q, L, cutoff, levels, screening, l_B, charge)
                        elseif Spin
                        H0 = spinfulHamiltonian(kx, ky, levels, harmonics, phi, p, q, L, bohrMagneton)
                        Vtensor = constructSpinFullVtensor(kx, ky, p, q, L, cutoff, levels, screening, l_B, charge)
                    else
                        H0 = Hamiltonian(kx, ky, levels, harmonics, phi, p, q, L)
                        Vtensor = constructFullVTensor(kx, ky, p, q, L, cutoff, levels, screening, l_B, charge)
                    end

                    Δ = hartree_fock_iteration(H0, Vtensor, nF; max_iter=200, tol=1e-6)
                    filledBands = extractFilledOrbitals(nF, Δ)
                    for band in 1:nF
                        fix_phase!(@view filledBands[:, band])
                    end
                    filledBandsGrid[i,j] = filledBands
                    next!(progress_points)
                end
            end
            numberOfGroups = size(orbitalsGroups,1)
            # Compute Chern number for each band
            chern_numbers = zeros(numberOfGroups)
            progress_bands = Progress(numberOfGroups, desc="Computing Chern numbers: ",barglyphs=BarGlyphs('|','█','▁','|',' '),output=stderr)

            for group in 1:numberOfGroups
                chern = 0.0
                # Inside the bandindex loop:
                for i in 1:length-1
                    for j in 1:length-1
                        u1 = matrixLinkVariable(i, j, i+1, j, filledBandsGrid, group)
                        u2 = matrixLinkVariable(i+1, j, i+1, j+1, filledBandsGrid, group)
                        u3 = matrixLinkVariable(i+1, j+1, i, j+1, filledBandsGrid, group)
                        u4 = matrixLinkVariable(i, j+1, i, j, filledBandsGrid, group)

                        plaquette = u1 * u2*u3 * u4
                        phase = angle(det(plaquette))
                        chern += phase
                    end
                end
                chern_numbers[group] = chern/(2*pi)
                next!(progress_bands)
            end

            return chern_numbers
        end

function linkVariable(i1, j1, i2, j2, filledBandsGrid, bandindex)
    u1 = filledBandsGrid[i1, j1][:, bandindex]
    u2 = filledBandsGrid[i2, j2][:, bandindex]

    overlap = dot(u1, u2)
    abs_overlap = abs(overlap)
    if abs_overlap < 1e-12
        @warn "Small overlap detected (|⟨u₁|u₂⟩| = $abs_overlap) at band $bandindex between k-points ($i1,$j1) and ($i2,$j2)"
        print(norm(u1),norm(u2))
        #return 1.0 + 0.0im  # Default to unit phase
    end

    return overlap / abs_overlap
end

function matrixLinkVariable(i1,j1,i2,j2,filledBandsGrid,orbitalsGroup)
    groupSize = length(orbitalsGroup)
    linkVariable = zeros(groupSize,groupSize)
    for l in 1:groupSize
        for k in 1:groupSize
            bandindex1 = orbitalsGroup[l]
            bandindex2 = orbitalsGroup[k]
            u1 = filledBandsGrid[i1, j1][:, bandindex1]
            u2 = filledBandsGrid[i2, j2][:, bandindex2]
            overlap = dot(u1, u2)
            abs_overlap = abs(overlap)
            linkVariable[l,k] = overlap / abs_overlap
        end
    end
    matrixNorm = norm(linkVariable)
    if matrixNorm < 1e-12
        @warn "Small overlap  ($abs_overlap) detected at ($i1,$j1) and ($i2,$j2)"
        print(norm(u1),norm(u2))
        #return 1.0 + 0.0im  # Default to unit phase
    end
    return linkVariable

end





