
using LinearAlgebra
using Random

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
        # DIIS requires at least two error vectors to be meaningful.
        m < 2 && return Float64[]

        # Take the last m error vectors from the history.
        evecs = errorvecs[end-m+1:end]

        # Build the (m+1)x(m+1) DIIS matrix.
        B = zeros(m + 1, m + 1)
        for i in 1:m
            for j in i:m # Only compute upper triangle, since B is symmetric
                dot_product = real(tr(evecs[i]' * evecs[j]))
                B[i, j] = dot_product
                B[j, i] = dot_product
            end
        end

        # Add constraints for the Lagrange multiplier.
        B[1:m, m + 1] .= 1.0
        B[m + 1, 1:m] .= 1.0
        # B[m+1, m+1] is already 0.0

        # Define the right-hand side of the equation B*c = b
        b = zeros(m + 1)
        b[m + 1] = 1.0

        # --- FIXES ---
        # 1. Solve B*c = b using the pseudoinverse for numerical stability.
        # 2. This replaces the incorrect "c = B / b'"
        c = pinv(B) * b

        # Return the first m coefficients, excluding the Lagrange multiplier.
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

"""
Pre-computes the form-factor tensor S(q).
"""
function precomputeFormFactorTensorExchangeTerm(indices, qx_grid, qy_grid, xreciVectors,yreciVectors,L, l_B)
    n, λ = indices["ll"], indices["sublattice"]
    iqx = Index(length(qx_grid), "qx_momentum")
    iqy = Index(length(qy_grid), "qy_momentum")
    xGVectors = Index(length(xreciVectors),"xreciVector")
    yGVectors = Index(length(yreciVectors),"yreciVector")
    K = 2*pi/L
    S = ITensor(n', λ', n, λ, iqx,iqy,xGVectors,yGvectors)

    @showprogress "Pre-computing Form Factor S(q)..." for qx_idx in 1:dim(iqx),qy_idx in 1:dim(iqy), Gx_idx in 1:dim(xGVectors),Gy_idx in 1:dim(yGVectors)
        qx = qx_grid[qx_idx]
        qy = qy_grid[qy_idx]
        Gx = xreciVectors[Gx_idx]
        Gy = yreciVectors[Gy_idx]
        for n1_idx in 1:dim(n), n2_idx in 1:dim(n), λ1_idx in 1:dim(λ), λ2_idx in 1:dim(λ)
            n1, n2 = n1_idx - 1, n2_idx - 1
            λ1, λ2 = (λ1_idx == 1) ? 1 : -1, (λ2_idx == 1) ? 1 : -1

            val = grapheneLandauFourierMatrixElement(qx+K*Gx, qy+Gy*K, n1, n2, l_B, λ1, λ2)
            S[n'(n1_idx), λ'(λ1_idx), n(n2_idx), λ(λ2_idx), iq(q_idx),Gvectors(G_idx)] = val
        end
    end
    return S
end

function precomputeFormFactorTensorDirectTerm(indices, xreciVectors,yreciVectors, L,l_B,Q)
    n, λ = indices["ll"], indices["sublattice"]
    K = 2*pi/L
    xGVectors = Index(length(xreciVectors),"xreciVector")
    yGVectors = Index(length(yreciVectors),"yreciVector")
    S = ITensor(n', λ', n, λ, xGVectors,yGVectors)

    @showprogress "Pre-computing Form Factor S(q)..." for Gx_idx in 1:dim(xGVectors),Gy_idx in 1:dim(yGVectors)

        Gx = xreciVectors[Gx_idx]
        Gy = yreciVectors[Gy_idx]
        for n1_idx in 1:dim(n), n2_idx in 1:dim(n), λ1_idx in 1:dim(λ), λ2_idx in 1:dim(λ)
            n1, n2 = n1_idx - 1, n2_idx - 1
            λ1, λ2 = (λ1_idx == 1) ? 1 : -1, (λ2_idx == 1) ? 1 : -1

            val = grapheneLandauFourierMatrixElement(qx+K*Gx, qy+K*Gy, n1, n2, l_B, λ1, λ2)
            S[n'(n1_idx), λ'(λ1_idx), n(n2_idx), λ(λ2_idx), xGvectors(Gx_idx),yGvectors(Gy_idx)] = val
        end
    end
    return S
end

#fix this and add other tensors! make sure to refactor code to include ep parameter

function precomputePotentialTensorExchangeTerm(qx_grid,qy_grid,xreciVectors,yreciVectors,screeningFn,ε,L,Q_x,Q_y)
    iqx = Index(length(qx_grid), "qxMomentum")
    iqy = Index(length(qy_grid), "qyMomentum")
    xGVectors = Index(length(reciVectors),"xreciVector")
    yGVectors = Index(length(reciVectors),"yreciVector")
    K = 2*pi/L
    V = ITensor(iqx,iqy,xGVectors,yGVectors)
    @showprogress "Pre-computing Potential V(q)..." for qx_idx in 1:dim(iqx),qy_idx in 1:dim(iqy), Gx_idx in 1:dim(xGVectors),Gy_idx in 1:dim(yGVectors)
        qx = qx_grid[qx_idx]
        qy = qy_grid[qy_idx]
        Gx = xreciVectors[Gx_idx]
        Gy = yreciVectors[Gy_idx]
        # Example: Coulomb potential 1/|q|
        q_norm = sqrt((qx+K*Gx)^2 + (qy+K*Gy)^2)
        V[iq(q_idx),GVectors(G_idx)] =  (2 * pi / q_norm)*screeningFn(q_x+Gx*K,q_y+Gy*K)/ε
    end
    return V
end

function precomputePotentialTensorDirectTerm(xreciVectors,yreciVectors,screeningFn,ε,L,Q_x,Q_y)

    xGVectors = Index(length(reciVectors),"xreciVector")
    yGVectors = Index(length(reciVectors),"yreciVector")
    K = 2*pi/L
    V = ITensor(xGVectors,yGvectors)
    @showprogress "Pre-computing Potential V(q)..." for Gx_idx in 1:dim(xGVectors),Gy_idx in 1:dim(yGVectors)

        Gx = xreciVectors[Gx_idx]
        Gy = yreciVectors[Gy_idx]


        q_norm = sqrt((Q_x+Gx)^2 + (Q_y+Gy)^2)
        V[GVectors(G_idx)] =  (2 * pi / q_norm)*screeningFn(Gx*K+Q_x,Gy*K+Q_y)/ε
    end
    return V
end

function exchangeTermPhase1(qx_grid,qy_grid,xreciVectors,subbands,L,l_B,q,Q_x,Q_y)
    K = 2*pi/L
    xGVectors = Index(length(reciVectors),"xreciVector")
    iqx = Index(length(qx_grid),"qxMomentum")
    iqy = Index(length(qy_grid),"qyMomentum")
    subbands3 = Index(length(subBands),"subBand3")
    subbands4 = Index(length(subBands),"subBand4")
    phase1 = ITensor(iqx,iqy,xGvectors,subbands3,subbands4)

    @showprogress "Pre-computing phase 1..." for  l3_idx in 1:dim(subbands),l4_idx in 1:dim(subbands),qx_idx in 1:dim(iqx),qy_idx in 1:dim(iqy), Gx_idx in 1:dim(xGvectors)
        Gx = xreciVectors[Gx_idx]
        l4 = subbands4[l4_idx]
        l3 = subbands3[l3_idx]
        qx = iqx[qx_idx]
        qy = iqy[qy_idy]

        phase1[iqx(qx_idx),iqy(qy_idx),xGVectors(Gx_idx),subbands3(l3_idx),subbands4(l4_idx)] = exp(im*(q_x+K*G_x)*l_B^2*(Q_y - qy - (l4-l3)*K/q))

    end
    return phase1
end

function exchangeTermPhase2(kx4_grid,yreciVectors,subBands,L,p,q,Q_x,Q_y)

    ikx4 = Index(length(kx4_grid), "kx4Momentum")
    yGVectors = Index(length(reciVectors),"yreciVector")
    subbands1 = Index(length(subBands),"subBand1")
    subbands3 = Index(length(subBands),"subBand3")
    phase2 = ITensor(ikx4,yGVectors,subbands1,subbands3)

    @showprogress "Pre-computing phase 2..." for  l1_idx in 1:dim(subbands),l3_idx in 1:dim(subbands),kx4_idx in 1:dim(ikx4), Gy_idx in 1:dim(yGvectors)
        Gy = yreciVectors[Gy_idx]
        l3 = subbands3[l3_idx]
        l1 = subbands1[l1_idx]
        if mod(l3 - l1 + q*Gy,p) != 0
            phase2[ikx4(kx4_idx),GVectors(G_idx),subbands1(l1_idx),subbands3(l3_idx)] = 0
        else
            kx4 = kx_grid[kx_idx]
            phase2[ikx4(kx4_idx),GVectors(G_idx),subbands1(l1_idx),subbands3(l3_idx)] = exp(im*L*(kx4-Q_x)*(l3-l1+q*Gy)/p)
        end
    end
    return phase2
end

function exchangeTermPhase3(kx4_grid,qx_grid,yreciVectors,subBands,L,p,q,Q_x,Q_y)
    iqx = Index(length(qx_grid),"qxMomentum")
    ikx4 = Index(length(kx4_grid), "kx4Momentum")
    yGVectors = Index(length(reciVectors),"yreciVector")
    subbands2 = Index(length(subBands),"subBand2")
    subbands4 = Index(length(subBands),"subBand4")
    phase3 = ITensor(ikx4,iqx,yGVectors,subbands2,subbands4)

    @showprogress "Pre-computing phase 3..." for  l1_idx in 1:dim(subbands),l3_idx in 1:dim(subbands),kx4_idx in 1:dim(ikx4), Gy_idx in 1:dim(yGvectors), qx_idx in 1:dim(iqx)
        Gy = yreciVectors[Gy_idx]
        qx = qx_grid[qx_idx]
        l3 = subbands3[l3_idx]
        l1 = subbands1[l1_idx]
        if mod(l4 - l2 - q*Gy,p) != 0
            phase3[ikx4(kx4_idx),iqx(qx_idx),GVectors(G_idx),subbands2(l2_idx),subbands4(l4_idx)] = 0
        else
            kx4 = kx_grid[kx_idx]
            phase3[ikx4(kx4_idx),iqx(qx_idx),GVectors(G_idx),subbands2(l2_idx),subbands4(l4_idx)] = exp(im*L*(kx4+q_x)*(l4-l2-q*Gy)/p)
        end
    end
    return phase3
end

function directTermPhase1(ky3_grid,ky4_grid,xreciVectors,subbands,L,l_B,q,Q_x,Q_y)
    K = 2*pi/L
    xGVectors = Index(length(reciVectors),"xreciVector")
    iky3x = Index(length(ky3_grid),"ky3Momentum")
    iky4x = Index(length(ky4_grid),"ky4Momentum")
    subbands3 = Index(length(subBands),"subBand3")
    subbands4 = Index(length(subBands),"subBand4")
    phase1 = ITensor(iky3x,iky4y,xGvectors,subbands3,subbands4)

    @showprogress "Pre-computing phase 1..." for  l3_idx in 1:dim(subbands),l4_idx in 1:dim(subbands),ky3_idx in 1:dim(iky3x),ky4_idx in 1:dim(iky4x), Gx_idx in 1:dim(xGvectors)
        Gx = xreciVectors[Gx_idx]
        l4 = subbands4[l4_idx]
        l3 = subbands3[l3_idx]
        ky3 = ky3_grid[ky3_idx]
        ky4 = ky4_grid[ky4_idx]

        phase1[iky3(iky3_idx),iky4(iky4_idx),xGVectors(Gx_idx),subbands3(l3_idx),subbands4(l4_idx)] = exp(im*(Q_x+K*G_x)*l_B^2*(ky4-ky3- (l4-l3)*K/q))

    end
    return phase1
end

function directTermPhase2(kx3_grid,yreciVectors,subBands,L,p,q,Q_x,Q_y)

    ikx3 = Index(length(kx3_grid), "kx4Momentum")
    yGVectors = Index(length(reciVectors),"yreciVector")
    subbands1 = Index(length(subBands),"subBand1")
    subbands3 = Index(length(subBands),"subBand3")
    phase2 = ITensor(ikx3,yGVectors,subbands1,subbands3)

    @showprogress "Pre-computing phase 2..." for  l1_idx in 1:dim(subbands),l3_idx in 1:dim(subbands),kx3_idx in 1:dim(ikx3), Gy_idx in 1:dim(yGvectors)
        Gy = yreciVectors[Gy_idx]
        l3 = subbands3[l3_idx]
        l1 = subbands1[l1_idx]
        if mod(l3 - l1 + q*Gy,p) != 0
            phase2[ikx4(kx3_idx),GVectors(G_idx),subbands1(l1_idx),subbands3(l3_idx)] = 0
        else
            kx3 = kx3_grid[kx3_idx]
            phase2[ikx3(kx3_idx),GVectors(G_idx),subbands1(l1_idx),subbands3(l3_idx)] = exp(im*L*(kx3-Q_x)*(l3-l1+q*Gy)/p)
        end
    end
    return phase2
end

function directTermPhase3(kx4_grid,yreciVectors,subBands,L,p,q,Q_x,Q_y)

    ikx4 = Index(length(kx4_grid), "kx4Momentum")
    yGVectors = Index(length(reciVectors),"yreciVector")
    subbands2 = Index(length(subBands),"subBand2")
    subbands4 = Index(length(subBands),"subBand4")
    phase3 = ITensor(ikx4,yGVectors,subbands2,subbands4)

    @showprogress "Pre-computing phase 3..." for  l1_idx in 1:dim(subbands),l3_idx in 1:dim(subbands),kx4_idx in 1:dim(ikx4), Gy_idx in 1:dim(yGvectors), qx_idx in 1:dim(iqx)
        Gy = yreciVectors[Gy_idx]
        l3 = subbands3[l3_idx]
        l1 = subbands1[l1_idx]
        if mod(l4 - l2 - q*Gy,p) != 0
            phase3[ikx4(kx4_idx),GVectors(G_idx),subbands2(l2_idx),subbands4(l4_idx)] = 0
        else
            kx4 = kx_grid[kx_idx]
            phase3[ikx4(kx4_idx),GVectors(G_idx),subbands2(l2_idx),subbands4(l4_idx)] = exp(im*L*(kx4+Q_x)*(l4-l2-q*Gy)/p)
        end
    end
    return phase1
end

function directTerm(potentialTensor,formFactorTensor,phase1Tensor,phase2Tensor,phase3Tensor,Δ,Qx,Qy)
    return Δ(n4,λ4,S4,K4,l4,n2,λ2,S2,K2,l4,kx4,ky4)*potentialTensor(Gx,Gy)*formFactorTensor1(n1,λ1,n3,λ3,Gx,Gy)*formFactorTensor2(n2,λ2,n4,λ4,-Gx,-Gy)*phase1(ky3,ky4,Gx,l3,l4)*phase2(kx3,Gy,l1,l3) *phase3(kx4,Gy,l2,l4)
end

function exchangeTerm(potentialTensor,formFactorTensor,phase1Tensor,phase2Tensor,phase3Tensor,Δ,Qx,Qy)
    return Δ(n3,λ3,S3,K3,l3,n2,λ2,S2,K2,l2,kx4+q_x,ky4+q_x)*potentialTensor(qx,qy,Gx,Gy)*formFactorTensor1(n1,λ1,n3,λ3,qx,qy,Gx,Gy)*formFactorTensor2(n2,λ2,n4,λ4,-qx,-qy,-Gx,-Gy)*phase1(ky3,ky4,qx,qy,Gx,l3,l4)*phase2(kx4,Gy,l1,l3) *phase3(kx4,qx,Gy,l2,l4)
end

function computeHFHamiltonian(potentialTensor,formFactorTensor,phase1Tensor,phase2Tensor,phase3Tensor,Δ,Qx,Qy)
    return directTerm(potentialTensor,formFactorTensor,phase1Tensor,phase2Tensor,phase3Tensor,Δ,Qx,Qy) + exchangeTerm(potentialTensor,formFactorTensor,phase1Tensor,phase2Tensor,phase3Tensor,Δ,Qx,Qy)
end





function random_binary_density_matrix(N::Int, trace_val::Int)
    # --- Input Validation ---
    if N <= 0
        throw(ArgumentError("Matrix dimension N must be positive."))
    end
    if trace_val < 0
        throw(ArgumentError("Trace must be non-negative."))
    end
    if trace_val > N
        throw(ArgumentError("Trace cannot exceed matrix dimension N."))
    end

    # --- Step 1: Generate a random orthonormal basis (eigenvectors) ---
    # Start with a random complex matrix from a Gaussian distribution.
    A = randn(ComplexF64, N, N)
    # Construct a random Hermitian matrix from A. This ensures the eigenvectors
    # form a random basis.
    H_rand = A + A'
        # The eigenvectors of any Hermitian matrix form a complete orthonormal basis.
    F = eigen(H_rand)
    U = F.vectors # This is our random unitary matrix (basis)

    # --- Step 2: Create binary eigenvalues ---
    # Create eigenvalue vector with exactly trace_val ones and (N - trace_val) zeros
    binary_eigs = zeros(Float64, N)
    binary_eigs[1:trace_val] .= 1.0

    # Randomly shuffle the eigenvalues to avoid bias in which eigenvectors
    # correspond to the non-zero eigenvalues
    shuffle!(binary_eigs)

    # --- Step 3: Construct the density matrix ---
    # Combine the random basis (U) with the binary eigenvalues.
    # The resulting matrix Δ = U * Λ * U' is guaranteed to have exactly
    # trace_val eigenvalues equal to 1 and (N - trace_val) eigenvalues equal to 0.
    Δ = U * Diagonal(binary_eigs) * U'

# Ensure the output is explicitly Hermitian to maintain type stability and
# leverage optimized methods for Hermitian matrices.
return Hermitian(Δ)
end

function compute_and_print_order_parameters(Δ::AbstractMatrix, levels::Int, p::Int)
    matrix_size = size(Δ, 1)
    nF = round(Int, real(tr(Δ)))
    if nF == 0
        println("No particles, all order parameters are zero.")
        return
    end
    norm = 1.0 / nF

    # Initialize expectation values
    ops = Dict(
        "FM_Sx" => 0.0im, "FM_Sy" => 0.0im, "FM_Sz" => 0.0im,
        "AFM_Sx" => 0.0im, "AFM_Sy" => 0.0im, "AFM_Sz" => 0.0im,
        "VFM_Tx" => 0.0im, "VFM_Ty" => 0.0im, "VFM_Tz" => 0.0im,
        "VAFM_Tx" => 0.0im, "VAFM_Ty" => 0.0im, "VAFM_Tz" => 0.0im,
        "Sublattice_Pol" => 0.0im
        )

    for i in 1:matrix_size
        v1, l1, s1, n1, S1 = decompose_index_valleyful(i, p, levels)

        # Diagonal operators (Sz, Tz, Sublattice)
        ops["FM_Sz"] += Δ[i, i] * s1
        ops["VFM_Tz"] += Δ[i, i] * v1
        # For n>0, S1 is sublattice sign. For n=0, S1=v1, so we must be careful.
        sublattice_sign = (n1 > 0) ? S1 : 0 # No sublattice DoF for n=0
        ops["Sublattice_Pol"] += Δ[i, i] * sublattice_sign
        ops["AFM_Sz"] += Δ[i, i] * s1 * sublattice_sign
        ops["VAFM_Tz"] += Δ[i, i] * v1 * sublattice_sign

        # Off-diagonal operators (Sx, Sy, Tx, Ty)
        # Find partner states with one flipped quantum number
        j_spin = find_index_valleyful(v1, l1, -s1, n1, S1, p, levels)
        if j_spin != -1
            ops["FM_Sx"] += Δ[j_spin, i] # <i|c_j^+ c_i|i> ~ <S_x>
            ops["FM_Sy"] += Δ[j_spin, i] * im * s1 # <i|c_j^+ c_i|i> ~ <S_y>
            ops["AFM_Sx"] += Δ[j_spin, i] * sublattice_sign
            ops["AFM_Sy"] += Δ[j_spin, i] * im * s1 * sublattice_sign
        end

        j_valley = find_index_valleyful(-v1, l1, s1, n1, S1, p, levels)
        if j_valley != -1
            ops["VFM_Tx"] += Δ[j_valley, i]
            ops["VFM_Ty"] += Δ[j_valley, i] * im * v1
            ops["VAFM_Tx"] += Δ[j_valley, i] * sublattice_sign
            ops["VAFM_Ty"] += Δ[j_valley, i] * im * v1 * sublattice_sign
        end
    end

    println("\n--- Hartree-Fock Order Parameters ---")
    println("Normalization Factor (1/nF): $norm")
    println("\nSpin Polarization:")
    @printf "  - Ferromagnetic (FM):   Sx=%.4f, Sy=%.4f, Sz=%.4f\n" real(norm*ops["FM_Sx"]) real(norm*ops["FM_Sy"]) real(0.5*norm*ops["FM_Sz"])
    @printf "  - Antiferro (AFM):      Sx=%.4f, Sy=%.4f, Sz=%.4f\n" real(norm*ops["AFM_Sx"]) real(norm*ops["AFM_Sy"]) real(0.5*norm*ops["AFM_Sz"])

    println("\nValley Polarization:")
    @printf "  - Valley-FM (VFM):      Tx=%.4f, Ty=%.4f, Tz=%.4f\n" real(norm*ops["VFM_Tx"]) real(norm*ops["VFM_Ty"]) real(0.5*norm*ops["VFM_Tz"])
    @printf "  - Valley-AFM (VAFM):    Tx=%.4f, Ty=%.4f, Tz=%.4f\n" real(norm*ops["VAFM_Tx"]) real(norm*ops["VAFM_Ty"]) real(0.5*norm*ops["VAFM_Tz"])

    println("\nSublattice Polarization:")
    @printf "  - <Σ_z>: %.4f\n" real(norm*ops["Sublattice_Pol"])
    println("-------------------------------------\n")
end





