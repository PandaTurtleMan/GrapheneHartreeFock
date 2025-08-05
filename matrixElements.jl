import QuadGK
import Kronecker
import LinearAlgebra
import AngleBetweenVectors
include("Utilities.jl")

function screening_fn(q_norm)
    return tanh(q_norm)
end

function landauFourierMatrixElement(q_x,q_y,n_1,n_2,l_B)
    if n_1 < 0 || n_2 < 0
        return 0
    end
    n,m = max(n_1,n_2),min(n_1,n_2)
    deltaN = n_1-n_2
    theta = angle((qx,qy),(1,0))
    phaseFactor = exp(-0.5*im*(q_x*q_y*l_B^2) - i*deltaN*theta)
    return phaseFactor*laguerreTilde(abs(deltaN),m,0.5*(q_x^2+q_y^2)*l_B^2)
end

function grapheneLandauFourierMatrixElement(q_x,q_y,n_1,n_2,l_B,lambda1,lambda2)
    deltaFactor = sqrt(2)^((n_1 == 0)+(n_2==0))
    return deltaFactor*(landauFourierMatrixElement(q_x,q_y,n_1,n_2,l_B) + lambda1*lambda2*landauFourierMatrixElement(q_x,q_y,n_1-1,n_2-1,l_B))
end

function fourierMatrixElement(k_x,k_y,n1,n2,l1,l2,lambda1,lambda2,spin1,spin2,valley1,valley2,L,p,q,q1,q2)

    K= 2*pi/L

    k1 = k_x/K
    k2 = k_y/K

    l_B = (q/p)*L
    if valley1 != valley2 || spin1 != spin2
        return 0
    end

    if mod((l2 - l1 - q*q2),p) != 0
        return 0
    end
    s = (l2-l1-q*q2)/p
    S = grapheneLandauFourierMatrixElement(q1*K,q2*K,n_1,n_2,l_B,lambda1,lambda2)
    kTilde = (k_1 + l_1/q)*K
    return S*exp(2*pi*im*p*s*(k_x-q1*K))*exp(im*q1*K*kTilde*l_B^2)
end

function matrixElement(k_x, k_y, n1, n2, l1, l2, lambda1, lambda2, spin1, spin2, valley1, valley2, L, p, q, harmonics)
    total_element = 0.0 + 0.0im
    if isempty(harmonics) || all(h -> h==0, harmonics)
        return total_element
    end

    for (i, coeff) in enumerate(harmonics)
        if coeff == 0
            continue
        end
        q_vectors = get_c4_harmonic_vectors(i)
        for (q1, q2) in q_vectors
            total_element += coeff * fourierMatrixElement(k_x, k_y, n1, n2, l1, l2, lambda1, lambda2, spin1, spin2, valley1, valley2, L, p, q, q1, q2)
        end
    end
    return total_element
end

function evaluateLandauLevelMatrixElements(n, B, lambda,m)
    hbar = 1.0
    e = 1.0
    v_F = 1.0
    return v_F * lambda*sqrt(m + 2.0 * hbar * abs(B) * e * abs(n))
end

function zeemanEnergy(bohrMagneton,B,spinSign)
    return bohrMagneton*B*spinSign
end

function precomputeFormFactorTensorNonInt(levels, G_vectors, L, l_B)
    # Create indices
    i_n = Index(levels, "n")
    i_λ = Index(2, "λ")  # sublattice index
    iGx = Index(length(G_vectors[1]), "Gx")
    iGy = Index(length(G_vectors[2]), "Gy")

    S = ITensor(i_n, i_λ, i_n', i_λ', iGx, iGy)

    K = 2π / L
    for (Gx_idx, Gx) in enumerate(G_vectors[1]), (Gy_idx, Gy) in enumerate(G_vectors[2])
        qx = Gx * K
        qy = Gy * K

        for n1 in 1:levels, n2 in 1:levels, λ1 in 1:2, λ2 in 1:2
            λ1_val = λ1 == 1 ? 1 : -1
            λ2_val = λ2 == 1 ? 1 : -1
            val = grapheneLandauFourierMatrixElement(qx, qy, n1-1, n2-1, l_B, λ1_val, λ2_val)
            S[i_n(n1), i_λ(λ1), i_n'(n2), i_λ'(λ2), iGx(Gx_idx), iGy(Gy_idx)] = val
        end
    end
    return S, (i_n, i_λ, iGx, iGy)
end








