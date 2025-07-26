import QuadGK
import Kronecker
import LinearAlgebra
include("Utilities.jl")



function landauFourierMatrixElement(q_x,q_y,n_1,n_2,l_B)
    if n_1 < 0 || n_2 < 0
        return 0
    end
    n,m = max(n_1,n_2),min(n_1,n_2)
    deltaN = n_1-n_2
    theta = atan2(q_y/q_x)
    phaseFactor = exp(-0.5*im*(q_x*q_y*l_B^2) - i*deltaN*theta)
    return phaseFactor*laguerreTilde(abs(deltaN),m,0.5*(q_x^2+q_y^2)*l_B^2)
end

function grapheneLandauFourierMatrixElement(q_x,q_y,n_1,n_2,l_B,lambda1,lambda2)
    deltaFactor = sqrt(2)^((n_1 == 0)+(n_2==0))
    return deltaFactor*(landauFourierMatrixElement(q_x,q_y,n_1,n_2,l_B) + lambda1*lambda2*landauFourierMatrixElement(q_x,q_y,n_1-1,n_2-1,l_B))
end

function fourierMatrixElement(k_x,k_y,n1,n2,l1,l2,lambda1,lambda2,spin1,spin2,valley1,valley2,L,p,q,q1,q2)
    K= 2*pi/L
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









