include("Utilities.jl")

export landauLevelWaveFunction, realSpaceWaveFunction, coeffVectorToRealSpace

function landauLevelWaveFunction(n, k_y, position, B, L)
    x, y = position
    ϵ = 1e-12
    l_B = 1/(sqrt(B) + ϵ)
    # Fix undefined variables and add Hermite-Gaussian function
    return exp(im * k_y * y) / (sqrt(L * l_B)) *
        exp(-(x - k_y * l_B^2)/l_B / 2) *
            hermite_poly(n, (x - k_y * l_B^2)/l_B)
end

function realSpaceWaveFunction(n, S, l, p, q, momentum, position, L)
    ϵ = 1e-12
    B = (p/q)
    l_B = 1/(sqrt(B) + ϵ)
    k_x, k_y = momentum
    K_y = 2*pi/L
    result = 0
    for s in -5:5
        # Fix undefined 'sp' -> 's*p'
        result += exp(-im * k_x * l_B^2 * (s*p + l) * K_y) *
            landauLevelWaveFunction(n, k_y + (s*p + l)*K_y, position, B, L)
    end
    return result
end

function coeffVectorToRealSpace(coeffVector,p,q,momentum,position,L,levels)
    result = 0
    vectorSize = length(coeffVector)

    for i in 1:vectorSize
        S = iseven(i) ? -1 : 1
        l = div(i-1, 2*levels) + 1
        n = div(mod(i-1, 2*levels), 2)
        result += realSpaceWaveFunction(n,S,l,p,q,momentum,position,L)
    end
    return result
end
