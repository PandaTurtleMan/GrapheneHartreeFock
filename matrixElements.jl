import QuadGK
import Kronecker
import LinearAlgebra
include("Utilities.jl")

#potential matrix elements
function matrixElement( k_x, k_y, S, S_prime,l, l_prime,n, n_prime,  N, B, L,valley1,valley2; ϵ=1e-12)
    # Constants and basic terms (with ϵ protection)
    V0 = 1.0
    Cn = 1.0
    Cn_prime = 1.0
    phi_0 = 1
    ξ = abs(n - n_prime)
    l_b = phi_0/(sqrt(B) + ϵ)
    β = (B + ϵ)*L^2/phi_0  # beta is p/q
    K1 = 2π /L  # Protect against k_y=0
    d_x = L + ϵ
    d_y = L + ϵ
    K2 = 2π/(L + ϵ)  # Protect L=0 case
    # Protected division operations
    function safe_div(num, den)
        num/(den + ϵ*(den == 0))
    end
    # Helper functions with ϵ protection
    W_rs(r, s) = π * (s^2*K2^2 + r^2*K1^2)/safe_div(K1*K2, 1.0)
    function D_rs_B(r, s)
        min_n = min(n, n_prime)
        max_n = max(n, n_prime)
        W = W_rs(r, s)
        x = safe_div(W, β)
        term1 = sqrt(safe_div(factorial(min_n), factorial(max_n)))
        term2 = exp(-safe_div(W, 2β))
        term3 = x^(ξ/2)
        term4 = laguerre_poly( ξ,min_n, x)
        term1 * term2 * term3 * term4
    end

    function D_rs_A(r, s)
        n_A = n - 1
        n_prime_A = n_prime - 1
        (n_A < 0 || n_prime_A < 0) && return 0.0
        ξ_A = abs(n_A - n_prime_A)
        min_n = min(n_A, n_prime_A)
        max_n = max(n_A, n_prime_A)
        W = W_rs(r, s)
        x = safe_div(W, β)
        term1 = sqrt(safe_div(factorial(min_n), factorial(max_n)))  # Fixed parenthesis
        term2 = exp(-safe_div(W, 2β))
        term3 = x^(ξ_A/2)
        #laguerre poly in wrong order
        term4 = laguerre_poly(ξ_A,min_n,  x)
        term1 * term2 * term3 * term4
    end

    # Protected trigonometric functions
    function T_l(s)
        delta1 = (mod(ξ,4) == 0) ? 1 : 0
        delta2 = (mod(ξ,4) == 2 ) ? 1 : 0
        delta3 = (mod(ξ,4) == 1) ? 1 : 0
        delta4 = (mod(ξ,4) == 3 ) ? 1 : 0
        2 * cos(s * safe_div(k_y*L - 2π*l, β))*(delta1-delta2) + 2*sin(s * safe_div(k_y*L - 2π*l, β))*(-delta3+delta4)
        #fix this
    end


    function A1_A(r, s)
        # Check if n-1 or n_prime-1 are valid indices first
        (n <= 0 || n_prime <= 0) && return 0.0
        (l == l_prime) ? D_rs_A(r, s) * T_l(s) : 0.0 # T_l might also need adjustment if it depends on n? Check formula.
    end

    function A2_A(r, s)
        (n <= 0 || n_prime <= 0) && return 0.0
        delta1 = (l - l_prime == r) ? 1 : 0
        delta2 = (l_prime - l == r) ? 1 : 0
        ξ = abs((n-1) - (n_prime-1))
        D_rs_A(r, s)* (delta1*(sign(n_prime-n))^ξ + delta2*(sign(n-n_prime))^ξ)
    end

    function A3_A(r, s)
        (n <= 0 || n_prime <= 0) && return 0.0
        delta1 = (l - l_prime == r) ? 1 : 0
        delta2 = (l_prime - l == r) ? 1 : 0
        ξ = abs((n-1) - (n_prime-1))
        theta = safe_div(s, β) * (k_y*L - 2π*(l + r/2)) -
            ξ* sign(n_prime - n) * atan(safe_div(s*L, r*L)) # Assuming ξ_A dependence here
        D_rs_A(r, s) * delta1*((sign(n_prime-n))^ξ)*cos(theta) + delta2*((sign(n-n_prime))^ξ)*cos(theta) # Check definition
    end

    function A1_B(r, s)
        (l == l_prime) ? D_rs_B(r, s) * T_l(s) : 0.0
    end

    function A2_B(r, s)
        (n <= 0 || n_prime <= 0) && return 0.0
        delta1 = (l - l_prime == r) ? 1 : 0
        delta2 = (l_prime - l == r) ? 1 : 0
        ξ = abs((n-1) - (n_prime-1))
        D_rs_B(r, s) * (delta1*(sign(n_prime-n))^ξ) + delta2*((sign(n-n_prime))^ξ)
    end

    # A functions with protected divisions
    function A3_B(r, s)
        (n <= 0 || n_prime <= 0) && return 0.0
        delta1 = (l - l_prime == r) ? 1 : 0
        delta2 = (l_prime - l == r) ? 1 : 0
        ξ = abs((n-1) - (n_prime-1))
        theta = safe_div(s, β) * (k_y*L - 2π*(l + r/2)) -
            ξ* sign(n_prime - n) * atan(safe_div(s*L, r*L)) # Assuming ξ_A dependence here
        D_rs_B(r, s) * delta1*((sign(n_prime-n))^ξ)*cos(theta) + delta2*((sign(n-n_prime))^ξ)*cos(theta)
    end

    F_B_sum = 0.0
    F_A_sum = 0.0
    for i in 0:N-1
        for j in 0:N-1
            # B-type terms
            #fix this?
            F_B = binomial(2N, i)*binomial(2N, N)*A1_B(0, N-i) +binomial(2N, j)*binomial(2N, N)*A2_B(N-j, 0) +2*binomial(2N, i)*binomial(2N, j)*A3_B(N-j, N-i)
            # A-type terms (mirror B-type with n->n-1)
            F_A = binomial(2N, i)*binomial(2N, N)*A1_A(0, N-i) + binomial(2N, j)*binomial(2N, N)*A2_A(N-j, 0) +2*binomial(2N, i)*binomial(2N, j)*A3_A(N-j, N-i)
            F_B_sum += F_B
            F_A_sum += F_A * S * S_prime  # Include alpha coefficients
        end
    end

    # Delta terms
    delta_ll = (l == l_prime) ? 1 : 0
    delta_nn = (n == n_prime) ? 1 : 0
    central_term = delta_ll * delta_nn * (1 + S*S_prime) * (factorial(2N)/(factorial(N)^2))^2

    prefactor = safe_div(V0 * Cn_prime * Cn, 4^(2N))
    exp_term = exp(im * k_x * K1* l_b^2 * (l - l_prime))
    if n== 0 && n_prime == 0
        return 0
    end
    if (n == 0) && valley1 == 0
        return prefactor*exp_term*F_A_sum*S_prime
    end
    if (n_prime == 0) && valley1 == 0
        return prefactor*exp_term*F_A_sum*S
    end
    if (n == 0 || n_prime == 0) && valley1 == 1
        return prefactor*exp_term*F_B_sum
    end


    return prefactor * (exp_term * (F_B_sum + F_A_sum))
end

#project a general c4 symmetric potential onto potentials of the form used in the referece paper
function compute_fourier_coefficients(N::Int)
    # Generate C4 orbit representatives (a, b) where 0 ≤ b ≤ a ≤ N
    representatives = Tuple{Int, Int}[]
    for a in 0:N
        for b in 0:a
            push!(representatives, (a, b))
        end
    end

    # Sort by distance squared (primary), then -a, then -b
    sort!(representatives, by = pair -> (pair[1]^2 + pair[2]^2, -pair[1], -pair[2]))

    # Compute coefficients for each representative
    coefficients = Float64[]
    for (a, b) in representatives
        coeff_x = binomial(2N, N - a)
        coeff_y = binomial(2N, N - b)
        coeff = (coeff_x * coeff_y) / 4^(2N)
        push!(coefficients, coeff)
    end

    return coefficients
end

#diagonal part of the Hamiltonian
function evaluateLandauLevelMatrixElements(k_x, k_y, m, n, B, S1, S2)
    hbar = 1.0
    e = 1.0
    v_F = 1.0
    l_B = sqrt(hbar / (abs(B) * e))
    (m == n && S1 == S2) ? S1 * v_F * sqrt(2.0 * hbar * abs(B) * e * abs(n)) : 0.0
end

#go from coloumb interaction in normal Landau bloch states to graphene Landau bloch states
function constructVTensorComponents(Si,Sj,Sk,Sl,ni,nj,nk,nl,li,lj,lk,ll,k_x,k_y,L,l_B,screeningFunction,cutoff)
    return  classicalPartOfV(ni,nj,nk,nl,li,lj,lk,ll,k_x,k_y,L,l_B,screeningFunction,cutoff) + Si*Sk* classicalPartOfV(ni-1,nj,nk-1,nl,li,lj,lk,ll,k_x,k_y,L,l_B,screeningFunction,cutoff)*theta(ni-1)*theta(nk-1)
    + Sj*Sl* classicalPartOfV(ni,nj-1,nk,nl-1,li,lj,lk,ll,k_x,k_y,L,l_B,screeningFunction,cutoff)*theta(nj-1)*theta(nl-1)
    + Si*Sj*Sk*Sl*classicalPartOfV(ni-1,nj-1,nk-1,nl-1,li,lj,lk,ll,k_x,k_y,L,l_B,screeningFunction,cutoff)*theta(ni-1)*theta(nk-1)*theta(nj-1)*theta(nl-1)
end

#coloumb interaction in normal Landau bloch state
function classicalPartOfV(ni,nj,nk,nl,li,lj,lk,ll,k_x,k_y,L,l_B,screeningFunction,cutoff)
    if lj != ll
        return 0
    end
    constant = im^(abs(ni-nk)+ni-nk+abs(nj-nl)+nj-nl)*exp(-im(k_x)(lk-ki)*l_B^2)/l_B
    function functionalForm(r,theta)
        return laguerreTilde(abs(ni-nk),min(ni,nk),r^2/2)*laguerreTilde(abs(nj-nl),min(nj,nl),r^2/2)*screeningFunction(r/l_B)*exp(im*(ni-nk+nj-nl)*theta + im*r^2*cos(theta)*sin(theta))
    end
    hubbardTerm = 0
    integrand = constant*functionalForm
    return integrationInPolarCoordinates(integrand,cutoff)
end

#add spins to the interactions, optional Hubbard parameter cuz why not
function constructSpinFullVTensor(k_x,k_y,p,q,L,cutoff,levels,screeningFunction,l_B,U,charge)
    matrixSize = 4 * levels * p
    Vtensor = zeros(ComplexF64, matrixSize, matrixSize, matrixSize, matrixSize)
    #return zeros(ComplexF64, matrixSize, matrixSize, matrixSize, matrixSize)
    # Precompute radial integrals I[ni, nj, nk, nl]
    I = zeros(ComplexF64, levels, levels, levels, levels)

    # Progress meter for integral precomputation
    total_combos = levels^4
    progress = Progress(total_combos, desc="Precomputing integrals: ", barglyphs=BarGlyphs('|','█','▁','|',' '), output=stderr)

    Threads.@threads for idx in 1:total_combos
        ni, nj, nk, nl = Tuple(CartesianIndices((levels, levels, levels, levels))[idx].I)
        n_i = ni-1; n_j = nj-1; n_k = nk-1; n_l = nl-1

        # actually prove this!!!! why is it zero if m is odd????
        if mod((n_i - n_k + n_j - n_l),2) == 0
            functionalForm(r,theta) = r * (
                laguerreTilde(abs(n_i - n_k), min(n_i, n_k), r^2/2) *
                    laguerreTilde(abs(n_j - n_l), min(n_j, n_l), r^2/2) *
                        screeningFunction(r / l_B)
                    )*cos((ni-nk+nj-nl)*theta + r^2*cos(theta)*sin(theta))
            I[ni, nj, nk, nl] = integrationInPolarCoordinates(functionalForm,cutoff)
        end

        next!(progress)
    end

    # Build tensor using precomputed integrals
    progress = Progress(matrixSize^4, desc="Building V tensor: ", barglyphs=BarGlyphs('|','█','▁','|',' '), output=stderr)

    for i in 1:matrixSize
        Si = iseven(i) ? -1 : 1
        spini = isOddMod4(i) ? -1 : 1
        li = div(i-1, 4*levels) + 1
        ni_idx = div(mod(i-1, 4*levels), 4)+1
        n_i = ni_idx - 1

        for j in 1:matrixSize
            Sj = iseven(j) ? -1 : 1
            spinj = isOddMod4(j) ? -1 : 1
            lj = div(j-1, 4*levels) + 1
            nj_idx = div(mod(j-1, 4*levels), 4)+1
            n_j = nj_idx - 1

            for k in 1:matrixSize
                Sk = iseven(k) ? -1 : 1
                spink = isOddMod4(k) ? -1 : 1
                lk = div(k-1, 4*levels) + 1
                nk_idx = div(mod(k-1, 4*levels), 4)+1
                n_k = nk_idx - 1

                for l in 1:matrixSize
                    Sl = iseven(l) ? -1 : 1
                    spinl = isOddMod4(l) ? -1 : 1
                    ll = div(l-1, 4*levels) + 1
                    nl_idx = div(mod(l-1, 4*levels), 4)+1
                    n_l = nl_idx - 1

                    if n_i == n_j && n_k == n_l && n_j == n_k && li == lj && lj == lk && lk == ll && Si == 1 && Sj == -1 && Sk == -1 && Sl == -1
                        Vtensor[i, j, k, l] += -U
                    end

                    # Skip if lj != ll (conservation condition)
                    if lj != ll
                        Vtensor[i, j, k, l] = 0.0
                        next!(progress)
                        continue
                    end

                    # Calculate constant factor
                    exp_val = abs(n_i - n_k) + n_i - n_k + abs(n_j - n_l) + n_j - n_l
                    constant = charge^2*im^exp_val * exp(-im * k_x * (lk - li) * l_B^2)

                    # Base term (no decrement)
                    term0 = I[ni_idx, nj_idx, nk_idx, nl_idx] * constant

                    # Initialize other terms
                    term1 = 0.0; term2 = 0.0; term3 = 0.0

                    # Term1: ni-1, nk-1
                    if n_i >= 1 && n_k >= 1
                        term1 = I[ni_idx-1, nj_idx, nk_idx-1, nl_idx] * constant
                    end

                    # Term2: nj-1, nl-1
                    if n_j >= 1 && n_l >= 1
                        term2 = I[ni_idx, nj_idx-1, nk_idx, nl_idx-1] * constant
                    end

                    # Term3: all decremented
                    if n_i >= 1 && n_j >= 1 && n_k >= 1 && n_l >= 1
                        term3 = I[ni_idx-1, nj_idx-1, nk_idx-1, nl_idx-1] * constant
                    end
                    numberOfZeros = (n_i == 0) + ( n_l == 0) + (n_j == 0) + (n_k == 0)
                    # Combine terms with spinor signs
                    Vtensor[i, j, k, l] += 0.25*(sqrt(2))^(numberOfZeros)*(term0 + Si*Sk*term1 + Sj*Sl*term2 + Si*Sj*Sk*Sl*term3)
                    next!(progress)
                end
            end
        end
    end

    return Vtensor
end

#add valley back in, coloumb interaction is SU(4) symmetric and thus indepdednt of the valley index
function constructValleyfulSpinFullVtensor(k_x,k_y,p,q,L,cutoff,levels,screeningFunction,l_B,charge)
    matrixSize = 8 * levels * p + 4 * p
    Vtensor = zeros(ComplexF64, matrixSize, matrixSize, matrixSize, matrixSize)
    #return zeros(ComplexF64, matrixSize, matrixSize, matrixSize, matrixSize)
    # Precompute radial integrals I[ni, nj, nk, nl]

    I = zeros(ComplexF64, levels+1, levels+1, levels+1, levels+1)
    total_combos = (levels+1)^4
    progress = Progress(total_combos, desc="Precomputing integrals: ", barglyphs=BarGlyphs('|','█','▁','|',' '), output=stderr)
    Threads.@threads for idx in 1:total_combos
        ni, nj, nk, nl = Tuple(CartesianIndices((levels+1, levels+1, levels+1, levels+1))[idx].I)
        n_i = ni-1; n_j = nj-1; n_k = nk-1; n_l = nl-1



        # actually prove this!!!! why is it zero if m is odd????
        if mod((n_i - n_k + n_j - n_l),2) == 0
            functionalForm(r,theta) = r * (
                laguerreTilde(abs(n_i - n_k), min(n_i, n_k), r^2/2) *
                    laguerreTilde(abs(n_j - n_l), min(n_j, n_l), r^2/2) *
                        screeningFunction(r / l_B)
                    )*cos((ni-nk+nj-nl)*theta + r^2*cos(theta)*sin(theta))
            I[ni, nj, nk, nl] = integrationInPolarCoordinates(functionalForm,cutoff)
        end

        next!(progress)
    end

    # Build tensor using precomputed integrals
    progress = Progress(matrixSize^4, desc="Building V tensor: ", barglyphs=BarGlyphs('|','█','▁','|',' '), output=stderr)

    for i in 1:matrixSize
        valley1, l1, spin_sign1, n1, S1 = decompose_index_valleyful(i, p, levels)
        for j in 1:matrixSize
            valley2, l2, spin_sign2, n2, S2 = decompose_index_valleyful(j, p, levels)
            for k in 1:matrixSize
                valley3, l3, spin_sign3, n3, S3 = decompose_index_valleyful(k, p, levels)
                for l in 1:matrixSize
                    valley4, l4, spin_sign4, n4, S4 = decompose_index_valleyful(l, p, levels)
                    if l2 != l4
                        Vtensor[i, j, k, l] = 0.0
                        next!(progress)
                        continue
                    end

                    # Calculate constant factor
                    exp_val = abs(n1 - n3) + n1 - n3 + abs(n2 - n4) + n2 - n4
                    constant = charge^2 * im^exp_val * exp(-im * k_x * (l3 - l1) * l_B^2)

                    # Base term (no decrement)
                    ni_idx = n1 + 1
                    nj_idx = n2 + 1
                    nk_idx = n3 + 1
                    nl_idx = n4 + 1
                    term0 = I[ni_idx, nj_idx, nk_idx, nl_idx] * constant

                    # Initialize other terms
                    term1 = 0.0; term2 = 0.0; term3 = 0.0

                    # Term1: n1-1, n3-1
                    if n1 >= 1 && n3 >= 1
                        term1 = I[ni_idx-1, nj_idx, nk_idx-1, nl_idx] * constant
                    end

                    # Term2: n2-1, n4-1
                    if n2 >= 1 && n4 >= 1
                        term2 = I[ni_idx, nj_idx-1, nk_idx, nl_idx-1] * constant
                    end

                    # Term3: all decremented
                    if n1 >= 1 && n2 >= 1 && n3 >= 1 && n4 >= 1
                        term3 = I[ni_idx-1, nj_idx-1, nk_idx-1, nl_idx-1] * constant
                    end

                    numberOfZeros = (n1 == 0) + (n4 == 0) + (n2 == 0) + (n3 == 0)
                    # Combine terms with spinor signs
                    Vtensor[i, j, k, l] += 0.25 * (sqrt(2))^numberOfZeros * (
                        term0 +
                            S1 * S3 * term1 +
                                S2 * S4 * term2 +
                                    S1 * S2 * S3 * S4 * term3
                                )
                    next!(progress)
                end
            end
        end
    end

    return Vtensor
end

#no spin or valley indices
function constructFullVTensor(k_x, k_y, p, q, L, cutoff, levels, screeningFunction, l_B,charge)
    matrixSize = 2 * levels * p
    Vtensor = zeros(ComplexF64, matrixSize, matrixSize, matrixSize, matrixSize)
    #return zeros(ComplexF64, matrixSize, matrixSize, matrixSize, matrixSize)
    # Precompute radial integrals I[ni, nj, nk, nl]
    I = zeros(ComplexF64, levels, levels, levels, levels)

    # Progress meter for integral precomputation
    total_combos = levels^4
    progress = Progress(total_combos, desc="Precomputing integrals: ", barglyphs=BarGlyphs('|','█','▁','|',' '), output=stderr)

    Threads.@threads for idx in 1:total_combos
        ni, nj, nk, nl = Tuple(CartesianIndices((levels, levels, levels, levels))[idx].I)
        n_i = ni-1; n_j = nj-1; n_k = nk-1; n_l = nl-1

        # actually prove this!!!! why is it zero if m is odd????
        if mod((n_i - n_k + n_j - n_l),2) == 0
            functionalForm(r,theta) = r * (
                laguerreTilde(abs(n_i - n_k), min(n_i, n_k), r^2/2) *
                    laguerreTilde(abs(n_j - n_l), min(n_j, n_l), r^2/2) *
                        screeningFunction(r / l_B)
                    )*cos((n_i-n_k+n_j-n_l)*theta + r^2*cos(theta)*sin(theta))
            I[ni, nj, nk, nl] = integrationInPolarCoordinates(functionalForm,cutoff)
        end

        next!(progress)
    end

    # Build tensor using precomputed integrals
    progress = Progress(matrixSize^4, desc="Building V tensor: ", barglyphs=BarGlyphs('|','█','▁','|',' '), output=stderr)

    for i in 1:matrixSize
        Si = iseven(i) ? -1 : 1
        li = div(i-1, 2*levels) + 1
        ni_idx = div(mod(i-1, 2*levels), 2) + 1
        n_i = ni_idx - 1

        for j in 1:matrixSize
            Sj = iseven(j) ? -1 : 1
            lj = div(j-1, 2*levels) + 1
            nj_idx = div(mod(j-1, 2*levels), 2) + 1
            n_j = nj_idx - 1

            for k in 1:matrixSize
                Sk = iseven(k) ? -1 : 1
                lk = div(k-1, 2*levels) + 1
                nk_idx = div(mod(k-1, 2*levels), 2) + 1
                n_k = nk_idx - 1

                for l in 1:matrixSize
                    Sl = iseven(l) ? -1 : 1
                    ll = div(l-1, 2*levels) + 1
                    nl_idx = div(mod(l-1, 2*levels), 2) + 1
                    n_l = nl_idx - 1

                    # Skip if lj != ll (conservation condition)
                    if lj != ll
                        Vtensor[i, j, k, l] = 0.0
                        next!(progress)
                        continue
                    end

                    # Calculate constant factor
                    exp_val = abs(n_i - n_k) + n_i - n_k + abs(n_j - n_l) + n_j - n_l
                    constant = charge^2*im^exp_val * exp(-im * k_x * (lk - li) * l_B^2)

                    # Base term (no decrement)
                    term0 = I[ni_idx, nj_idx, nk_idx, nl_idx] * constant

                    # Initialize other terms
                    term1 = 0.0; term2 = 0.0; term3 = 0.0

                    # Term1: ni-1, nk-1
                    if n_i >= 1 && n_k >= 1
                        term1 = I[ni_idx-1, nj_idx, nk_idx-1, nl_idx] * constant
                    end

                    # Term2: nj-1, nl-1
                    if n_j >= 1 && n_l >= 1
                        term2 = I[ni_idx, nj_idx-1, nk_idx, nl_idx-1] * constant
                    end

                    # Term3: all decremented
                    if n_i >= 1 && n_j >= 1 && n_k >= 1 && n_l >= 1
                        term3 = I[ni_idx-1, nj_idx-1, nk_idx-1, nl_idx-1] * constant
                    end
                    numberOfZeros = (n_i == 0) + ( n_l == 0) + (n_j == 0) + (n_k == 0)
                    # Combine terms with spinor signs
                    Vtensor[i, j, k, l] = 0.25*(sqrt(2))^(numberOfZeros)*(term0 + Si*Sk*term1 + Sj*Sl*term2 + Si*Sj*Sk*Sl*term3)
                    next!(progress)
                end
            end
        end
    end

    return Vtensor
end

#function constructClassicalDirectTerm(k_x,k_y,n_i,n_j,l_i,l_j,levels,Δ,L,l_B,screeningFunction,cutoff)
 #   matrixSize = size(Δ,1)
 #   result = 0
 #   #in our case is indepdent of l
 #   for l in 1:matrixSize
 #       Sl = iseven(l) ? -1 : 1
 #       ll = div(l-1, 2*levels) + 1
 #       n_l = div(mod(l-1, 2*levels), 2)
 #       for k in 1:matrixSize
 #           Sk = iseven(k) ? -1 : 1
 #           lk = div(k-1, 2*levels) + 1
 #           n_k = div(mod(k-1, 2*levels), 2)
 #           Δn = (n_i-n_l + n_k - n_j)
 #           integral_result = QuadGK.quadgk(x -> begin
 #                                               term1 = integrateBesselFunctionFrom0ToA(Δn, pi/L - #k_x*x*l_B, cutoff)
  #                                              term2 = integrateBesselFunctionFrom0ToA(Δn, -pi/L - k_x*x*l_B, cutoff)

  #                                              bessel_sum = iseven(Δn) ? (term1 + term2) : (term1 - term2)

   #                                             lag1 = laguerreTilde(abs(n_i-n_l), min(n_i,n_l), x^2/2)
    #                                            lag2 = laguerreTilde(abs(n_k-n_j), min(n_j,n_k), x^2/2)

     #                                           screening_val = screeningFunction(x/l_B)
      #                                          weight = x / l_B

       #                                         bessel_sum * lag1 * lag2 * screening_val * weight
        #                                        end, 0, cutoff)[1]
        #    result += integral_result*Δ[l,k]
    #end
   # return result
#end
#end

#function constructClassicalExchangeTerm(k_x,k_y,n_i,n_j,l_i,l_j,levels,Δ,L,l_B,screeningFunction,cutoff)
#matrixSize = size(Δ,1)
#result = 0
##in our case is indepdent of l
#for l in 1:matrixSize[1]
#    Sl = iseven(l) ? -1 : 1
#    ll = div(l-1, 2*levels) + 1
#    n_l = div(mod(l-1, 2*levels), 2)
#    for k in 1:matrixSize
#        Sk = iseven(k) ? -1 : 1
#        lk = div(k-1, 2*levels) + 1
#        n_k = div(mod(k-1, 2*levels), 2)
#        Δn = (n_i-n_j + n_k - n_l)
#        integral_result = QuadGK.quadgk(x -> begin
#                                            term1 = integrateBesselFunctionFrom0ToA(Δn, pi/L - k_x*x*l_B, cutoff)
#                                            term2 = integrateBesselFunctionFrom0ToA(Δn, -pi/L - k_x*x*l_B, cutoff)

 #                                           bessel_sum = iseven(Δn) ? (term1 + term2) : (term1 - term2)

  #                                          lag1 = laguerreTilde(abs(n_i-n_j), min(n_i,n_j), x^2/2)
   #                                         lag2 = laguerreTilde(abs(n_k-n_l), min(n_l,n_k), x^2/2)

    #                                        screening_val = screeningFunction(x/l_B)
     #                                       weight = x / l_B

      #                                      bessel_sum * lag1 * lag2 * screening_val * weight
       #                                     end, 0, cutoff)[1]
    #end
#end
#return result
#end




