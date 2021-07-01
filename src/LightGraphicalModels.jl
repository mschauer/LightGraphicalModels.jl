module LightGraphicalModels
using LightGraphs, NamedTupleTools, Random, Statistics, LinearAlgebra
export CausalModel
export rand

# design choice: kappa(x) or kappa(x...)

"Causal model with `dag` representing the model and `scm` holds the name of the variables and the SCM"
struct CausalModel{U, T, S, R} <: AbstractGraph{T}
    dag::SimpleDiGraph{T}  # Representing the DAG of the causal model
    kernel::S  # Name and function (or distribution) of each variable
    topo::R
end
CausalModel(::Type{U}, dag::SimpleDiGraph{T}, kernel::S) where {U,T,S} = CausalModel{U,T,S,Vector{T}}(dag, kernel, topological_sort_by_dfs(dag))

struct FactorModel{T, S<:AbstractVector}
    graph::SimpleGraph{T}
    laws::S
end

kernel_call(kernel, X, v, J) = kernel[v](X[J]...)

import Random.rand
function rand(rng::Random.AbstractRNG, m::CausalModel{T}) where {T}
    topo = m.topo
    dag = m.dag
    kernel = m.kernel
    X = zeros(T, nv(dag))
    for v in topo
        X[v] = rand(rng, kernel_call(kernel, X, v, inneighbors(dag, v)))
    end
    X
end

end
