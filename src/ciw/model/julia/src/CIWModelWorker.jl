"""
Persistent Julia worker for CIW model operations over the SCR boundary.

Requests carry committed program, configuration and input bytes; the worker
accepts only registered operation descriptors bound to its own runtime digest.
"""
module CIWModelWorker

using LinearAlgebra
using SHA
using ForwardDiff
using OrdinaryDiffEqTsit5: ODEProblem, Tsit5, solve
using SciMLBase: SciMLBase
using ControlSystemsBase: ss, poles, ctrb, obsv
using JuMP: JuMP, Model, @variable, @constraint, @objective, optimize!, termination_status, primal_status,
    objective_value, value, set_silent, set_attribute, MOI
using HiGHS: HiGHS

include("codec.jl")
include("commitments.jl")
include("expressions.jl")
include("descriptors.jl")
include("operations.jl")
include("protocol.jl")

"Load the symbolic profile's packages and operation into this module at runtime."
function load_symbolic!()
    Base.eval(@__MODULE__, :(import ModelingToolkit, Symbolics, Latexify))
    Base.include(@__MODULE__, joinpath(@__DIR__, "symbolic.jl"))
    return nothing
end

end
