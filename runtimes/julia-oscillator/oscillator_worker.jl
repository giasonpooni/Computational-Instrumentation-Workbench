#!/usr/bin/env julia
"""Framed, data-only CIW Julia oscillator worker.

Stdout is a length-prefixed JSON protocol.  Diagnostics and package messages
remain on stderr.  Requests contain only the fixed oscillator data contract;
they can never contain Julia source, package names, or executable paths.
"""
module CIWJuliaOscillatorWorker

using JSON3
using LinearAlgebra
using OrdinaryDiffEqTsit5
using SciMLBase
using SHA

const OPERATION = "ciw.julia-oscillator.v1"
const REQUEST_SCHEMA = "ciw.julia-oscillator-request.v1"
const RESULT_SCHEMA = "ciw.julia-oscillator-result.v1"
const RESPONSE_SCHEMA = "ciw.julia-worker-response.v1"
const MAX_FRAME = 4 * 1024 * 1024
const MAX_SAMPLES = 4096

function keys_exact(value, expected)
    value isa AbstractDict && Set(String.(keys(value))) == Set(expected) ||
        throw(ArgumentError("request fields do not match the fixed allowlist"))
end

function number(value, lo, hi)
    value isa Real && !(value isa Bool) && isfinite(value) && lo <= value <= hi ||
        throw(ArgumentError("number outside declared finite bounds"))
    Float64(value)
end

function validate(request)
    keys_exact(request, ["schema", "operation_id", "model", "initial_state", "time_s", "solver"])
    request["schema"] == REQUEST_SCHEMA && request["operation_id"] == OPERATION ||
        throw(ArgumentError("unsupported operation or schema"))
    model = request["model"]
    keys_exact(model, ["omega_0_rad_s", "gamma_s_inv", "mass_kg"])
    omega = number(model["omega_0_rad_s"], eps(Float64), 20.0)
    gamma = number(model["gamma_s_inv"], 0.0, 10.0)
    number(model["mass_kg"], eps(Float64), 100.0)
    gamma <= 0.5 * omega || throw(ArgumentError("profile requires underdamped or undamped dynamics"))
    initial = request["initial_state"]
    keys_exact(initial, ["q0_m", "v0_m_s"])
    number(initial["q0_m"], -10.0, 10.0)
    number(initial["v0_m_s"], -100.0, 100.0)
    times = request["time_s"]
    times isa AbstractVector && 2 <= length(times) <= MAX_SAMPLES || throw(ArgumentError("time grid exceeds bounds"))
    previous = -1.0
    for item in times
        current = number(item, 0.0, 12.0)
        current > previous || throw(ArgumentError("time grid must be strictly increasing"))
        previous = current
    end
    times[1] == 0.0 || throw(ArgumentError("time grid must start at zero"))
    solver = request["solver"]
    keys_exact(solver, ["abstol", "reltol", "maxiters"])
    number(solver["abstol"], 1e-14, 1e-3)
    number(solver["reltol"], 1e-14, 1e-3)
    solver["maxiters"] isa Integer && !(solver["maxiters"] isa Bool) && 1 <= solver["maxiters"] <= 10_000_000 ||
        throw(ArgumentError("maxiters outside bounds"))
    nothing
end

function solve_request(request, request_id)
    validate(request)
    omega = Float64(request["model"]["omega_0_rad_s"])
    gamma = Float64(request["model"]["gamma_s_inv"])
    mass = Float64(request["model"]["mass_kg"])
    initial = Float64[request["initial_state"]["q0_m"], request["initial_state"]["v0_m_s"]]
    times = Float64.(request["time_s"])
    f!(du, u, _, t) = begin
        du[1] = u[2]
        du[2] = -2.0 * gamma * u[2] - omega^2 * u[1]
    end
    problem = ODEProblem(f!, initial, (times[1], times[end]))
    solution = solve(problem, Tsit5(); abstol=Float64(request["solver"]["abstol"]),
        reltol=Float64(request["solver"]["reltol"]), maxiters=Int(request["solver"]["maxiters"]),
        saveat=times, dense=false, save_everystep=false)
    solution.retcode == ReturnCode.Success || throw(ArgumentError("Tsit5 returned $(solution.retcode)"))
    length(solution.u) == length(times) || throw(ArgumentError("solver did not cover the requested grid"))
    q = [Float64(state[1]) for state in solution.u]
    v = [Float64(state[2]) for state in solution.u]
    energy = [0.5 * mass * (velocity^2 + omega^2 * position^2) for (position, velocity) in zip(q, v)]
    all(isfinite, q) && all(isfinite, v) && all(isfinite, energy) || throw(ArgumentError("solver returned nonfinite values"))
    accepted = try Int(solution.destats.naccept) catch; 0 end
    rejected = try Int(solution.destats.nreject) catch; 0 end
    Dict("schema" => RESULT_SCHEMA, "operation_id" => OPERATION, "request_id" => request_id,
        "time_s" => times, "q_m" => q, "v_m_s" => v, "energy_j" => energy,
        "solver" => Dict("algorithm" => "Tsit5", "retcode" => "Success",
            "abstol" => request["solver"]["abstol"], "reltol" => request["solver"]["reltol"],
            "accepted_steps" => accepted, "rejected_steps" => rejected))
end

function identity()
    project = joinpath(@__DIR__, "Project.toml")
    manifest = joinpath(@__DIR__, "Manifest.toml")
    Dict("schema" => "ciw.julia-worker-identity.v1", "profile" => "ordinarydiffeqtsit5",
        "operation_id" => OPERATION, "julia_version" => string(VERSION),
        "platform" => string(Sys.MACHINE), "threads" => Threads.nthreads(),
        "startup_file" => "disabled", "worker_sha256" => "sha256:" * bytes2hex(sha256(read(@__FILE__))),
        "project_sha256" => "sha256:" * bytes2hex(sha256(read(project))),
        "manifest_sha256" => "sha256:" * bytes2hex(sha256(read(manifest))))
end

function read_frame(io)
    header = read(io, UInt8, 4)
    isempty(header) && return nothing
    length(header) == 4 || throw(ArgumentError("truncated frame header"))
    length = (Int(header[1]) << 24) | (Int(header[2]) << 16) | (Int(header[3]) << 8) | Int(header[4])
    0 <= length <= MAX_FRAME || throw(ArgumentError("oversized frame"))
    read(io, UInt8, length)
end

function write_frame(io, value)
    bytes = Vector{UInt8}(codeunits(JSON3.write(value)))
    length(bytes) <= MAX_FRAME || throw(ArgumentError("oversized response"))
    write(io, UInt8[(length >> 24) & 0xff, (length >> 16) & 0xff, (length >> 8) & 0xff, length & 0xff])
    write(io, bytes)
    flush(io)
end

function main()
    BLAS.set_num_threads(1)
    first = read_frame(stdin)
    first === nothing && return
    handshake = JSON3.read(String(first), Dict{String,Any})
    keys_exact(handshake, ["schema", "request_id", "operation_id"])
    handshake["schema"] == "ciw.julia-worker-handshake-request.v1" && handshake["operation_id"] == OPERATION ||
        throw(ArgumentError("invalid handshake"))
    write_frame(stdout, Dict("schema" => "ciw.julia-worker-handshake-response.v1", "status" => "ok",
        "request_id" => handshake["request_id"], "operation_id" => OPERATION, "identity" => identity()))
    while true
        raw = read_frame(stdin)
        raw === nothing && break
        request = JSON3.read(String(raw), Dict{String,Any})
        response = try
            keys_exact(request, ["schema", "operation_id", "model", "initial_state", "time_s", "solver", "request_id"])
            data = solve_request(request, String(request["request_id"]))
            Dict("schema" => RESPONSE_SCHEMA, "status" => "ok", "request_id" => request["request_id"],
                "operation_id" => OPERATION, "data" => data)
        catch error
            println(stderr, "CIW Julia oscillator refusal: ", sprint(showerror, error))
            Dict("schema" => RESPONSE_SCHEMA, "status" => "refused", "request_id" => get(request, "request_id", "unknown"),
                "operation_id" => OPERATION, "refusal" => Dict("code" => "INVALID_REQUEST_OR_NUMERICAL_FAILURE", "message" => sprint(showerror, error)))
        end
        write_frame(stdout, response)
    end
end

end # module

if abspath(PROGRAM_FILE) == @__FILE__
    CIWJuliaOscillatorWorker.main()
end
