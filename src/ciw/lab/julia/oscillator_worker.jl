# CIW Julia worker, protocol ciw.julia.worker-protocol.v1 (docs/JULIA_SP1.md).
#
# Serves one allowlisted operation, ciw.julia.damped-oscillator.v1: Tsit5
# integration of q' = v, v' = -2*gamma*v - omega_0^2*q with the derived energy
# E = 0.5*mass*(v^2 + omega_0^2*q^2), sampled on the requested time grid.
# Requests carry data only (the SCR specification's program, configuration and
# input bytes); never source, package names or paths. Protocol frames go to
# the original stdout and every diagnostic to stderr. Each request builds fresh
# problem and solver state: compiled code and loaded packages are the only
# state kept between requests. Provisioning (scripts/provision_julia.py)
# instantiates and precompiles the environment; the worker refuses to start on
# an environment that was not precompiled, so it never resolves, downloads or
# compiles packages while serving.
#
# Frames (both directions), little-endian: magic "CIWJ", protocol version u8 = 1,
# kind u8, reserved u16 = 0, request id u64, payload length u32, payload.
# Kinds: 1 handshake (worker), 2 request (host), 3 completed, 4 refused,
# 5 halted (worker), 6 shutdown (host).

const PROTOCOL_OUT = stdout
redirect_stdout(stderr)  # anything printed from here on is a diagnostic, never protocol data

const PACKAGES = [
    Base.PkgId(Base.UUID("b1df2697-797e-41e3-8120-5422d3b24e4a"), "OrdinaryDiffEqTsit5"),
    Base.PkgId(Base.UUID("bbf590c4-e513-4bbe-9b18-05decba2e5d8"), "OrdinaryDiffEqCore"),
    Base.PkgId(Base.UUID("0bca4576-84f4-4d90-8ffe-ffa030f20462"), "SciMLBase"),
    Base.PkgId(Base.UUID("ea8e919c-243c-51af-8825-aaa63cd721ce"), "SHA"),
]
for id in PACKAGES
    # Exit code 70: the bound depot does not hold the provisioned environment (a package missing or not precompiled).
    if !(Base.in_sysimage(id) || (Base.locate_package(id) !== nothing && Base.isprecompiled(id)))
        println(stderr, "ciw-julia-worker: ", id.name, " is not installed and precompiled in the bound depot; ",
                "run scripts/provision_julia.py")
        exit(70)
    end
end

using OrdinaryDiffEqTsit5: ODEProblem, Tsit5, solve
import OrdinaryDiffEqCore
import SciMLBase
import SHA

const MAGIC = b"CIWJ"
const PROTOCOL_VERSION = 0x01
const KIND_HANDSHAKE, KIND_REQUEST, KIND_COMPLETED, KIND_REFUSED, KIND_HALTED, KIND_SHUTDOWN =
    0x01, 0x02, 0x03, 0x04, 0x05, 0x06
const HEADER_BYTES = 20
const MAX_REQUEST_PAYLOAD = 65536
const EXIT_MALFORMED_FRAME, EXIT_FRAME_TOO_LARGE, EXIT_TRUNCATED_FRAME = 65, 66, 67

const OPERATION = "ciw.julia.damped-oscillator.v1"
const INPUT_SCHEMA = "ciw.julia.oscillator-input.v1"
const CONFIGURATION_SCHEMA = "ciw.julia.tsit5-configuration.v1"
const OUTPUT_SCHEMA = "ciw.julia.oscillator-output.v1"
# The program bytes a request must carry, byte for byte (ciw.lab.julia_worker.OPERATION_DESCRIPTOR).
const DESCRIPTOR = Vector{UInt8}(
    "ciw.julia.damped-oscillator.v1\n" *
    "state [q, v] in m and m/s, binary64; q' = v; v' = -2*gamma*v - omega_0^2*q; E = 0.5*mass*(v^2 + omega_0^2*q^2) in J\n" *
    "input: ciw.julia.oscillator-input.v1; configuration: ciw.julia.tsit5-configuration.v1; output: ciw.julia.oscillator-output.v1\n" *
    "solver: OrdinaryDiffEqTsit5.Tsit5() on ODEProblem{true, SciMLBase.AutoSpecialize} over [0, t_last]; adaptive; " *
    "PIController with the configured gains and limits; internalnorm ODE_DEFAULT_NORM; values at the requested " *
    "times from the Tsit5 free interpolant (saveat, save_everystep=false, save_start=true, save_end=true, dense=false)\n" *
    "completed only on ReturnCode.Success with every requested time saved exactly and finite; otherwise halted without output")

# The v1 controller profile: Tsit5's OrdinaryDiffEqCore defaults, made explicit and checked (order 5).
const CONTROLLER = (qmin = 0.2, qmax = 10.0, qmax_first_step = 10000.0, gamma = 0.9, qsteady_min = 1.0,
                    qsteady_max = 1.0, beta1 = 0.14, beta2 = 0.08, qoldinit = 1.0e-4, failfactor = 2.0)

struct Refusal <: Exception
    code::String
    detail::String
end

function le_uint(::Type{T}, b, i) where {T <: Unsigned}
    value = zero(T)
    for k in 0:sizeof(T) - 1
        value |= T(b[i + k]) << (8k)
    end
    return value
end
le_u16(b, i) = le_uint(UInt16, b, i)
le_u32(b, i) = le_uint(UInt32, b, i)
le_u64(b, i) = le_uint(UInt64, b, i)
le_f64(b, i) = reinterpret(Float64, le_u64(b, i))

mutable struct Reader
    bytes::Vector{UInt8}
    at::Int
end
function next_bytes!(r::Reader, n::Integer, what)
    r.at + n - 1 <= length(r.bytes) || throw(Refusal("input_malformed", "$what is truncated"))
    value = r.bytes[r.at:r.at + n - 1]
    r.at += n
    return value
end
f64!(r::Reader, what) = le_f64(next_bytes!(r, 8, what), 1)
u32!(r::Reader, what) = le_u32(next_bytes!(r, 4, what), 1)
u64!(r::Reader, what) = le_u64(next_bytes!(r, 8, what), 1)
function tag!(r::Reader, expected, code)
    n = Int(le_u16(next_bytes!(r, 2, "schema tag length"), 1))
    String(next_bytes!(r, n, "schema tag")) == expected || throw(Refusal(code, "schema tag is not $expected"))
end
finished!(r::Reader, code) = r.at == length(r.bytes) + 1 || throw(Refusal(code, "trailing bytes after the last field"))

function finite!(value, name)
    isfinite(value) || throw(Refusal("nonfinite_number", name))
    return value
end
function within!(value, low, high, name; low_open = false)
    finite!(value, name)
    ok = (low_open ? value > low : value >= low) && value <= high
    ok || throw(Refusal("out_of_bounds", name))
    return value
end

function parse_configuration(bytes)
    r = Reader(bytes, 1)
    tag!(r, CONFIGURATION_SCHEMA, "configuration_malformed")
    abstol = within!(f64!(r, "abstol"), 1.0e-14, 1.0e-2, "abstol")
    reltol = within!(f64!(r, "reltol"), 1.0e-14, 1.0e-2, "reltol")
    dt = within!(f64!(r, "dt"), 0.0, 12.0, "dt"; low_open = true)
    dtmax = within!(f64!(r, "dtmax"), 0.0, 12.0, "dtmax"; low_open = true)
    maxiters = u64!(r, "maxiters")
    1 <= maxiters <= 10_000_000 || throw(Refusal("out_of_bounds", "maxiters"))
    controller = map(name -> finite!(f64!(r, String(name)), String(name)), keys(CONTROLLER))
    finished!(r, "configuration_malformed")
    # Only the declared profile runs: a request cannot choose other controller gains or limits.
    controller == values(CONTROLLER) || throw(Refusal("configuration_unsupported", "controller profile"))
    return (; abstol, reltol, dt, dtmax, maxiters = Int(maxiters))
end

function parse_input(bytes)
    r = Reader(bytes, 1)
    tag!(r, INPUT_SCHEMA, "input_malformed")
    omega0 = within!(f64!(r, "omega_0"), 0.0, 20.0, "omega_0"; low_open = true)
    gamma = within!(f64!(r, "gamma"), 0.0, 0.5 * omega0, "gamma")
    mass = within!(f64!(r, "mass"), 0.0, 100.0, "mass"; low_open = true)
    q0 = within!(f64!(r, "q0"), -10.0, 10.0, "q0")
    v0 = within!(f64!(r, "v0"), -100.0, 100.0, "v0")
    duration = within!(f64!(r, "duration"), 0.0, 12.0, "duration"; low_open = true)
    n = Int(u32!(r, "sample count"))
    2 <= n <= 4096 || throw(Refusal("sample_count_out_of_bounds", "sample count"))
    times = [f64!(r, "time") for _ in 1:n]
    finished!(r, "input_malformed")
    all(isfinite, times) || throw(Refusal("nonfinite_number", "time"))
    times[1] == 0.0 || throw(Refusal("grid_origin", "the first requested time is the initial time 0"))
    all(times[k] < times[k + 1] for k in 1:n - 1) || throw(Refusal("grid_not_increasing", "requested times"))
    times[end] < duration || throw(Refusal("grid_outside_interval", "requested times must lie in [0, duration)"))
    return (; omega0, gamma, mass, q0, v0, duration, times)
end

function oscillator!(du, u, p, t)
    omega0, gamma = p[1], p[2]
    du[1] = u[2]
    du[2] = -2.0 * gamma * u[2] - omega0 * omega0 * u[1]
    return nothing
end

"""(:completed, output bytes) or (:halted, diagnostic text); fresh problem and solver state on every call."""
function run_oscillator(input, config)
    times = input.times
    problem = ODEProblem{true, SciMLBase.AutoSpecialize}(oscillator!, [input.q0, input.v0], (0.0, times[end]),
                                                         [input.omega0, input.gamma])
    controller = OrdinaryDiffEqCore.PIController(Float64, Tsit5(); CONTROLLER...)
    solution = solve(problem, Tsit5(); abstol = config.abstol, reltol = config.reltol, dt = config.dt,
                     dtmax = config.dtmax, maxiters = config.maxiters, adaptive = true, controller = controller,
                     saveat = times, save_everystep = false, save_start = true, save_end = true, dense = false)
    stats = solution.stats
    counts = "naccept $(stats.naccept)\nnreject $(stats.nreject)\nnf $(stats.nf)\nsaved $(length(solution.t))\n"
    if solution.retcode != SciMLBase.ReturnCode.Success
        return :halted, "ciw.julia.worker-halt.v1\nretcode $(solution.retcode)\n" * counts
    end
    if length(solution.t) != length(times) || solution.t != times
        return :halted, "ciw.julia.worker-halt.v1\nretcode incomplete_time_coverage\n" * counts
    end
    q = [u[1] for u in solution.u]
    v = [u[2] for u in solution.u]
    energy = [0.5 * input.mass * (v[k] * v[k] + input.omega0 * input.omega0 * q[k] * q[k]) for k in eachindex(q)]
    if !(all(isfinite, q) && all(isfinite, v) && all(isfinite, energy))
        return :halted, "ciw.julia.worker-halt.v1\nretcode nonfinite_output\n" * counts
    end
    io = IOBuffer()
    write(io, htol(UInt16(ncodeunits(OUTPUT_SCHEMA))), OUTPUT_SCHEMA)
    retcode = string(solution.retcode)
    write(io, htol(UInt16(ncodeunits(retcode))), retcode)
    write(io, htol(UInt64(stats.naccept)), htol(UInt64(stats.nreject)), htol(UInt64(stats.nf)))
    write(io, htol(UInt32(length(times))))
    for series in (times, q, v, energy), value in series
        write(io, htol(value))
    end
    return :completed, take!(io)
end

function write_frame(kind, request_id, payload)
    io = IOBuffer()
    write(io, MAGIC, PROTOCOL_VERSION, kind, htol(UInt16(0)), htol(UInt64(request_id)),
          htol(UInt32(length(payload))), payload)
    write(PROTOCOL_OUT, take!(io))
    flush(PROTOCOL_OUT)
end

refusal_payload(code, detail) = Vector{UInt8}("ciw.julia.worker-refusal.v1\ncode $code\ndetail $detail\n")

function serve(request_id, payload)
    r = Reader(payload, 1)
    fields = Vector{Vector{UInt8}}()
    try
        for name in ("program", "configuration", "input")
            n = u64!(r, "$name length")
            n <= length(payload) || throw(Refusal("request_malformed", "$name length"))
            push!(fields, next_bytes!(r, Int(n), name))
        end
        finished!(r, "request_malformed")
        fields[1] == DESCRIPTOR || throw(Refusal("unknown_operation", "program bytes are not an allowlisted operation"))
        config = parse_configuration(fields[2])
        input = parse_input(fields[3])
        status, result = run_oscillator(input, config)
        if status === :completed
            write_frame(KIND_COMPLETED, request_id, result)
        else
            write_frame(KIND_HALTED, request_id, Vector{UInt8}(result))
        end
    catch err
        err isa Refusal || rethrow()
        write_frame(KIND_REFUSED, request_id, refusal_payload(err.code, err.detail))
    end
end

sha256_hex(bytes) = bytes2hex(SHA.sha256(bytes))

function handshake()
    project = Base.active_project()
    manifest = joinpath(dirname(project), "Manifest.toml")
    image = unsafe_string(Base.JLOptions().image_file)
    packages = String[]
    for (id, mod) in Base.loaded_modules
        path = pathof(mod)
        (path === nothing || id.uuid === nothing || startswith(path, Sys.STDLIB)) && continue
        # <depot>/packages/<package>/<slug>/src/<name>.jl, or .../ext/<name>.jl for an extension
        where_ = join(splitpath(relpath(path, joinpath(DEPOT_PATH[1], "packages"))), "/")
        push!(packages, "$(id.name)=$(id.uuid)=$(pkgversion(mod))=$where_")
    end
    fields = [
        "protocol" => "ciw.julia.worker-protocol.v1",
        "operations" => OPERATION,
        "operation_descriptor_sha256" => sha256_hex(DESCRIPTOR),
        "julia_version" => string(VERSION),
        "julia_commit" => Base.GIT_VERSION_INFO.commit,
        "machine" => Sys.MACHINE,
        "word_size" => string(Sys.WORD_SIZE),
        "executable_sha256" => sha256_hex(read(joinpath(Sys.BINDIR, Base.julia_exename()))),
        "sysimage" => relpath(image, Sys.BINDIR),
        "worker_source_sha256" => sha256_hex(read(@__FILE__)),
        "project" => project,
        "project_sha256" => sha256_hex(read(project)),
        "manifest_sha256" => sha256_hex(read(manifest)),
        "local_preferences" => string(isfile(joinpath(dirname(project), "LocalPreferences.toml"))),
        "depot" => DEPOT_PATH[1],
        "depot_count" => string(length(DEPOT_PATH)),
        "load_path" => join(LOAD_PATH, "|"),
        "packages" => join(sort(packages), ";"),
        "threads" => string(Threads.nthreads()),
        "interactive_threads" => string(Threads.nthreads(:interactive)),
        "rounding" => string(rounding(Float64)),
        "zero_subnormals" => string(get_zero_subnormals()),
        "opt_level" => string(Base.JLOptions().opt_level),
        "check_bounds" => string(Base.JLOptions().check_bounds),
        "fast_math" => string(Base.JLOptions().fast_math),
        "cpu_target" => Base.JLOptions().cpu_target == C_NULL ? "native" : unsafe_string(Base.JLOptions().cpu_target),
        "cpu_name" => Sys.CPU_NAME,
        "controller_profile" => join(("$k=$v" for (k, v) in pairs(CONTROLLER)), ","),
    ]
    text = "ciw.julia.worker-handshake.v1\n" * join(("$k $v" for (k, v) in sort(fields; by = first)), "\n") * "\n"
    return Vector{UInt8}(text)
end

function main()
    write_frame(KIND_HANDSHAKE, 0, handshake())
    header = Vector{UInt8}(undef, HEADER_BYTES)
    while true
        got = readbytes!(stdin, header, HEADER_BYTES)
        got == 0 && return 0                      # the host closed the channel between frames
        got < HEADER_BYTES && return EXIT_TRUNCATED_FRAME
        request_id = le_u64(header, 9)
        length_ = le_u32(header, 17)
        if header[1:4] != MAGIC || header[5] != PROTOCOL_VERSION || le_u16(header, 7) != 0 ||
           !(header[6] in (KIND_REQUEST, KIND_SHUTDOWN))
            write_frame(KIND_REFUSED, request_id, refusal_payload("malformed_frame", "header"))
            return EXIT_MALFORMED_FRAME
        end
        header[6] == KIND_SHUTDOWN && return 0
        if length_ > MAX_REQUEST_PAYLOAD
            # The payload is not read: a frame over the limit ends the session.
            write_frame(KIND_REFUSED, request_id, refusal_payload("frame_too_large", "payload over $MAX_REQUEST_PAYLOAD bytes"))
            return EXIT_FRAME_TOO_LARGE
        end
        payload = Vector{UInt8}(undef, length_)
        readbytes!(stdin, payload, length_) == length_ || return EXIT_TRUNCATED_FRAME
        serve(request_id, payload)
    end
end

exit(main())
