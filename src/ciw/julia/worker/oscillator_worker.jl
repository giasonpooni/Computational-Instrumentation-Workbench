# CIW persistent Julia oscillator worker, protocol version 1.
#
# The host owns this process. It starts it with an explicit executable, project
# and thread count, reads one hello frame, then sends framed binary requests one
# at a time. Requests carry data only: an allowlisted operation name and the
# exact SCR configuration and input bytes of an execution specification. No
# Julia source, package name or executable path is ever read from a request.
#
# Every frame is [u32 little-endian payload length][payload]. Protocol data uses
# stdout only; every diagnostic goes to stderr so compilation or package output
# can never become protocol bytes. A malformed frame ends the process.
#
# Fresh problem and solver state are built for every request. Compiled code and
# loaded packages are reused; trajectory state and numerical buffers are not.

using SHA
using TOML
using OrdinaryDiffEqTsit5
using SciMLBase

const PROTOCOL_VERSION = UInt32(1)
const HELLO_SCHEMA = "ciw.julia-worker-hello.v1"
const OPERATION = "ciw.julia-oscillator.integrate.v1"
const REQUEST_MAGIC = b"CIWQ"
const CONFIGURATION_MAGIC = b"OSCC"
const INPUT_MAGIC = b"OSCI"
const OUTPUT_MAGIC = b"OSCO"
const MAX_REQUEST_BYTES = 1 << 20
const MAX_SAMPLES = 4096
const MIN_SAMPLES = 2
const MAX_DURATION_S = 12.0
const IDENTIFIED_PACKAGES = ("OrdinaryDiffEqTsit5", "OrdinaryDiffEqCore", "DiffEqBase", "SciMLBase", "SHA", "TOML")

# Worker status codes; 0 is the only completed status.
const STATUS_COMPLETED = UInt32(0)
const STATUS_MALFORMED = UInt32(2)
const STATUS_UNSUPPORTED = UInt32(3)
const STATUS_BOUND = UInt32(4)
const STATUS_SOLVER = UInt32(5)
const STATUS_NONFINITE = UInt32(6)
const STATUS_COVERAGE = UInt32(7)

struct Refusal <: Exception
    status::UInt32
    message::String
end

# --- byte cursor -----------------------------------------------------------

mutable struct Cursor
    bytes::Vector{UInt8}
    position::Int
end

function take_bytes!(cursor::Cursor, count::Integer)
    stop = cursor.position + count - 1
    stop > length(cursor.bytes) && throw(Refusal(STATUS_MALFORMED, "frame ended inside a field"))
    slice = cursor.bytes[cursor.position:stop]
    cursor.position = stop + 1
    slice
end

take_u8!(cursor::Cursor) = take_bytes!(cursor, 1)[1]
take_u32!(cursor::Cursor) = ltoh(reinterpret(UInt32, take_bytes!(cursor, 4))[1])
take_u64!(cursor::Cursor) = ltoh(reinterpret(UInt64, take_bytes!(cursor, 8))[1])
take_f64!(cursor::Cursor) = reinterpret(Float64, ltoh(reinterpret(UInt64, take_bytes!(cursor, 8))[1]))

function take_string!(cursor::Cursor, limit::Integer)
    count = take_u32!(cursor)
    count > limit && throw(Refusal(STATUS_MALFORMED, "string field exceeds its bound"))
    text = String(take_bytes!(cursor, Int(count)))
    isvalid(text) && all(c -> ' ' <= c <= '~', text) || throw(Refusal(STATUS_MALFORMED, "string field must be printable ASCII"))
    text
end

function take_f64s!(cursor::Cursor, count::Integer)
    raw = take_bytes!(cursor, 8 * count)
    values = Vector{Float64}(undef, count)
    for index in 1:count
        values[index] = reinterpret(Float64, ltoh(reinterpret(UInt64, raw[8 * index - 7:8 * index])[1]))
    end
    values
end

function finished!(cursor::Cursor, what::String)
    cursor.position == length(cursor.bytes) + 1 || throw(Refusal(STATUS_MALFORMED, what * " has trailing bytes"))
end

# --- encoders --------------------------------------------------------------

put_u32!(out::IO, value::Integer) = write(out, htol(UInt32(value)))
put_u64!(out::IO, value::Integer) = write(out, htol(UInt64(value)))
put_f64!(out::IO, value::Float64) = write(out, htol(reinterpret(UInt64, value)))

function put_string!(out::IO, text::String)
    put_u32!(out, ncodeunits(text))
    write(out, codeunits(text))
end

function put_f64s!(out::IO, values::Vector{Float64})
    for value in values
        put_f64!(out, value)
    end
end

# --- contract --------------------------------------------------------------

struct Configuration
    algorithm::String
    abstol::Float64
    reltol::Float64
    dt_initial::Float64
    dtmax::Float64
    maxiters::UInt64
    adaptive::UInt8
    output_policy::UInt8
end

struct Input
    omega_0::Float64
    gamma::Float64
    mass::Float64
    q0::Float64
    v0::Float64
    times::Vector{Float64}
end

bounded(value::Float64, low::Float64, high::Float64) = isfinite(value) && low <= value <= high

function parse_configuration(bytes::Vector{UInt8})
    cursor = Cursor(bytes, 1)
    take_bytes!(cursor, 4) == CONFIGURATION_MAGIC || throw(Refusal(STATUS_MALFORMED, "configuration magic mismatch"))
    take_u32!(cursor) == 1 || throw(Refusal(STATUS_MALFORMED, "unsupported configuration version"))
    algorithm = take_string!(cursor, 64)
    abstol = take_f64!(cursor)
    reltol = take_f64!(cursor)
    dt_initial = take_f64!(cursor)
    dtmax = take_f64!(cursor)
    maxiters = take_u64!(cursor)
    adaptive = take_u8!(cursor)
    output_policy = take_u8!(cursor)
    finished!(cursor, "configuration")
    algorithm == "Tsit5" || throw(Refusal(STATUS_UNSUPPORTED, "only the Tsit5 profile is registered"))
    bounded(abstol, 1e-14, 1e-2) || throw(Refusal(STATUS_BOUND, "abstol outside [1e-14, 1e-2]"))
    bounded(reltol, 1e-14, 1e-2) || throw(Refusal(STATUS_BOUND, "reltol outside [1e-14, 1e-2]"))
    (dt_initial == 0.0 || bounded(dt_initial, 1e-9, MAX_DURATION_S)) || throw(Refusal(STATUS_BOUND, "dt_initial must be 0 or within [1e-9, 12]"))
    (dtmax == 0.0 || bounded(dtmax, 1e-9, MAX_DURATION_S)) || throw(Refusal(STATUS_BOUND, "dtmax must be 0 or within [1e-9, 12]"))
    1000 <= maxiters <= 100_000_000 || throw(Refusal(STATUS_BOUND, "maxiters outside [1000, 1e8]"))
    adaptive == 1 || throw(Refusal(STATUS_UNSUPPORTED, "only adaptive stepping is registered"))
    output_policy in (0x01, 0x02) || throw(Refusal(STATUS_UNSUPPORTED, "unknown output policy"))
    Configuration(algorithm, abstol, reltol, dt_initial, dtmax, maxiters, adaptive, output_policy)
end

function parse_input(bytes::Vector{UInt8})
    cursor = Cursor(bytes, 1)
    take_bytes!(cursor, 4) == INPUT_MAGIC || throw(Refusal(STATUS_MALFORMED, "input magic mismatch"))
    take_u32!(cursor) == 1 || throw(Refusal(STATUS_MALFORMED, "unsupported input version"))
    omega_0 = take_f64!(cursor)
    gamma = take_f64!(cursor)
    mass = take_f64!(cursor)
    q0 = take_f64!(cursor)
    v0 = take_f64!(cursor)
    count = take_u32!(cursor)
    MIN_SAMPLES <= count <= MAX_SAMPLES || throw(Refusal(STATUS_BOUND, "sample count outside [2, 4096]"))
    times = take_f64s!(cursor, Int(count))
    finished!(cursor, "input")
    bounded(omega_0, nextfloat(0.0), 20.0) && omega_0 > 0.0 || throw(Refusal(STATUS_BOUND, "omega_0 outside (0, 20] rad/s"))
    bounded(gamma, 0.0, 0.5 * omega_0) || throw(Refusal(STATUS_BOUND, "gamma outside [0, 0.5*omega_0]"))
    bounded(mass, nextfloat(0.0), 100.0) && mass > 0.0 || throw(Refusal(STATUS_BOUND, "mass outside (0, 100] kg"))
    bounded(q0, -10.0, 10.0) || throw(Refusal(STATUS_BOUND, "abs(q0) above 10 m"))
    bounded(v0, -100.0, 100.0) || throw(Refusal(STATUS_BOUND, "abs(v0) above 100 m/s"))
    all(isfinite, times) || throw(Refusal(STATUS_BOUND, "sample times must be finite"))
    times[1] >= 0.0 || throw(Refusal(STATUS_BOUND, "sample times start before the initial state"))
    times[end] <= MAX_DURATION_S || throw(Refusal(STATUS_BOUND, "sample times exceed 12 s"))
    all(times[index] < times[index + 1] for index in 1:count - 1) || throw(Refusal(STATUS_BOUND, "sample times must be strictly increasing"))
    Input(omega_0, gamma, mass, q0, v0, times)
end

function oscillator!(du, u, p, t)
    du[1] = u[2]
    du[2] = -p[2] * u[2] - p[1] * u[1]
end

function integrate(configuration::Configuration, input::Input)
    parameters = (input.omega_0 * input.omega_0, 2.0 * input.gamma)
    problem = ODEProblem(oscillator!, [input.q0, input.v0], (0.0, input.times[end]), parameters)
    options = Dict{Symbol, Any}(
        :abstol => configuration.abstol, :reltol => configuration.reltol,
        :maxiters => Int(configuration.maxiters), :adaptive => true,
        :saveat => input.times, :save_everystep => false, :dense => false,
        :save_start => true, :save_end => true,
    )
    configuration.dt_initial > 0.0 && (options[:dt] = configuration.dt_initial)
    configuration.dtmax > 0.0 && (options[:dtmax] = configuration.dtmax)
    configuration.output_policy == 0x02 && (options[:tstops] = input.times)
    solution = solve(problem, Tsit5(); options...)
    code = string(solution.retcode)
    SciMLBase.successful_retcode(solution) || throw(Refusal(STATUS_SOLVER, "solver return code " * code))
    solution.t == input.times || throw(Refusal(STATUS_COVERAGE, "saved times differ from the requested grid"))
    count = length(input.times)
    q = Vector{Float64}(undef, count)
    v = Vector{Float64}(undef, count)
    energy = Vector{Float64}(undef, count)
    for index in 1:count
        state = solution.u[index]
        q[index] = state[1]
        v[index] = state[2]
        # Same operation order as the Python analytical reference.
        energy[index] = 0.5 * input.mass * (v[index] * v[index] + input.omega_0 * input.omega_0 * q[index] * q[index])
    end
    (all(isfinite, q) && all(isfinite, v) && all(isfinite, energy)) || throw(Refusal(STATUS_NONFINITE, "solver produced a nonfinite value"))
    stats = solution.stats
    (code, q, v, energy, UInt64(stats.naccept), UInt64(stats.nreject), UInt64(stats.nf))
end

# --- environment identity -------------------------------------------------

function file_sha256(path::AbstractString)
    open(path, "r") do stream
        bytes2hex(sha256(stream))
    end
end

function json(value)::String
    if value === nothing
        return "null"
    elseif value isa Bool
        return value ? "true" : "false"
    elseif value isa Integer
        return string(value)
    elseif value isa AbstractString
        buffer = IOBuffer()
        write(buffer, '"')
        for c in value
            if c == '"'
                write(buffer, "\\\"")
            elseif c == '\\'
                write(buffer, "\\\\")
            elseif c < ' '
                write(buffer, "\\u" * lpad(string(UInt32(c), base = 16), 4, '0'))
            else
                write(buffer, c)
            end
        end
        write(buffer, '"')
        return String(take!(buffer))
    elseif value isa AbstractVector
        return "[" * join((json(item) for item in value), ",") * "]"
    elseif value isa AbstractDict
        keys_sorted = sort!(collect(String.(keys(value))))
        return "{" * join((json(k) * ":" * json(value[k]) for k in keys_sorted), ",") * "}"
    end
    error("unsupported hello value")
end

function package_identities(manifest_path::AbstractString)
    manifest = TOML.parsefile(manifest_path)
    entries = manifest["deps"]
    packages = Dict{String, Any}()
    for name in IDENTIFIED_PACKAGES
        haskey(entries, name) || error("manifest lacks " * name)
        entry = entries[name][1]
        packages[name] = Dict{String, Any}(
            "uuid" => entry["uuid"], "version" => entry["version"],
            "git_tree_sha1" => get(entry, "git-tree-sha1", nothing))
    end
    packages
end

function hello()
    options = Base.JLOptions()
    project = Base.active_project()
    manifest = joinpath(dirname(project), "Manifest.toml")
    packages = package_identities(manifest)
    loaded = Dict("OrdinaryDiffEqTsit5" => string(pkgversion(OrdinaryDiffEqTsit5)), "SciMLBase" => string(pkgversion(SciMLBase)))
    for (name, version) in loaded
        packages[name]["version"] == version || error("loaded " * name * " differs from the manifest")
    end
    image = unsafe_string(options.image_file)
    Dict{String, Any}(
        "schema" => HELLO_SCHEMA,
        "protocol_version" => Int(PROTOCOL_VERSION),
        "operations" => [OPERATION],
        "julia_version" => string(VERSION),
        "platform" => Dict{String, Any}("machine" => Sys.MACHINE, "arch" => string(Sys.ARCH),
            "kernel" => string(Sys.KERNEL), "word_size" => Sys.WORD_SIZE),
        "threads" => Threads.nthreads(),
        "options" => Dict{String, Any}("startup_file" => options.startupfile == 2 ? "disabled" : "enabled",
            "fast_math" => options.fast_math == 0 ? "default" : (options.fast_math == 1 ? "on" : "off"),
            "check_bounds" => Int(options.check_bounds), "opt_level" => Int(options.opt_level)),
        "worker_source_sha256" => file_sha256(PROGRAM_FILE),
        "project_sha256" => file_sha256(project),
        "manifest_sha256" => file_sha256(manifest),
        "sysimage" => Dict{String, Any}("name" => basename(image), "sha256" => file_sha256(image)),
        "packages" => packages,
        "solver" => Dict{String, Any}("algorithm" => "Tsit5", "package" => "OrdinaryDiffEqTsit5",
            "version" => loaded["OrdinaryDiffEqTsit5"], "arithmetic" => "binary64"),
    )
end

# --- framing ---------------------------------------------------------------

function read_exact(stream::IO, count::Int)
    bytes = read(stream, count)
    length(bytes) == count || return nothing
    bytes
end

function write_frame(stream::IO, payload::Vector{UInt8})
    put_u32!(stream, length(payload))
    write(stream, payload)
    flush(stream)
end

function refusal_payload(request_number::UInt64, status::UInt32, message::String)
    buffer = IOBuffer()
    write(buffer, OUTPUT_MAGIC)
    put_u32!(buffer, 1)
    put_u64!(buffer, request_number)
    put_u32!(buffer, status)
    put_string!(buffer, message)
    take!(buffer)
end

function completed_payload(request_number::UInt64, code::String, times, q, v, energy, naccept, nreject, nf)
    buffer = IOBuffer()
    write(buffer, OUTPUT_MAGIC)
    put_u32!(buffer, 1)
    put_u64!(buffer, request_number)
    put_u32!(buffer, STATUS_COMPLETED)
    put_string!(buffer, code)
    put_u32!(buffer, length(times))
    put_f64s!(buffer, times)
    put_f64s!(buffer, q)
    put_f64s!(buffer, v)
    put_f64s!(buffer, energy)
    put_u64!(buffer, naccept)
    put_u64!(buffer, nreject)
    put_u64!(buffer, nf)
    take!(buffer)
end

function serve_request(frame::Vector{UInt8}, occurrence::Int)
    cursor = Cursor(frame, 1)
    take_bytes!(cursor, 4) == REQUEST_MAGIC || throw(Refusal(STATUS_MALFORMED, "request magic mismatch"))
    take_u32!(cursor) == PROTOCOL_VERSION || throw(Refusal(STATUS_MALFORMED, "unsupported request version"))
    request_number = take_u64!(cursor)
    operation = take_string!(cursor, 128)
    configuration_length = take_u32!(cursor)
    configuration_bytes = take_bytes!(cursor, Int(configuration_length))
    input_length = take_u32!(cursor)
    input_bytes = take_bytes!(cursor, Int(input_length))
    finished!(cursor, "request")
    request_number == occurrence || throw(Refusal(STATUS_MALFORMED, "request number is not the next occurrence"))
    started = time_ns()
    payload = try
        operation == OPERATION || throw(Refusal(STATUS_UNSUPPORTED, "operation is not registered on this worker"))
        configuration = parse_configuration(configuration_bytes)
        input = parse_input(input_bytes)
        code, q, v, energy, naccept, nreject, nf = integrate(configuration, input)
        completed_payload(request_number, code, input.times, q, v, energy, naccept, nreject, nf)
    catch exception
        exception isa Refusal || rethrow()
        refusal_payload(request_number, exception.status, exception.message)
    end
    elapsed = (time_ns() - started) / 1e9
    metadata = json(Dict{String, Any}("schema" => "ciw.julia-worker-occurrence.v1",
        "request_number" => Int(request_number), "worker_occurrence" => occurrence,
        "elapsed_ns" => Int(time_ns() - started)))
    payload, Vector{UInt8}(codeunits(metadata))
end

function main()
    stdout_stream = stdout
    stdin_stream = stdin
    # Warm the compiled solve path before signalling readiness. No trajectory
    # state survives this call; it is not an occurrence and returns nothing.
    integrate(Configuration("Tsit5", 1e-8, 1e-8, 0.0, 0.0, UInt64(100_000), 0x01, 0x01),
              Input(1.0, 0.0, 1.0, 1.0, 0.0, [0.0, 0.5, 1.0]))
    write_frame(stdout_stream, Vector{UInt8}(codeunits(json(hello()))))
    occurrence = 0
    while true
        header = read_exact(stdin_stream, 4)
        header === nothing && return 0
        count = Int(ltoh(reinterpret(UInt32, header)[1]))
        if count > MAX_REQUEST_BYTES
            println(stderr, "ciw julia worker: request frame exceeds the bound; ending the session")
            return 3
        end
        frame = read_exact(stdin_stream, count)
        if frame === nothing
            println(stderr, "ciw julia worker: request frame truncated; ending the session")
            return 3
        end
        occurrence += 1
        payload, metadata = try
            serve_request(frame, occurrence)
        catch exception
            if exception isa Refusal
                println(stderr, "ciw julia worker: malformed request; ending the session: ", exception.message)
                return 3
            end
            rethrow()
        end
        write_frame(stdout_stream, payload)
        write_frame(stdout_stream, metadata)
    end
end

exit(main())
