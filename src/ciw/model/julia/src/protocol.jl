# Framed persistent-worker protocol. Frames on stdout are the only protocol
# data: b"CIWF", u32 LE length, CIWB payload. Diagnostics go to stderr.

const FRAME_MAGIC = UInt8['C', 'I', 'W', 'F']
const MAX_FRAME = 64 * 1024 * 1024
const PROTOCOL = "ciw.julia-model-worker.v1"
const SOURCE_TAG = "ciw.julia-worker.source.v1"
const WORKER_ROOT = normpath(joinpath(@__DIR__, ".."))

file_sha256(path) = isfile(path) ? bytes2hex(open(sha256, path)) : nothing

"Digest of the worker's own code: every src/*.jl file and bin/worker.jl, by relative name."
function worker_source_sha256(root=WORKER_ROOT)
    names = sort(vcat(["bin/worker.jl"], ["src/" * f for f in readdir(joinpath(root, "src")) if endswith(f, ".jl")]))
    fields = Vector{UInt8}[]
    for name in names
        push!(fields, Vector{UInt8}(codeunits(name)))
        push!(fields, read(joinpath(root, name)))
    end
    return commit_hex(SOURCE_TAG, fields)
end

const CORE_PACKAGES = ["ControlSystemsBase", "ForwardDiff", "HiGHS", "JuMP", "OrdinaryDiffEqTsit5", "SciMLBase"]
const SYMBOLIC_PACKAGES = ["Latexify", "ModelingToolkit", "Symbolics"]

function loaded_versions(profile)
    names = profile == "symbolic" ? vcat(CORE_PACKAGES, SYMBOLIC_PACKAGES) : CORE_PACKAGES
    versions = Dict{String,Any}()
    for name in names
        mod = Base.root_module(Base.PkgId(Base.identify_package(name).uuid, name))
        versions[name] = string(pkgversion(mod))
    end
    return versions
end

function runtime_identity(profile)
    image = unsafe_string(Base.JLOptions().image_file)
    blas = LinearAlgebra.BLAS.get_config().loaded_libs
    identity = Dict{String,Any}(
        "protocol" => PROTOCOL, "profile" => profile,
        "julia_version" => string(VERSION), "kernel" => string(Sys.KERNEL), "arch" => string(Sys.ARCH),
        "word_size" => Int64(Sys.WORD_SIZE), "threads" => Int64(Threads.nthreads()),
        "blas_threads" => Int64(LinearAlgebra.BLAS.get_num_threads()),
        "blas_libraries" => [basename(lib.libname) for lib in blas],
        "worker_source_sha256" => worker_source_sha256(),
        "project_sha256" => file_sha256(joinpath(WORKER_ROOT, "Project.toml")),
        "manifest_sha256" => file_sha256(joinpath(WORKER_ROOT, "Manifest.toml")),
        "preferences_sha256" => file_sha256(joinpath(WORKER_ROOT, "LocalPreferences.toml")),
        "packages" => loaded_versions(profile),
        "operations" => Dict{String,Any}(op => bytes2hex(sha256(DESCRIPTORS[op])) for op in PROFILE_OPERATIONS[profile]),
        "julia_executable_sha256" => file_sha256(joinpath(Sys.BINDIR, Base.julia_exename())),
        "sysimage_sha256" => file_sha256(image),
        "cpu_target" => unsafe_string(Base.JLOptions().cpu_target),
    )
    context = Dict{String,Any}("cpu_name" => Sys.CPU_NAME, "image_file" => image, "pid" => Int64(getpid()),
                               "julia_bindir" => Sys.BINDIR)
    return identity, context
end

function read_frame(io)
    header = read(io, 8)
    isempty(header) && return nothing
    length(header) == 8 || error("truncated frame header")
    header[1:4] == FRAME_MAGIC || error("bad frame magic")
    n = Int(ltoh(reinterpret(UInt32, header[5:8])[1]))
    n <= MAX_FRAME || error("frame exceeds $(MAX_FRAME) bytes")
    payload = read(io, n)
    length(payload) == n || error("truncated frame payload")
    return payload
end

function write_frame(io, value)
    payload = ciwb_encode(value)
    length(payload) <= MAX_FRAME || error("response frame exceeds its bound")
    write(io, FRAME_MAGIC)
    write(io, htol(UInt32(length(payload))))
    write(io, payload)
    flush(io)
end

const OPERATIONS = Dict{String,Function}(
    "ciw.model.simulate.v1" => (i, c) -> op_simulate(i, c),
    "ciw.model.linearize.v1" => (i, c) -> op_linearize(i, c),
    "ciw.model.measurement-selection.v1" => (i, c) -> op_measurement_selection(i, c),
    "ciw.model.symbolic.v1" => (i, c) -> Base.invokelatest(getfield(@__MODULE__, :op_symbolic), i, c),
)

function program_bytes(operation, runtime_digest)
    return Vector{UInt8}(codeunits(DESCRIPTORS[operation] * "runtime: sha256:" * runtime_digest * "\n"))
end

function execute(message, profile, runtime_digest)
    operation = message["operation"]
    program::Vector{UInt8} = message["program"]
    configuration::Vector{UInt8} = message["configuration"]
    input::Vector{UInt8} = message["input"]
    spec_id = specification_identity(program, configuration, input)
    response = Dict{String,Any}(
        "type" => "result", "occurrence" => message["occurrence"], "specification_identity" => spec_id,
        "program_identity" => program_identity(program), "input_identity" => input_identity(input),
        "output" => nothing, "output_identity" => nothing, "computation_identity" => nothing,
        "detail" => nothing, "exit_code" => Int64(0))
    if message["specification_identity"] != spec_id
        response["status"] = "unrunnable"
        response["detail"] = "request specification identity does not match its bytes"
        return response
    end
    if !(operation in PROFILE_OPERATIONS[profile]) || program != program_bytes(operation, runtime_digest)
        response["status"] = "unrunnable"
        response["detail"] = "program is not registered in this worker runtime"
        return response
    end
    started = time_ns()
    try
        decoded_input = ciwb_decode(input)
        decoded_config = ciwb_decode(configuration)
        output = ciwb_encode(OPERATIONS[operation](decoded_input, decoded_config))
        out_id = output_identity(output)
        response["status"] = "completed"
        response["output"] = output
        response["output_identity"] = out_id
        response["computation_identity"] = computation_identity(response["program_identity"],
                                                                response["input_identity"], out_id, 0)
    catch err
        code, detail = if err isa Halt
            err.code, err.detail
        elseif err isa MalformedInput || err isa CodecError || err isa KeyError || err isa MethodError ||
               err isa TypeError || err isa ArgumentError || err isa BoundsError
            2, "malformed request: " * sprint(showerror, err)
        elseif err isa DomainError
            4, "domain error: " * sprint(showerror, err)
        else
            1, "internal error: " * sprint(showerror, err)
        end
        response["status"] = "halted"
        response["exit_code"] = Int64(code)
        response["detail"] = first(detail, 2000)
    end
    response["elapsed_ns"] = Int64(time_ns() - started)
    return response
end

function main(profile::String)
    haskey(PROFILE_OPERATIONS, profile) || error("unknown worker profile $(profile)")
    protocol_out = stdout
    redirect_stdout(stderr)  # accidental prints can never become protocol data
    identity, context = runtime_identity(profile)
    runtime_digest = bytes2hex(sha256(ciwb_encode(identity)))
    write_frame(protocol_out, Dict{String,Any}("type" => "hello", "identity" => identity,
                                               "context" => context, "runtime_digest" => runtime_digest))
    expected = 1
    while true
        payload = read_frame(stdin)
        payload === nothing && return
        message = ciwb_decode(payload)
        kind = message["type"]
        kind == "shutdown" && return
        kind == "execute" || error("unknown message type $(kind)")
        message["occurrence"] == expected || error("occurrence $(message["occurrence"]) out of order; expected $(expected)")
        write_frame(protocol_out, execute(message, profile, runtime_digest))
        expected += 1
    end
end
