# SCR execution commitments (execution.commitments), recomputed by the worker
# from the bytes it actually consumed. The host recomputes them again.

const PROGRAM_TAG = "scout.execution.program.v1"
const INPUT_TAG = "scout.execution.input.v1"
const OUTPUT_TAG = "scout.execution.output.v1"
const COMPUTATION_TAG = "scout.execution.computation.v1"
const SPECIFICATION_TAG = "scout.execution.specification.v1"

function scr_canonical(tag::String, fields::Vector{Vector{UInt8}})
    io = IOBuffer()
    tag_bytes = Vector{UInt8}(codeunits(tag))
    write(io, htol(UInt64(length(tag_bytes))))
    write(io, tag_bytes)
    write(io, htol(UInt64(length(fields))))
    for field in fields
        write(io, htol(UInt64(length(field))))
        write(io, field)
    end
    return take!(io)
end

commit_hex(tag::String, fields::Vector{Vector{UInt8}}) = bytes2hex(sha256(scr_canonical(tag, fields)))

specification_identity(program, configuration, input) =
    commit_hex(SPECIFICATION_TAG, Vector{UInt8}[program, configuration, input])
program_identity(program) = commit_hex(PROGRAM_TAG, Vector{UInt8}[program])
input_identity(input) = commit_hex(INPUT_TAG, Vector{UInt8}[input])
output_identity(output) = commit_hex(OUTPUT_TAG, Vector{UInt8}[output])

function computation_identity(program_id::String, input_id::String, output_id::String, exit_code::Integer)
    return commit_hex(COMPUTATION_TAG, Vector{UInt8}[hex2bytes(program_id), hex2bytes(input_id),
                                                     hex2bytes(output_id), reinterpret(UInt8, [htol(UInt32(exit_code))])])
end
