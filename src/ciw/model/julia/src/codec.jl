# CIWB v1 canonical binary values; byte-for-byte ciw.model.codec.
# Maps have strictly increasing UTF-8 keys; binary64 arrays are row-major.

const CIWB_MAGIC = UInt8['C', 'I', 'W', 'B']
const CIWB_VERSION = UInt16(1)
const CIWB_MAX_DEPTH = 32
const CIWB_MAX_ITEMS = 1 << 22

struct CodecError <: Exception
    message::String
end

"A binary64 array with explicit dimensions and row-major data."
struct F64Array
    dims::Vector{Int}
    data::Vector{Float64}
end

F64Array(v::AbstractVector{<:Real}) = F64Array([length(v)], Float64.(collect(v)))
F64Array(m::AbstractMatrix{<:Real}) = F64Array([size(m, 1), size(m, 2)], Float64.(vec(permutedims(m))))

function as_matrix(a::F64Array)
    length(a.dims) == 2 || throw(CodecError("expected a two-dimensional array"))
    return permutedims(reshape(a.data, a.dims[2], a.dims[1]))
end

function as_vector(a::F64Array)
    length(a.dims) == 1 || throw(CodecError("expected a one-dimensional array"))
    return copy(a.data)
end

function ciwb_encode(value; allow_nonfinite::Bool=false)
    io = IOBuffer()
    write(io, CIWB_MAGIC)
    write(io, htol(CIWB_VERSION))
    encode_value!(io, value, 0, allow_nonfinite)
    return take!(io)
end

function write_len(io, n::Integer)
    0 <= n <= typemax(UInt32) || throw(CodecError("length exceeds u32"))
    write(io, htol(UInt32(n)))
end

function encode_value!(io, value, depth, allow_nonfinite)
    depth > CIWB_MAX_DEPTH && throw(CodecError("value exceeds its nesting bound"))
    if value === nothing
        write(io, 0x00)
    elseif value === false
        write(io, 0x01)
    elseif value === true
        write(io, 0x02)
    elseif value isa Integer
        write(io, 0x03)
        write(io, htol(Int64(value)))
    elseif value isa AbstractFloat
        (allow_nonfinite || isfinite(value)) || throw(CodecError("nonfinite binary64 value"))
        write(io, 0x04)
        write(io, htol(Float64(value)))
    elseif value isa AbstractString
        raw = Vector{UInt8}(codeunits(String(value)))
        write(io, 0x05)
        write_len(io, length(raw))
        write(io, raw)
    elseif value isa Vector{UInt8}
        write(io, 0x06)
        write_len(io, length(value))
        write(io, value)
    elseif value isa F64Array
        (allow_nonfinite || all(isfinite, value.data)) || throw(CodecError("nonfinite binary64 array"))
        1 <= length(value.dims) <= 4 || throw(CodecError("arrays have 1..4 dimensions"))
        prod(value.dims) == length(value.data) || throw(CodecError("array dimensions do not match data"))
        write(io, 0x09)
        write(io, UInt8(length(value.dims)))
        for d in value.dims
            write_len(io, d)
        end
        for x in value.data
            write(io, htol(x))
        end
    elseif value isa AbstractDict
        keys_sorted = sort([(Vector{UInt8}(codeunits(String(k))), String(k)) for k in keys(value)]; by=first)
        write(io, 0x08)
        write_len(io, length(keys_sorted))
        for (raw, key) in keys_sorted
            write_len(io, length(raw))
            write(io, raw)
            encode_value!(io, value[key], depth + 1, allow_nonfinite)
        end
    elseif value isa AbstractVector || value isa Tuple
        write(io, 0x07)
        write_len(io, length(value))
        for item in value
            encode_value!(io, item, depth + 1, allow_nonfinite)
        end
    else
        throw(CodecError("unsupported value type $(typeof(value))"))
    end
end

mutable struct Reader
    raw::Vector{UInt8}
    pos::Int
end

function take_bytes!(r::Reader, n::Integer)
    (n < 0 || r.pos + n - 1 > length(r.raw)) && throw(CodecError("truncated CIWB value"))
    chunk = r.raw[r.pos:r.pos+n-1]
    r.pos += n
    return chunk
end

read_u32(r) = ltoh(reinterpret(UInt32, take_bytes!(r, 4))[1])

function ciwb_decode(raw::Vector{UInt8}; allow_nonfinite::Bool=false)
    length(raw) >= 6 || throw(CodecError("not a CIWB v1 value"))
    raw[1:4] == CIWB_MAGIC || throw(CodecError("not a CIWB v1 value"))
    ltoh(reinterpret(UInt16, raw[5:6])[1]) == CIWB_VERSION || throw(CodecError("unsupported CIWB version"))
    r = Reader(raw, 7)
    value = decode_value!(r, 0, allow_nonfinite)
    r.pos == length(raw) + 1 || throw(CodecError("trailing bytes after CIWB value"))
    return value
end

function decode_value!(r::Reader, depth, allow_nonfinite)
    depth > CIWB_MAX_DEPTH && throw(CodecError("value exceeds its nesting bound"))
    tag = take_bytes!(r, 1)[1]
    if tag == 0x00
        return nothing
    elseif tag == 0x01
        return false
    elseif tag == 0x02
        return true
    elseif tag == 0x03
        return ltoh(reinterpret(Int64, take_bytes!(r, 8))[1])
    elseif tag == 0x04
        x = ltoh(reinterpret(Float64, take_bytes!(r, 8))[1])
        (allow_nonfinite || isfinite(x)) || throw(CodecError("nonfinite binary64 value"))
        return x
    elseif tag == 0x05
        s = String(take_bytes!(r, read_u32(r)))
        isvalid(s) || throw(CodecError("invalid UTF-8 string"))
        return s
    elseif tag == 0x06
        return take_bytes!(r, read_u32(r))
    elseif tag == 0x07
        n = read_u32(r)
        n > CIWB_MAX_ITEMS && throw(CodecError("container exceeds its item bound"))
        return Any[decode_value!(r, depth + 1, allow_nonfinite) for _ in 1:n]
    elseif tag == 0x08
        n = read_u32(r)
        n > CIWB_MAX_ITEMS && throw(CodecError("container exceeds its item bound"))
        result = Dict{String,Any}()
        previous = nothing
        for _ in 1:n
            key_raw = take_bytes!(r, read_u32(r))
            (previous === nothing || key_raw > previous) || throw(CodecError("map keys must be strictly increasing"))
            previous = key_raw
            key = String(copy(key_raw))
            isvalid(key) || throw(CodecError("invalid UTF-8 key"))
            result[key] = decode_value!(r, depth + 1, allow_nonfinite)
        end
        return result
    elseif tag == 0x09
        ndim = Int(take_bytes!(r, 1)[1])
        1 <= ndim <= 4 || throw(CodecError("arrays have 1..4 dimensions"))
        dims = [Int(read_u32(r)) for _ in 1:ndim]
        count = prod(dims)
        count > CIWB_MAX_ITEMS && throw(CodecError("array exceeds its item bound"))
        data = Float64[ltoh(x) for x in reinterpret(Float64, take_bytes!(r, 8 * count))]
        (allow_nonfinite || all(isfinite, data)) || throw(CodecError("nonfinite binary64 array"))
        return F64Array(dims, data)
    end
    throw(CodecError("unknown CIWB tag $(tag)"))
end
