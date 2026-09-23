# Entry point: julia --startup-file=no --project=<worker> bin/worker.jl <core|symbolic>
length(ARGS) == 1 || error("usage: worker.jl <core|symbolic>")
using CIWModelWorker
ARGS[1] == "symbolic" && CIWModelWorker.load_symbolic!()
CIWModelWorker.main(ARGS[1])
