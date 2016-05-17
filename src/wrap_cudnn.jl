# The following is adapted from CUDArt/gen-6.5/wrap_cuda.jl

using Clang

# The following two likely need to be modified for the host system
includes = ["/usr/include",
            "/usr/lib/clang/3.4.2/include",
            "/ai/opt/cuda/7.5.18/include"]
cudnnpath = "/ai/opt/cudnn/5.0.5/include"
headers = ["cudnn.h"]
headers = [joinpath(cudnnpath,h) for h in headers]

# Customize how functions, constants, and structs are written
const skip_expr = [] # [:(const CUDART_DEVICE = __device__)]
const skip_error_check = [] # [:cudaStreamQuery,:cudaGetLastError,:cudaPeekAtLastError]
function rewriter(ex::Expr)
    if in(ex, skip_expr)
        return :()
    end
    # Empty types get converted to Void
    if ex.head == :type
        a3 = ex.args[3]
        if isempty(a3.args)
            objname = ex.args[2]
            return :(typealias $objname Void)
        end
    end
    ex.head == :function || return ex
    decl, body = ex.args[1], ex.args[2]
    # omit types from function prototypes
    for i = 2:length(decl.args)
        a = decl.args[i]
        if a.head == :(::)
            decl.args[i] = a.args[1]
        end
    end
    # Error-check functions that return a cudnnStatus_t (with some omissions)
    ccallexpr = body.args[1]
    if ccallexpr.head != :ccall
        error("Unexpected body expression: ", body)
    end
    rettype = ccallexpr.args[2]
    if rettype == :cudnnStatus_t
        fname = decl.args[1]
        if !in(fname, skip_error_check)
            body.args[1] = Expr(:call, :cudnnCheck, deepcopy(ccallexpr))
        end
    end
    ex
end

rewriter(A::Array) = [rewriter(a) for a in A]

rewriter(s::Symbol) = string(s)

rewriter(arg) = arg

context=wrap_c.init(output_file="libcudnn.jl",
                    common_file="types.jl",
                    header_library=x->"libcudnn",
                    headers = headers,
                    clang_includes=includes,
                    clang_diagnostics=true,
                    header_wrapped=(header,cursorname)->(contains(cursorname,"cudnn")),
                    cursor_wrapped=(cursorname,cursor)->!isempty(cursorname),
                    rewriter=rewriter)

context.options = wrap_c.InternalOptions(true,true)  # wrap structs, too

# Execute the wrap
run(context)


# DEAD CODE:

# using Clang
# run(wrap_c.init(headers = ["/ai/opt/cudnn/5.0.5/include/cudnn.h"],
#                 # index = None,
#                 common_file="types.jl",
#                 output_file="libcudnn.jl",
#                 # output_dir = "",
#                 # clang_args = ASCIIString[],
#                 clang_includes = ["/usr/lib/clang/3.4.2/include","/ai/opt/cuda/7.5.18/include","/usr/include"],
#                 # clang_includes = ["/usr/usc/clang/default/include", "/usr/usc/cuda/default/include"],
#                 # clang_diagnostics = true,
#                 header_wrapped=(header,cursorname)->(contains(cursorname,"cudnn")),
#                 header_library=x->"libcudnn",
#                 # header_outputfile = None,
#                 cursor_wrapped=(cursorname,cursor)->!isempty(cursorname),
#                 # options = InternalOptions(),
#                 # rewriter = x->x,
#                 ))
