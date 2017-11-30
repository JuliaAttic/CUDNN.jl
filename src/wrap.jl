
using Clang, Clang.cindex
# We'll replace some of the wrap functions, and we need to call the following:
import Clang.wrap_c: repr_jl, rep_type, rep_args, name_safe

const HEADER = "/usr/local/cuda-8.0/targets/x86_64-linux/include/cudnn.h"




context=wrap_c.init(output_file="gen_libcudnn.jl",
                    common_file="gen_libcudnn_h.jl",
                    header_library=x->"libcufft",
                    clang_includes=["/usr/include"],
                    header_wrapped=(x,y)->(contains(x,"cuda")))

# context.options = wrap_c.InternalOptions(true)  # wrap structs, too
path = "/usr/local/cuda-8.0/targets/x86_64-linux/include"
headers = [joinpath(path,"cudnn.h")]

# Customize the wrap function for functions. This was copied
# from Clang/src/wrap_c.jl, with the following customizations:
#   - error-check each function that returns a cudaError_t
#   - omit types from function prototypes
function wrap_c.wrap(buf::IO, funcdecl::FunctionDecl, libname::String)
    function print_args(buf::IO, cursors, types)
        i = 1
#        for (c,t) in zip(cursors,types)
#            print(buf, name_safe(c), "::", t)
#            (i < length(cursors)) && print(buf, ", ")
#            i += 1
#        end
        for c in cursors
            print(buf, name_safe(c))
            (i < length(cursors)) && print(buf, ", ")
            i += 1
        end
    end

    cu_spelling = spelling(funcdecl)
    
    funcname = spelling(funcdecl)
    arg_types = cindex.function_args(funcdecl)
    args = [x for x in search(funcdecl, ParmDecl)]
    arg_list = tuple( [repr_jl(x) for x in arg_types]... )
    ret_type = repr_jl(return_type(funcdecl))

    print(buf, "function ")
    print(buf, spelling(funcdecl))
    print(buf, "(")
    print_args(buf, args, [myrepr_jl(x) for x in arg_types])
    println(buf, ")")
    print(buf, "  ")
    ret_type == "cufftResult" && print(buf, "checkerror(")
    print(buf, "ccall( (:", funcname, ", ", libname, "), ")
    print(buf, rep_type(ret_type))
    print(buf, ", ")
    print(buf, rep_args(arg_list), ", ")
    for (i,arg) in enumerate(args)
        print(buf, name_safe(arg))
        (i < length(args)) && print(buf, ", ")
    end
    ret_type == "cufftResult" && print(buf, ")")
    println(buf, ")")
    println(buf, "end")
end

function myrepr_jl(x)
    str = repr_jl(x)
    return (str == "Ptr{Cint}") ? "Array{Cint}" : str
end

wrap_c.wrap_c_headers(context, headers)


# run(wrap_c.init(headers = [HEADER],
#                 # index = None,
#                 # common_file="types.jl",
#                 output_file="libcudnn.jl",
#                 # output_dir = "",
#                 # clang_args = String[],
#                 # clang_includes = ["/usr/lib/clang/3.4.2/include","/usr/local/cuda/include","/usr/include"],
#                 clang_includes = ["/usr/usc/clang/default/include", "/usr/usc/cuda/default/include"],
#                 # clang_diagnostics = true,
#                 header_wrapped=(header,cursorname)->(contains(cursorname,"cudnn")),
#                 header_library=x->"libcudnn",
#                 # header_outputfile = None,
#                 cursor_wrapped=(cursorname,cursor)->!isempty(cursorname),
#                 # options = InternalOptions(),
#                 # rewriter = x->x,
#                 ))
