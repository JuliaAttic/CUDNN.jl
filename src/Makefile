all:

libcudnn.jl types.jl: wrap_cudnn.jl
	julia wrap_cudnn.jl

exports.jl: libcudnn.jl types.jl
	echo "# types.jl" > $@
	perl -lne 'print "export $$2" if /(const|typealias|immutable)\s+(\S+)/' types.jl >> $@
	echo "# libcudnn.jl" >> $@
	perl -lne 'print "export $$2" if /(function)\s+([^()\s]+)/' libcudnn.jl >> $@
