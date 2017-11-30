using CUDNN
using CuArrays
using NNlib
using Base.Test


cu_isapprox(a, ca; kw...) = isapprox(CuArray(a), ca; kw...)


@testset "conv" begin

    for T in [Float32, Float64]
        x = rand(T, 7, 7, 3, 2)
        w = rand(T, 3, 3, 3, 2)
        cx = CuArray(x)
        cw = CuArray(w)

        @test cu_isapprox(conv2d(x, w), conv2d(cx, cw))
        @test cu_isapprox(conv2d(x, w; stride=2), conv2d(cx, cw; stride=2))
        @test cu_isapprox(conv2d(x, w; padding=1), conv2d(cx, cw; padding=1))
        @test cu_isapprox(conv2d(x, w; mode=1), conv2d(cx, cw; mode=1))

        y = conv2d(x, w)
        dy = randn(T, size(y))
        cy = CuArray(y)
        cdy = CuArray(dy)
        @test cu_isapprox(conv2d_grad_x(x, w, dy), conv2d_grad_x(cx, cw, cdy); atol=1e-5)
        @test cu_isapprox(conv2d_grad_w(x, w, dy), conv2d_grad_w(cx, cw, cdy); atol=1e-5)

        # with stride
        y = conv2d(x, w; stride=2)
        dy = randn(T, size(y))
        cy = CuArray(y)
        cdy = CuArray(dy)
        @test cu_isapprox(conv2d_grad_x(x, w, dy; stride=2),
                       conv2d_grad_x(cx, cw, cdy; stride=2); atol=1e-5)
        @test cu_isapprox(conv2d_grad_w(x, w, dy; stride=2),
                       conv2d_grad_w(cx, cw, cdy; stride=2); atol=1e-5)

        # with padding
        y = conv2d(x, w; padding=1)
        dy = randn(T, size(y))
        cy = CuArray(y)
        cdy = CuArray(dy)
        @test cu_isapprox(conv2d_grad_x(x, w, dy; padding=1),
                       conv2d_grad_x(cx, cw, cdy; padding=1); atol=1e-5)
        @test cu_isapprox(conv2d_grad_w(x, w, dy; padding=1),
                       conv2d_grad_w(cx, cw, cdy; padding=1); atol=1e-5)
    end

end


@testset "pool" begin

    for T in [Float32, Float64]
        x = rand(T, 8, 8, 3, 2)
        cx = CuArray(x)

        @test cu_isapprox(pool(x), pool(cx))
        @test cu_isapprox(pool(x; stride=2), pool(cx; stride=2))
        @test cu_isapprox(pool(x; padding=1), pool(cx; padding=1))

        y = pool(x)
        dy = randn(T, size(y))
        cy = CuArray(y)
        cdy = CuArray(dy)
        @test cu_isapprox(pool_grad(x, y, dy), pool_grad(cx, cy, cdy); atol=1e-5)

        # with stride
        y = pool(x; stride=2)
        dy = randn(T, size(y))
        cy = CuArray(y)
        cdy = CuArray(dy)
        @test cu_isapprox(pool_grad(x, y, dy; stride=2), pool_grad(cx, cy, cdy; stride=2))

        # with padding
        y = pool(x; padding=1)
        dy = randn(T, size(y))
        cy = CuArray(y)
        cdy = CuArray(dy)
        @test cu_isapprox(pool_grad(x, y, dy; padding=1), pool_grad(cx, cy, cdy; padding=1))
        @test cu_isapprox(pool_grad(x, y, dy; padding=1), pool_grad(cx, cy, cdy; padding=1))
    end

end


@testset "batchnorm" begin

    # smoke tests
    for T in [Float32, Float64]
        x = CuArray(randn(T, 5, 4, 3, 2))
        s = BatchNormState(x)        

        y = batchnorm_train(x, s)
        y = batchnorm_infer(x, s)
        dy = similar(y)
        dx = batchnorm_grad(x, dy, s)
    end

end


@testset "softmax" begin

    # softmax() in NNlib acts on 2D arrays, while cuDNN seems to only accept 4D ones
    # so it's unclear how to test it and where we need it at all
    # I keep a smoke test here to make sure functions are at least callable, but
    # their correctness is in question
    for T in [Float32, Float64]
        x = CuArray(randn(T, 5, 4, 3, 2))
        y = softmax4d(x)
        dy = similar(y)

        dx = softmax4d_grad(y, dy)
    end

end
