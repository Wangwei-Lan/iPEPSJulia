function GammaMultiLambda(x,fargs = Any[])
    Gamma = fargs[1];Lambda = fargs[2]
    chi = size(Gamma,1)
    x = reshape(x,chi,chi)
    @tensor y[-1,-2] := Gamma[-1,4,3]*Gamma[-2,4,5]*Lambda[3,1]*Lambda[5,2]*x[1,2]
    y = reshape(y,chi^2)
    return y 
end

x = 1

#########################
function impeigs(f::Function; n = 0, issym = true, nev = 1, tol = 1e-10, which=:LM, maxiter = 300, v0 = [0.0], fargs = Any[])
  # function impeigs(f::Function; n = 0, issym = true, nev = 1, tol = 1e-10, which=:LM, maxiter = 300, v0 = [0.0], fargs = Any[])
  #
  # 'impeigs' is a wrapper for 'eigs' which reproduces the functionality (and syntax)
  # of the MATLAB eigs in allowing it to accept a function 'f', rather than a matrix,
  # as an input. Optional argument 'fargs' allows you to pass additional arguments
  # to the function 'f', (similar to the MATLAB version). (C'mon Julia developers,
  # the ability to pass functions to eigs should come as standard!!!)

  Atemp = SimpleLinOp(n, f, issym, fargs)
  if length(v0) != n
    v0 = rand(n);
  end
  D, U = Arpack.eigs(Atemp; nev=nev, tol=tol, which=which, maxiter = maxiter, v0 = v0)
  return D, U
end

#########################
struct SimpleLinOp
  osize::Int64 #size of linear operator A
  oloc::Function #function that computes A*x for input vector x
  osym::Bool #is operator symmetric
  fargs::Array{Any} #additional arguments passed to function
end

function Base.:size(A::SimpleLinOp)
  return A.osize, A.osize
end
function Base.:eltype(A::SimpleLinOp)
  return Float64
end
function Base.:*(A::SimpleLinOp,x::Array{Float64})
  return A.oloc(x,A.finputs)
end
function Base.:ndims(A::SimpleLinOp)
  return 2
end
function Base.:indices(A::SimpleLinOp, d::Int64)
  return Base.OneTo(A.osize)
end
function LinearAlgebra.:issymmetric(A::SimpleLinOp)
  return A.osym
end
function LinearAlgebra.:A_mul_B!(y, A::SimpleLinOp, x)
  y[:] = A.oloc(x,A.fargs);
end
