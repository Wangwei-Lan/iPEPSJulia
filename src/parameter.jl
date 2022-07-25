mutable struct PEPSParam
    spin::Float64
    dphy::Int64
    Dstep::Array{Float64}
    Dlink::Array{Int64}
    chistep::Array{Float64}
    chimax::Array{Int64}
    MoverType::String
    Magnetism::Float64
    Sx::Array{Float64}
    Sy::Array{Complex{Float64}}
    Sz::Array{Float64}
    Hamiltonian::Array{Float64}
    UnitCell::Array{Array{Float64}}
    tao::Array{Float64}
    step::Array{Float64}
    g::Array{Array{Float64}}
    Tol::Float64
    SimpleUpdateOn::String
    FullUpdateOn::String
    SimpTol::Float64
    MaxLoop::Int64
    MaxSimpLoop::Int64
    ProjectorFunction::Function
    EnergyArray::Array{Float64}
    CornerEnvironment::Array{Array{Array{Float64}}}
    EdgeEnvironment::Array{Array{Array{Float64}}}
end

function SetParam(;spin::Float64 = 0.5,dphy::Int64 = 2,Dlink::Array{Int64} = [2],Dstep::Array{Float64} = [2],chistep::Array{Float64} = [10],chimax:: Array{Int64}=[10],MoverType::String="Normal",HamiType::String= "Heisenberg",
                    tao::Array{Float64}=[0.0],step::Array{Float64}=[0.0],g=nothing,Tol::Float64=1.0e-8,SimpTol::Float64=0.0,MaxLoop::Int64=0.0,
                    MaxSimpLoop::Int64=0,ProjectorFunction::Function,SimpleUpdateOn::String="false",FullUpdateOn::String="false",Magnetism::Float64=0.0)
    Sx,Sy,Sz = Spin(spin)
    Hamiltonian = ConstructHamiltonian(Sx,Sy,Sz,HamiType,Magnetism=Magnetism)
    if tao != [0.0]
        g = exp.([-real(Hamiltonian)*j for j in tao])
        g = [reshape(j,dphy.*(1,1,1,1)) for j in g]
    end
    #UnitCell =  Array{Array{Float64}}(undef,2,2)
    #Dp = Dlink
    UnitCell = [rand(dphy,Dlink[1],Dlink[1],Dlink[1],Dlink[1]) for i in 1:2,j in 1:2]
    UnitCell[1,2] = deepcopy(UnitCell[1,1])
    UnitCell[2,1] = deepcopy(UnitCell[1,1])
    UnitCell[2,2] = deepcopy(UnitCell[1,1])

    Hamiltonian = reshape(real(Hamiltonian),dphy.*(1,1,1,1))
    EnergyArray = Array{Float64}(undef,0)
    

    #=
    CornerEnvironment =[[Matrix(1.0I,chimax[1],chimax[1]) for k in 1:4] for i in 1:2, j in 1:2]
    CornerEnvironment[2,2] = deepcopy(CornerEnvironment[1,1])
    CornerEnvironment[2,1] = deepcopy(CornerEnvironment[1,2])
    #EdgeEnvironment =[[rand(chimax[1],Dlink[1],Dlink[1],chimax[1]) for k in 1:4] for i in 1:2, j in 1:2]
    EdgeEnvironment =[[reshape(Matrix(1.0I,chimax[1]*Dlink[1],chimax[1]*Dlink[1]),chimax[1],Dlink[1],Dlink[1],chimax[1]) for k in 1:4] for i in 1:2, j in 1:2]
    EdgeEnvironment[2,2] = deepcopy(EdgeEnvironment[1,1])
    EdgeEnvironment[2,1] = deepcopy(EdgeEnvironment[1,2])
    =#
    #
    CornerEnvironment =[[rand(chimax[1],chimax[1]) for k in 1:4] for i in 1:2, j in 1:2]
    CornerEnvironment[2,2] = deepcopy(CornerEnvironment[1,1])
    CornerEnvironment[2,1] = deepcopy(CornerEnvironment[1,2])
    EdgeEnvironment =[[rand(chimax[1],Dlink[1],Dlink[1],chimax[1]) for k in 1:4] for i in 1:2, j in 1:2]
    EdgeEnvironment[2,2] = deepcopy(EdgeEnvironment[1,1])
    EdgeEnvironment[2,1] = deepcopy(EdgeEnvironment[1,2])
    #

    param = PEPSParam(spin,dphy,Dstep,Dlink,chistep,chimax,MoverType,Magnetism,Sx,Sy,Sz,Hamiltonian,UnitCell,tao,step,g,Tol,SimpleUpdateOn,
                FullUpdateOn,SimpTol,MaxLoop,MaxSimpLoop,ProjectorFunction,EnergyArray,CornerEnvironment,EdgeEnvironment)
    return param
end

function reset_parameter(param,chimax,Dlink)

    CornerEnvironment =[[Matrix(1.0I,chimax[1],chimax[1]) for k in 1:4] for i in 1:2, j in 1:2]
    EdgeEnvironment =[[reshape(Matrix(1.0I,chimax[1]*Dlink[1],chimax[1]*Dlink[1]),
                    chimax[1],Dlink[1],Dlink[1],chimax[1]) for k in 1:4] for i in 1:2, j in 1:2]
    param.CornerEnvironment = CornerEnvironment
    param.EdgeEnvironment = EdgeEnvironment

end