
"""
    using Upper Plane and qr decomposition to get Projector
"""
function LeftMover(c1,c2,c3,c4,e1,e2,e3,e4,e5,e6,e7,e8,a,b,chimax)
    Dlink=size(e1)[2];chi = size(c1)[1];dphy = size(a,1)

    #---------  Construct Projector for Middle plane
    P2,Pdagger2 = ConstructProjector2(c1,c2,c3,c4,e1,e2,e3,e4,e5,e6,e7,e8,a,b,chimax)
    #-----      Simple Version to Construct Projector. If we want to be more accurate
    #-----      We should modify ConstructProjector2 to be consistent with unit cell ipeps
    @tensor c1tilt[-1,-2,-3,-4] := c1[-1,1]*e3[1,-2,-3,-4]
    @tensor c4tilt[-1,-2,-3,-4] := c4[1,-1]*e8[-4,-2,-3,1]
    @tensor c1c1dagger[-1,-2,-3,-4,-5,-6] := c1tilt[-1,-2,-3,1]*c1tilt[-4,-5,-6,1]
    @tensor c4c4dagger[-1,-2,-3,-4,-5,-6] := c4tilt[-1,-2,-3,1]*c4tilt[-4,-5,-6,1]
    P1,Pdagger1 = ConstructProjector(c1c1dagger,c4c4dagger,chimax)
    P3=P1;Pdagger3=Pdagger1
    #P1,Pdagger1 = ConstructProjector1(c1c1dagger,chimax)
    #Pdagger3,P3 = ConstructProjector1(c4c4dagger,chimax)
    @tensor c1new[-1,-2] := c1[1,2]*e3[2,3,4,-2]*P1[1,3,4,-1]
    @tensor c4new[-1,-2] := c4[2,1]*e8[-1,3,4,2]*Pdagger3[1,3,4,-2]
    @tensor e2new[-1,-2,-3,-4] := e2[1,4,2,7]*a[6,9,2,3,-3]*conj(a)[6,8,4,5,-2]*P2[1,5,3,-1]*Pdagger1[7,8,9,-4]
    @tensor e1new[-1,-2,-3,-4] := e1[1,4,2,7]*b[6,9,2,3,-3]*conj(b)[6,8,4,5,-2]*P3[1,5,3,-1]*Pdagger2[7,8,9,-4]
    c1=c1new/maximum(c1new);c4=c4new/maximum(c4new)
    e1=e1new/maximum(e1new);e2=e2new/maximum(e2new);
    return c1,c4,e1,e2

end

function RightMover(c3,c4,c1,c2,e5,e6,e7,e8,e1,e2,e3,e4,a,b,chimax)
    a = permutedims(a,[1,4,5,2,3])
    b = permutedims(b,[1,4,5,2,3])
    c3,c2,e5,e6 = LeftMover(c3,c4,c1,c2,e5,e6,e7,e8,e1,e2,e3,e4,a,b,chimax)
    return c3,c2,e5,e6
end


function UpMover(c2,c3,c4,c1,e3,e4,e5,e6,e7,e8,e1,e2,a,b,chimax)
    a = permutedims(a,[1,5,2,3,4])
    b = permutedims(b,[1,5,2,3,4])
    c2,c1,e3,e4 = LeftMover(c2,c3,c4,c1,e3,e4,e5,e6,e7,e8,e1,e2,a,b,chimax)
    return c2,c1,e3,e4
end

function DownMover(c4,c1,c2,c3,e7,e8,e1,e2,e3,e4,e5,e6,a,b,chimax)
    a = permutedims(a,[1,3,4,5,2])
    b = permutedims(b,[1,3,4,5,2])
    c4,c3,e7,e8 = LeftMover(c4,c1,c2,c3,e7,e8,e1,e2,e3,e4,e5,e6,a,b,chimax)
    return c4,c3,e7,e8
end

#---------------------------------------------
#
#
#
#
#                         Mover For UC: Corner and Edge Environment Separate
#
#
#
#------------------------------------------
function PermuteEnvironment(UnitCell::Array{Array{Float64}},CornerEnvironment::Array{Array{Array{Float64}}},
                        EdgeEnvironment::Array{Array{Array{Float64}}},cellper::Array{Int64},envper::Array{Int64})
    UnitCelltemp = copy(UnitCell)
    for i in 1:length(UnitCell)
            UnitCelltemp[i]=permutedims(UnitCell[i],cellper)
            permute!(CornerEnvironment[i],envper)
            permute!(EdgeEnvironment[i],envper)
    end
    return UnitCelltemp,CornerEnvironment,EdgeEnvironment
end



function LeftMoverUC(UnitCell::Array{Array{Float64}},CornerEnvironment::Array{Array{Array{Float64}}},
                EdgeEnvironment::Array{Array{Array{Float64}}},chimax::Int64,constructprojector::Function)
    x = size(UnitCell,1);y = size(UnitCell,2)
    for i in 1:x
        isometry = Array{Array{Float64}}(undef,0)
        isometrydagger = Array{Array{Float64}}(undef,0)
        for j in 1:y
            i == x ? inext = 1 : inext = i+1; j == y ? jnext = 1 : jnext = j+1 # define inext and jnext
            # extract the environment tensor
            c1 = CornerEnvironment[i,j][1];c2=CornerEnvironment[inext,j][2];c3=CornerEnvironment[inext,jnext][3];c4=CornerEnvironment[i,jnext][4]
            e1 = EdgeEnvironment[i,jnext][1];e2 = EdgeEnvironment[i,j][1];e3 = EdgeEnvironment[i,j][2];e4 = EdgeEnvironment[inext,j][2];
            e5 = EdgeEnvironment[inext,j][3];e6 = EdgeEnvironment[inext,jnext][3];e7 = EdgeEnvironment[inext,jnext][4];e8 = EdgeEnvironment[i,jnext][4]
            a = UnitCell[i,j];b = UnitCell[i,jnext]
            P,Pdagger = constructprojector(c1,c2,c3,c4,e1,e2,e3,e4,e5,e6,e7,e8,a,b,chimax)

            @tensor CornerEnvironment[inext,j][4][-1,-2] = CornerEnvironment[i,j][4][2,1]*EdgeEnvironment[i,j][4][-1,3,4,2]*Pdagger[1,3,4,-2]
            @tensor CornerEnvironment[inext,jnext][1][-1,-2] =CornerEnvironment[i,jnext][1][1,2]*EdgeEnvironment[i,jnext][2][2,3,4,-2]*P[1,3,4,-1]
            CornerEnvironment[inext,j][4][:]= CornerEnvironment[inext,j][4]/maximum(CornerEnvironment[inext,j][4])
            CornerEnvironment[inext,jnext][1][:] = CornerEnvironment[inext,jnext][1]/maximum(CornerEnvironment[inext,jnext][1])
            append!(isometry,[P])
            append!(isometrydagger,[Pdagger])
            if j > 1 
                @tensor EdgeEnvironment[inext,j][1][-1,-2,-3,-4] = EdgeEnvironment[i,j][1][1,4,2,7]*isometry[end][1,5,3,-1]*isometrydagger[end-1][7,8,9,-4]*
                                UnitCell[i,j][6,9,2,3,-3]*UnitCell[i,j][6,8,4,5,-2]
                EdgeEnvironment[inext,j][1][:] = EdgeEnvironment[inext,j][1]/maximum(EdgeEnvironment[inext,j][1])
            end
            if j == y
                @tensor EdgeEnvironment[inext,1][1][-1,-2,-3,-4] = EdgeEnvironment[i,1][1][1,4,2,7]*isometry[1][1,5,3,-1]*isometrydagger[end][7,8,9,-4]*
                                UnitCell[i,1][6,9,2,3,-3]*UnitCell[i,1][6,8,4,5,-2]
                EdgeEnvironment[inext,1][1][:] = EdgeEnvironment[inext,1][1]/maximum(EdgeEnvironment[inext,1][1])
            end
        end
    end
    return CornerEnvironment,EdgeEnvironment
end



"""
    This is a test for LeftMover + Isometries
"""
function LeftMoverUCIsometry(UnitCell::Array{Array{Float64}},CornerEnvironment::Array{Array{Array{Float64}}},
    EdgeEnvironment::Array{Array{Array{Float64}}},chimax::Int64,Dp::Int64,constructprojector::Function)

    x = size(UnitCell,1);y = size(UnitCell,2)
    for i in 1:x
        isometry = Array{Array{Float64}}(undef,0)
        isometrydagger = Array{Array{Float64}}(undef,0)
        aUMatrix = Array{Array{Float64}}(undef,0)
        aDMatrix = Array{Array{Float64}}(undef,0)
        for j in 1:y
            i == x ? inext = 1 : inext = i+1; j == y ? jnext = 1 : jnext = j+1 # define inext and jnext
            # extract the environment tensor
            c1 = CornerEnvironment[i,j][1];c2=CornerEnvironment[inext,j][2];c3=CornerEnvironment[inext,jnext][3];c4=CornerEnvironment[i,jnext][4]
            e1 = EdgeEnvironment[i,jnext][1];e2 = EdgeEnvironment[i,j][1];e3 = EdgeEnvironment[i,j][2];e4 = EdgeEnvironment[inext,j][2];
            e5 = EdgeEnvironment[inext,j][3];e6 = EdgeEnvironment[inext,jnext][3];e7 = EdgeEnvironment[inext,jnext][4];e8 = EdgeEnvironment[i,jnext][4]
            a = UnitCell[i,j];b = UnitCell[i,jnext]
            
        
            #aU,aL,aD,aR,bU,bL,bD,bR,P,Pdagger = constructprojector(c1,c2,c3,c4,e1,e2,e3,e4,e5,e6,e7,e8,a,b,chimax,Dp)
            aU,aL,aD,aR,bU,bL,bD,bR,P,Pdagger = constructprojector(CornerEnvironment[i,j],CornerEnvironment[i,jnext],
                                            EdgeEnvironment[i,j],EdgeEnvironment[i,jnext],c1,c2,c3,c4,e1,e2,e3,e4,e5,e6,e7,e8,a,b,chimax,Dp)
            #---- Update Corner Environment
            @tensor CornerEnvironment[inext,j][4][-1,-2] = CornerEnvironment[i,j][4][3,4]*EdgeEnvironment[i,j][4][-1,1,2,3]*Pdagger[4,5,-2]*bU[1,2,5]
            @tensor CornerEnvironment[inext,jnext][1][-1,-2] =CornerEnvironment[i,jnext][1][4,3]*EdgeEnvironment[i,jnext][2][3,1,2,-2]*P[4,5,-1]*aD[1,2,5]
            CornerEnvironment[inext,j][4][:]= CornerEnvironment[inext,j][4]/maximum(CornerEnvironment[inext,j][4])
            CornerEnvironment[inext,jnext][1][:] = CornerEnvironment[inext,jnext][1]/maximum(CornerEnvironment[inext,jnext][1])
            append!(isometry,[P])
            append!(isometrydagger,[Pdagger])
            append!(aUMatrix,[aU])
            append!(aDMatrix,[aD])
            #---- Update Edge Environment

            if j > 1
                @tensor EdgeEnvironment[inext,j][1][-1,-2,-3,-4] = EdgeEnvironment[i,j][1][2,5,3,9]*isometry[end][2,1,-1]*isometrydagger[end-1][9,8,-4]*
                                UnitCell[i,j][7,11,3,4,-3]*UnitCell[i,j][7,10,5,6,-2]*aU[10,11,8]*aD[6,4,1]
                EdgeEnvironment[inext,j][1][:] = EdgeEnvironment[inext,j][1]/maximum(EdgeEnvironment[inext,j][1])
            end
            if j == y
                @tensor EdgeEnvironment[inext,1][1][-1,-2,-3,-4] = EdgeEnvironment[i,1][1][2,5,3,9]*isometry[1][2,1,-1]*isometrydagger[end][9,8,-4]*
                                UnitCell[i,1][7,11,3,4,-3]*UnitCell[i,1][7,10,5,6,-2]*aUMatrix[1][10,11,8]*aDMatrix[1][6,4,1]
                EdgeEnvironment[inext,1][1][:] = EdgeEnvironment[inext,1][1]/maximum(EdgeEnvironment[inext,1][1])
            end
        end
    end

    return CornerEnvironment,EdgeEnvironment
end


function LeftMoverSingleLayer(UnitCell::Array{Array{Float64}},CornerEnvironment::Array{Array{Array{Float64}}},
    EdgeEnvironment::Array{Array{Array{Float64}}},chimax::Int64,Dp::Int64,constructprojector::Function)


    x = size(UnitCell,1);y = size(UnitCell,2)
    for i in 1:x
        isometryd1 = Array{Array{Float64}}(undef,0)
        isometrydaggerd1 = Array{Array{Float64}}(undef,0)
        isometryd2 = Array{Array{Float64}}(undef,0)
        isometrydaggerd2 = Array{Array{Float64}}(undef,0)
        for j in 1:y
            i == x ? inext = 1 : inext = i+1; j == y ? jnext = 1 : jnext = j+1 # define inext and jnext
            # extract the environment tensor
            c1 = CornerEnvironment[i,j][1];c2 = CornerEnvironment[i,j][2]
            c3 = CornerEnvironment[i,j][3];c4 = CornerEnvironment[i,j][4]
            e1 = EdgeEnvironment[i,j][1];e2 = EdgeEnvironment[i,j][2]
            e3 = EdgeEnvironment[i,j][3];e4 = EdgeEnvironment[i,j][4]
            a = UnitCell[i,j];b = UnitCell[i,jnext]

            ptype = "Double"
            #
            P1,Pdagger1 = ConstructProjectorSingleLayer(c1,c2,c3,c4,e1,e2,e3,e4,a,b,chimax,direction="up",type="Double")
            #---- Update Corner Environment
            @tensor CornerEnvironment[inext,j][4][-1,-2] = CornerEnvironment[i,j][4][1,2]*EdgeEnvironment[i,j][4][-1,3,4,1]*Pdagger1[2,3,4,-2]
            @tensor CornerEnvironment[inext,jnext][1][-1,-2] = CornerEnvironment[i,jnext][1][1,2]*EdgeEnvironment[i,jnext][2][2,3,4,-2]*P1[1,3,4,-1]
            CornerEnvironment[inext,j][4][:]= CornerEnvironment[inext,j][4]/maximum(CornerEnvironment[inext,j][4])
            CornerEnvironment[inext,jnext][1][:] = CornerEnvironment[inext,jnext][1]/maximum(CornerEnvironment[inext,jnext][1])

            append!(isometryd1,[P1])
            append!(isometrydaggerd1,[Pdagger1])
            
            #---- Update Edge Environment
            if j > 1
                @tensor EdgeEnvironment[inext,j][1][-1,-2,-3,-4] = EdgeEnvironment[i,j][1][1,4,2,7]*isometryd1[end][1,5,3,-1]*
                                            isometrydaggerd1[end-1][7,8,9,-4]*UnitCell[i,j][6,9,2,3,-3]*UnitCell[i,j][6,8,4,5,-2]
                EdgeEnvironment[inext,j][1][:] = EdgeEnvironment[inext,j][1]/maximum(EdgeEnvironment[inext,j][1])
            end
            if j == y
                @tensor EdgeEnvironment[inext,1][1][-1,-2,-3,-4] = EdgeEnvironment[i,1][1][1,4,2,7]*isometryd1[1][1,5,3,-1]*
                                            isometrydaggerd1[end][7,8,9,-4]*UnitCell[i,1][6,9,2,3,-3]*UnitCell[i,1][6,8,4,5,-2]
                EdgeEnvironment[inext,1][1][:] = EdgeEnvironment[inext,1][1]/maximum(EdgeEnvironment[inext,1][1])
            end
            #

            #----- Single Layer
            #=
            P1,Pdagger1 = ConstructProjectorSingleLayer(c1,c2,c3,c4,e1,e2,e3,e4,a,b,chimax,direction="dn",type="Ket")
            @tensor c4edge[:] := c4[2,1]*e4[-1,-2,3,2]*Pdagger1[1,3,-3]
            P2,Pdagger2 = ConstructProjectorSingleLayer(c1,c2,c3,c4edge,e1,e2,e3,e4,a,b,chimax,direction="dn",type="Bra")

            #---- Update Corner Environment
            @tensor CornerEnvironment[inext,j][4][-1,-2] = CornerEnvironment[i,j][4][2,1]*EdgeEnvironment[i,j][4][-1,5,3,2]*Pdagger1[1,3,4]*Pdagger2[4,5,-2]
            @tensor CornerEnvironment[inext,jnext][1][-1,-2] = CornerEnvironment[i,jnext][1][1,2]*EdgeEnvironment[i,jnext][2][2,5,3,-2]*P1[1,3,4]*P2[4,5,-1]
            CornerEnvironment[inext,j][4][:]= CornerEnvironment[inext,j][4]/maximum(CornerEnvironment[inext,j][4])
            CornerEnvironment[inext,jnext][1][:] = CornerEnvironment[inext,jnext][1]/maximum(CornerEnvironment[inext,jnext][1])

            append!(isometryd1,[P1])
            append!(isometryd2,[P2])
            append!(isometrydaggerd1,[Pdagger1])
            append!(isometrydaggerd2,[Pdagger2])
            
            #---- Update Edge Environment
            if j > 1
                @tensor EdgeEnvironment[inext,j][1][-1,-2,-3,-4] = EdgeEnvironment[i,j][1][1,7,2,4]*isometryd1[end][1,3,6]*isometryd2[end][6,8,-1]*
                                isometrydaggerd1[end-1][4,5,10]*isometrydaggerd2[end-1][10,11,-4]*UnitCell[i,j][9,5,2,3,-3]*UnitCell[i,j][9,11,7,8,-2]
                EdgeEnvironment[inext,j][1][:] = EdgeEnvironment[inext,j][1]/maximum(EdgeEnvironment[inext,j][1])
            end
            if j == y
                @tensor EdgeEnvironment[inext,1][1][-1,-2,-3,-4] = EdgeEnvironment[i,1][1][1,7,2,4]*isometryd1[1][1,3,6]*isometryd2[1][6,8,-1]*
                                isometrydaggerd1[end][4,5,10]*isometrydaggerd2[end][10,11,-4]*UnitCell[i,1][9,5,2,3,-3]*UnitCell[i,1][9,11,7,8,-2]
                EdgeEnvironment[inext,1][1][:] = EdgeEnvironment[inext,1][1]/maximum(EdgeEnvironment[inext,1][1])
            end
            =#
        end
        #
    end
    return CornerEnvironment,EdgeEnvironment
end

"""
    mover(param::PEPSParam;direction=1::Int64)

Mover for computing the CTM environment.
    direction == 1   :    Left Mover
    direction == 2   :    Up Mover
    direction == 3   :    Right Mover
    direction == 4   :    Down Mover
"""
function mover(param::PEPSParam;direction=1::Int64)
    x = size(param.UnitCell,1); y = size(param.UnitCell,2)
    #   permarg stores the permute parameter that is used to different direction movers
    @match direction begin
        1 => (permarg= [[1,2],[1:1:x,1:1:y],[1,2,3,4,5],[1,2,3,4],[1,2,3,4,5],[1,2,3,4]])
        2 => (permarg= [[2,1],[x:-1:1,1:1:y],[1,5,2,3,4],[2,3,4,1],[1,3,4,5,2],[4,1,2,3]])
        3 => (permarg= [[1,2],[x:-1:1,y:-1:1],[1,4,5,2,3],[3,4,1,2],[1,4,5,2,3],[3,4,1,2]])
        4 => (permarg= [[2,1],[1:1:x,y:-1:1],[1,3,4,5,2],[4,1,2,3],[1,5,2,3,4],[2,3,4,1]])
    end

    UnitCelltemp = permutedims(param.UnitCell,permarg[1])[permarg[2]...]
    CornerEnvironmenttemp = permutedims(param.CornerEnvironment,permarg[1])[permarg[2]...]
    EdgeEnvironmenttemp = permutedims(param.EdgeEnvironment,permarg[1])[permarg[2]...]

    UnitCelltemp,CornerEnvironmenttemp,EdgeEnvironmenttemp = PermuteEnvironment(UnitCelltemp,CornerEnvironmenttemp,
                                            EdgeEnvironmenttemp,permarg[3],permarg[4])
    if param.MoverType == "Normal"
        CornerEnvironmenttemp,EdgeEnvironmenttemp= LeftMoverUC(UnitCelltemp,CornerEnvironmenttemp,EdgeEnvironmenttemp,
                                                param.chimax,param.ProjectorFunction)
    elseif param.MoverType == "Isometry"
        CornerEnvironmenttemp,EdgeEnvironmenttemp= LeftMoverUCIsometry(UnitCelltemp,CornerEnvironmenttemp,
                                            EdgeEnvironmenttemp,param.chimax,param.Dp,param.ProjectorFunction)
    elseif param.MoverType == "SingleLayer"
        CornerEnvironmenttemp,EdgeEnvironmenttemp= LeftMoverSingleLayer(UnitCelltemp,CornerEnvironmenttemp,
        EdgeEnvironmenttemp,param.chimax,param.Dp,param.ProjectorFunction)
    end

    UnitCelltemp,CornerEnvironmenttemp,EdgeEnvironmenttemp = PermuteEnvironment(UnitCelltemp,CornerEnvironmenttemp,
                                            EdgeEnvironmenttemp,permarg[5],permarg[6])

    param.CornerEnvironment[:] = permutedims(CornerEnvironmenttemp[permarg[2]...],permarg[1])
    param.EdgeEnvironment[:] = permutedims(EdgeEnvironmenttemp[permarg[2]...],permarg[1])
    #return CornerEnvironment,EdgeEnvironment
end


#-----------------------------------------------------------------------------------------------------------------
#
#
#
#
#                                          Movers for Hamiltonian
#
#
#
#
#-----------------------------------------------------------------------------------------------------------------



"""
    using Upper Plane and qr decomposition to get Projector
"""
function NLeftMoverWithH(c1,c2,c3,c4,e1,e2,e3,e4,e5,e6,e7,e8,a,b,chimax;caltype="NORM",calstep="Normal",
                        e1hm=nothing,e2hm=nothing,MPO=nothing)
    Dlink=size(e1)[2];chi = size(c1)[1];dphy = size(a,1)
    #---------  Construct Projector for Middle plane
    P2,Pdagger2 = ConstructProjector2(c1,c2,c3,c4,e1,e2,e3,e4,e5,e6,e7,e8,a,b,chimax)
    #-----      Simple Version to Construct Projector. If we want to be more accurate
    #-----      We should modify ConstructProjector2 to be consistent with unit cell ipeps
    @tensor c1tilt[-1,-2,-3,-4] := c1[-1,1]*e3[1,-2,-3,-4]
    @tensor c4tilt[-1,-2,-3,-4] := c4[1,-1]*e8[-4,-2,-3,1]
    @tensor c1c1dagger[-1,-2,-3,-4,-5,-6] := c1tilt[-1,-2,-3,1]*c1tilt[-4,-5,-6,1]
    @tensor c4c4dagger[-1,-2,-3,-4,-5,-6] := c4tilt[-1,-2,-3,1]*c4tilt[-4,-5,-6,1]
    P1,Pdagger1 = ConstructProjector(c1c1dagger,c4c4dagger,chimax)
    P3=P1;Pdagger3=Pdagger1
    @tensor c1new[-1,-2] := c1[1,2]*e3[2,3,4,-2]*P1[1,3,4,-1]
    @tensor c4new[-1,-2] := c4[2,1]*e8[-1,3,4,2]*Pdagger3[1,3,4,-2]
    @tensor e2new[-1,-2,-3,-4] := e2[1,4,2,7]*a[6,9,2,3,-3]*conj(a)[6,8,4,5,-2]*P2[1,5,3,-1]*Pdagger1[7,8,9,-4]
    @tensor e1new[-1,-2,-3,-4] := e1[1,4,2,7]*b[6,9,2,3,-3]*conj(b)[6,8,4,5,-2]*P3[1,5,3,-1]*Pdagger2[7,8,9,-4]
    if caltype == "HENV"
        if calstep == "Initial"
            @tensor e1Hmnew[-1,-2,-3,-4,-5] := e1[1,4,2,8]*b[7,10,2,3,-4]*conj(b)[6,9,4,5,-2]*P3[1,5,3,-1]*Pdagger2[8,9,10,-5]*MPO[6,7,-3]
            @tensor e2Hmnew[-1,-2,-3,-4,-5] := e2[1,4,2,8]*a[7,10,2,3,-4]*conj(a)[6,9,4,5,-2]*P2[1,5,3,-1]*Pdagger1[8,9,10,-5]*MPO[6,7,-3]
        elseif calstep == "Normal"
            e1hm == nothing || e2hm == nothing ? throw("e1Hm and e2Hm does not defined!") : ( )
            @tensor e1Hmnew[-1,-2,-3,-4,-5] := e1hm[1,6,4,2,9]*b[5,11,2,3,-4]*conj(b)[8,10,6,7,-2]*P3[1,7,3,-1]*Pdagger2[9,10,11,-5]*MPO[8,4,5,-3]
            @tensor e2Hmnew[-1,-2,-3,-4,-5] := e2hm[1,6,4,2,9]*a[5,11,2,3,-4]*conj(a)[8,10,6,7,-2]*P2[1,7,3,-1]*Pdagger1[9,10,11,-5]*MPO[8,4,5,-3]
        else
            throw("calstep: Only \"Initial\" and \"Normal\" are supported ")
        end
        e1Hmnew = e1Hmnew/maximum(e1new);e2Hmnew = e2Hmnew/maximum(e2new)
        c1=c1new/maximum(c1new);c4=c4new/maximum(c4new)
        e1=e1new/maximum(e1new);e2=e2new/maximum(e2new);
        return c1,c4,e1,e2,e1Hmnew,e2Hmnew
    elseif caltype == "NORM"
        c1=c1new/maximum(c1new);c4=c4new/maximum(c4new)
        e1=e1new/maximum(e1new);e2=e2new/maximum(e2new);
        return c1,c4,e1,e2
    else
        throw("caltype: Only \"NORM\" and \"HENV\" are supported ")
    end
end

function NRightMoverWithH(c3,c4,c1,c2,e5,e6,e7,e8,e1,e2,e3,e4,a,b,chimax;caltype="NORM",calstep="Normal",
                        e1hm=nothing,e2hm=nothing,MPO=nothing)
    a = permutedims(a,[1,4,5,2,3])
    b = permutedims(b,[1,4,5,2,3])
    if caltype == "HENV"
    if calstep == "Initial"
        MPO = permutedims(MPO,[1,3,2])
    else
        MPO = permutedims(MPO,[1,4,3,2])
    end
    end
    c3,c2,e5,e6 = NLeftMoverWithH(c3,c4,c1,c2,e5,e6,e7,e8,e1,e2,e3,e4,a,b,chimax;caltype=caltype,calstep=calstep,
                            e1hm=e1hm,e2hm=e2hm,MPO=MPO)
end


function NUpMoverWithH(c2,c3,c4,c1,e3,e4,e5,e6,e7,e8,e1,e2,a,b,chimax;caltype="NORM",calstep="Normal",
                        e1hm=nothing,e2hm=nothing,MPO=nothing)
    a = permutedims(a,[1,5,2,3,4])
    b = permutedims(b,[1,5,2,3,4])
    c2,c1,e3,e4 = NLeftMoverWithH(c2,c3,c4,c1,e3,e4,e5,e6,e7,e8,e1,e2,a,b,chimax;caltype=caltype,calstep=calstep,
                            e1hm=e1hm,e2hm=e2hm,MPO=MPO)
end

function NDownMoverWithH(c4,c1,c2,c3,e7,e8,e1,e2,e3,e4,e5,e6,a,b,chimax;caltype="NORM",calstep="Normal",
                        e1hm=nothing,e2hm=nothing,MPO=nothing)
    a = permutedims(a,[1,3,4,5,2])
    b = permutedims(b,[1,3,4,5,2])
    if caltype == "HENV"
    if calstep == "Initial"
        MPO = permutedims(MPO,[1,3,2])
    else
        MPO = permutedims(MPO,[1,4,3,2])
    end
    end
    c4,c3,e7,e8 = NLeftMoverWithH(c4,c1,c2,c3,e7,e8,e1,e2,e3,e4,e5,e6,a,b,chimax;caltype=caltype,calstep=calstep,
                            e1hm=e1hm,e2hm=e2hm,MPO=MPO)
end
