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


"""
    PermuteEnvironment()

    permute environments for different movers: according to given orders

"""
function PermuteEnvironment(UnitCell::Array{Array{Float64}},CornerEnvironment::Array{Array{Array{Float64}}},
                        EdgeEnvironment::Array{Array{Array{Float64}}},cellper::Array{Int64},envper::Array{Int64})
    UnitCelltemp = deepcopy(UnitCell)
    for i in 1:length(UnitCell)
        UnitCelltemp[i]=permutedims(UnitCell[i],cellper)
        permute!(CornerEnvironment[i],envper)
        permute!(EdgeEnvironment[i],envper)
    end
    return UnitCelltemp,CornerEnvironment,EdgeEnvironment
end



function PermuteEnvironment(paramc::PEPSParam,cellper::Array{Int64},envper::Array{Int64})
    UnitCelltemp = deepcopy(paramc.UnitCell)
    for i in 1:length(paramc.UnitCell)
        UnitCelltemp[i]=permutedims(paramc.UnitCell[i],cellper)
        permute!(paramc.CornerEnvironment[i],envper)
        permute!(paramc.EdgeEnvironment[i],envper)
    end
    paramc.UnitCell[:] = UnitCelltemp
end



"""
    LeftMover2x2UnitCell

    left mover according to 2x2 unit cells : projectors can be calculated either using ComputeProjectors_Unitary or 
"""
function LeftMover2x2UnitCell(UnitCell::Array{Array{Float64}},CornerEnvironment::Array{Array{Array{Float64}}},
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

            #------   Update CornerEnvironment
            @tensor CornerEnvironment[inext,j][4][:] := CornerEnvironment[i,j][4][2,1]*EdgeEnvironment[i,j][4][-1,3,4,2]*Pdagger[1,3,4,-2]
            @tensor CornerEnvironment[inext,jnext][1][:] :=CornerEnvironment[i,jnext][1][1,2]*EdgeEnvironment[i,jnext][2][2,3,4,-2]*P[1,3,4,-1]
            CornerEnvironment[inext,j][4][:]= CornerEnvironment[inext,j][4]/maximum(CornerEnvironment[inext,j][4])
            CornerEnvironment[inext,jnext][1][:] = CornerEnvironment[inext,jnext][1]/maximum(CornerEnvironment[inext,jnext][1])
            append!(isometry,[P])
            append!(isometrydagger,[Pdagger])

            if j > 1 
                @tensor  EdgeEnvironment[inext,j][1][-1,-2,-3,-4] := EdgeEnvironment[i,j][1][1,4,2,7]*isometry[end][1,5,3,-1]*isometrydagger[end-1][7,8,9,-4]*
                                            UnitCell[i,j][6,9,2,3,-3]*UnitCell[i,j][6,8,4,5,-2]
                EdgeEnvironment[inext,j][1][:] = EdgeEnvironment[inext,j][1]/maximum(EdgeEnvironment[inext,j][1])
            end
            if j == y
                @tensor  EdgeEnvironment[inext,1][1][-1,-2,-3,-4] := EdgeEnvironment[i,1][1][1,4,2,7]*isometry[1][1,5,3,-1]*isometrydagger[end][7,8,9,-4]*
                                UnitCell[i,1][6,9,2,3,-3]*UnitCell[i,1][6,8,4,5,-2]
                EdgeEnvironment[inext,1][1][:] = EdgeEnvironment[inext,1][1]/maximum(EdgeEnvironment[inext,1][1])
            end
        end
    end
    return CornerEnvironment,EdgeEnvironment
end



"""

    LeftMoverUCIsometry()


If `y` is unspecified

# Examples
```julia-repl
julia> LeftMoverUCIsometry(UnitCell)
1
```
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
            cij4 = CornerEnvironment[i,j][4]
            eij4 = EdgeEnvironment[i,j][4]
            cijnext1 = CornerEnvironment[i,jnext][1]
            eijnext2 = EdgeEnvironment[i,jnext][2]
            @tensor cijnext4[-1,-2] := cij4[3,4]*eij4[-1,1,2,3]*Pdagger[4,5,-2]*bU[1,2,5]
            @tensor cinextjnext1[-1,-2] :=cijnext1[4,3]*eijnext2[3,1,2,-2]*P[4,5,-1]*aD[1,2,5]
            CornerEnvironment[inext,j][4] = cinextj4
            CornerEnvironment[inext,jnext][1] = cinextjnext1

            CornerEnvironment[inext,j][4][:]= CornerEnvironment[inext,j][4]/maximum(CornerEnvironment[inext,j][4])
            CornerEnvironment[inext,jnext][1][:] = CornerEnvironment[inext,jnext][1]/maximum(CornerEnvironment[inext,jnext][1])
            append!(isometry,[P])
            append!(isometrydagger,[Pdagger])
            append!(aUMatrix,[aU])
            append!(aDMatrix,[aD])
            #---- Update Edge Environment

            if j > 1
                eij1 =  EdgeEnvironment[i,j][1]
                isoend = isometry[end]
                isodaggerend_1 = isometrydagger[end-1]
                ucij = UnitCell[i,j]
                @tensor einextj1[-1,-2,-3,-4] := eij1[2,5,3,9]*isoend[2,1,-1]*isodaggerend_1[9,8,-4]*
                                ucij[7,11,3,4,-3]*ucij[7,10,5,6,-2]*aU[10,11,8]*aD[6,4,1]
                #
                EdgeEnvironment[inext,j][1] = einextj1
                EdgeEnvironment[inext,j][1][:] = EdgeEnvironment[inext,j][1]/maximum(EdgeEnvironment[inext,j][1])
            end
            if j == y

                ei11 = EdgeEnvironment[i,1][1]
                iso1 = isometry[1]
                isodaggerend=isometrydagger[end]
                uci1 = UnitCell[i,1]
                aumatrix1 = aUMatrix[1]
                admatrix1 = aDMatrix[1]
                @tensor einext11[-1,-2,-3,-4] := ei11[2,5,3,9]*iso1[2,1,-1]*isodaggerend[9,8,-4]*
                                uci1[7,11,3,4,-3]*uci1[7,10,5,6,-2]*aumatrix1[10,11,8]*admatrix1[6,4,1]
                #
                EdgeEnvironment[inext,1][1] = einext11
                EdgeEnvironment[inext,1][1][:] = EdgeEnvironment[inext,1][1]/maximum(EdgeEnvironment[inext,1][1])
            end
        end
    end


    return CornerEnvironment,EdgeEnvironment
end


"""
    Compute Projector for update the environment 
"""
function compute_projector(c1,e2,e1,c4,e8,c3,e6,chitemp)


    # Update the projector separately
    #! could it be better to update the projectors simutaneously?
    P1 = rand(size(c1,2),size(e2,3),chitemp)
    @tensor temp[:] := e1[12,14,15,-1]*c4[10,12]*e8[6,8,9,10]*c3[4,6]*e6[1,2,3,4]*e1[13,14,15,-2]*c4[11,13]*e8[7,8,9,11]*c3[5,7]*e6[1,2,3,5]
    temp = temp/maximum(temp)
    @tensor c1e2[:] := c1[1,-3]*e2[-1,-2,-4,1]

    S = rand(chitemp)
    ######################### approximate by introducing isometries
    @tensor Temp[:] := c1e2[3,4,1,2]*c1e2[5,4,-1,-2]*P1[1,2,-3]*temp[3,5]
    for j in 1:15
        @tensor Temp[:] = c1e2[3,4,1,2]*c1e2[5,4,-1,-2]*P1[1,2,-3]*temp[3,5]
        Temp = Temp/maximum(Temp)
        sizeTemp = size(Temp)
        Rlt = svd(reshape(Temp,prod(sizeTemp[1:2]),sizeTemp[3]))
        P1 = reshape(Rlt.U*Rlt.V',sizeTemp)
        #=
        if norm(Rlt.S - S) ./norm(S) < 1.0e-9
            break
        else
            S = Rlt.S
        end
        =#
    end
    
    #=
    @tensor N[:] := c1e2[5,3,2,4]*c1e2[1,3,2,4]*temp[1,5]
    @tensor M[:] := c1e2[8,7,3,4]*c1e2[5,7,1,2]*P1[1,2,6]*P1[3,4,6]*temp[5,8]
    println("truncation1: ",(N[1]-M[1])/N[1])
    =#
    S = rand(chitemp)
    P2 = rand(chitemp,size(e2,2),chitemp)
    @tensor c1e2P1[:] := c1e2[-1,-3,1,2]*P1[1,2,-2] 
    @tensor Temp[:] := c1e2P1[3,1,2]*P2[1,2,-3]*c1e2P1[4,-1,-2]*temp[3,4]
    for j in 1:15
        @tensor Temp[:] = c1e2P1[3,1,2]*P2[1,2,-3]*c1e2P1[4,-1,-2]*temp[3,4]
        Temp = Temp/maximum(Temp)
        sizeTemp = size(Temp)
        Rlt = svd(reshape(Temp,prod(sizeTemp[1:2]),sizeTemp[3]))
        P2 = reshape(Rlt.U*Rlt.V',sizeTemp)
        #=
        if norm(Rlt.S - S) ./norm(S) < 1.0e-9
            break
        else
            S = Rlt.S
        end
        =#
    end


    #=
    @tensor N[:] := c1e2P1[1,2,3]*c1e2P1[4,2,3]*temp[1,4]
    @tensor M[:] := c1e2P1[3,1,2]*c1e2P1[7,4,5]*temp[3,7]*P2[1,2,6]*P2[4,5,6]
    println("truncation2: ",(N[1]-M[1])/N[1])
    =#

    return P1,P2
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
            #----------- Single Layer
            #---- Construct Projector: will be deleted
            #---- Extract Environment Tensor: cij and cinextj
            cij1 = CornerEnvironment[i,j][1];cij2 = CornerEnvironment[i,j][2]
            cij3 = CornerEnvironment[i,j][3];cij4 = CornerEnvironment[i,j][4]
            eij1 = EdgeEnvironment[i,j][1];eij2 = EdgeEnvironment[i,j][2]
            eij3 = EdgeEnvironment[i,j][3];eij4 = EdgeEnvironment[i,j][4]

            cijnext1 = CornerEnvironment[i,jnext][1];cijnext2 = CornerEnvironment[i,jnext][2]
            cijnext3 = CornerEnvironment[i,jnext][3];cijnext4 = CornerEnvironment[i,jnext][4]
            eijnext1 =   EdgeEnvironment[i,jnext][1];eijnext2 =   EdgeEnvironment[i,jnext][2]
            eijnext3 =   EdgeEnvironment[i,jnext][3];eijnext4 =   EdgeEnvironment[i,jnext][4]

            a = UnitCell[i,j];b = UnitCell[i,jnext]
            #---- Construct Projector : Approximate the environment 

            P1,Pdagger1,P2,Pdagger2 = ConstructProjectorSingleLayer(cij1,cij2,cij3,cij4,eij1,eij2,eij3,eij4,
                                    cijnext1,cijnext2,cijnext3,cijnext4,eijnext1,eijnext2,eijnext3,eijnext4,
                                    a,b,chimax)

            #---- Update Corner Environment

            @tensor CornerEnvironment[inext,j][4][-1,-2] := CornerEnvironment[i,j][4][2,1]*EdgeEnvironment[i,j][4][-1,5,3,2]*Pdagger1[1,3,4]*Pdagger2[4,5,-2]
            @tensor CornerEnvironment[inext,jnext][1][-1,-2] := CornerEnvironment[i,jnext][1][1,2]*EdgeEnvironment[i,jnext][2][2,5,3,-2]*P1[1,3,4]*P2[4,5,-1]
            CornerEnvironment[inext,j][4][:]= CornerEnvironment[inext,j][4]/maximum(CornerEnvironment[inext,j][4])
            CornerEnvironment[inext,jnext][1][:] = CornerEnvironment[inext,jnext][1]/maximum(CornerEnvironment[inext,jnext][1])

            append!(isometryd1,[P1])
            append!(isometryd2,[P2])
            append!(isometrydaggerd1,[Pdagger1])
            append!(isometrydaggerd2,[Pdagger2])
            

            #---- Update Edge Environment
            if j > 1
                #
                @tensor EdgeEnvironment[inext,j][1][-1,-2,-3,-4] := EdgeEnvironment[i,j][1][1,7,2,4]*isometryd1[end][1,3,6]*isometryd2[end][6,8,-1]*
                                isometrydaggerd1[end-1][4,5,10]*isometrydaggerd2[end-1][10,11,-4]*UnitCell[i,j][9,5,2,3,-3]*UnitCell[i,j][9,11,7,8,-2]
                #=
                @tensor EdgeEnvironment[inext,j][1][:] := EdgeEnvironment[i,j][1][1,8,3,4]*isometryd1[end][1,2,6]*isometryd2[end][6,7,-1]*
                    isometrydaggerd1[end-1][4,5,11]*isometrydaggerd2[end-1][11,10,-4]*UnitCell[i,j][9,5,3,2,-3]*UnitCell[i,j][9,10,8,7,-2]
                =#

                EdgeEnvironment[inext,j][1][:] = EdgeEnvironment[inext,j][1]/maximum(EdgeEnvironment[inext,j][1])
            end
            if j == y
                @tensor EdgeEnvironment[inext,1][1][-1,-2,-3,-4] := EdgeEnvironment[i,1][1][1,7,2,4]*isometryd1[1][1,3,6]*isometryd2[1][6,8,-1]*
                                isometrydaggerd1[end][4,5,10]*isometrydaggerd2[end][10,11,-4]*UnitCell[i,1][9,5,2,3,-3]*UnitCell[i,1][9,11,7,8,-2]
                EdgeEnvironment[inext,1][1][:] = EdgeEnvironment[inext,1][1]/maximum(EdgeEnvironment[inext,1][1])
            end

        end
    end



    #!    Test: The same tensor in the unit cell has the same environment
    #=
    param.CornerEnvironment[2,2] = deepcopy(param.CornerEnvironment[1,1])
    param.EdgeEnvironment[2,2] = deepcopy(param.EdgeEnvironment[1,1])
    param.CornerEnvironment[2,1] = deepcopy(param.CornerEnvironment[1,2])
    param.EdgeEnvironment[2,1] = deepcopy(param.EdgeEnvironment[1,2])
    =#
    return CornerEnvironment,EdgeEnvironment
end



function LeftMover2x1UnitCell(UnitCell::Array{Array{Float64}},CornerEnvironment::Array{Array{Array{Float64}}},
                EdgeEnvironment::Array{Array{Array{Float64}}},chimax::Int64,Dp::Int64,constructprojector::Function)
    x = size(UnitCell,1);y = size(UnitCell,2)
    for i in 1:x
        isometry = Array{Array{Float64}}(undef,0)
        isometrydagger = Array{Array{Float64}}(undef,0)
        for j in 1:y
            i == x ? inext = 1 : inext = i+1; j == y ? jnext = 1 : jnext = j+1 # define inext and jnext
            #----------- Single Layer
            #---- Construct Projector: will be deleted
            #---- Extract Environment Tensor: cij and cinextj
            cij1 = CornerEnvironment[i,j][1];cij2 = CornerEnvironment[i,j][2]
            cij3 = CornerEnvironment[i,j][3];cij4 = CornerEnvironment[i,j][4]
            eij1 = EdgeEnvironment[i,j][1];eij2 = EdgeEnvironment[i,j][2]
            eij3 = EdgeEnvironment[i,j][3];eij4 = EdgeEnvironment[i,j][4]

            cijnext1 = CornerEnvironment[i,jnext][1];cijnext2 = CornerEnvironment[i,jnext][2]
            cijnext3 = CornerEnvironment[i,jnext][3];cijnext4 = CornerEnvironment[i,jnext][4]
            eijnext1 =   EdgeEnvironment[i,jnext][1];eijnext2 =   EdgeEnvironment[i,jnext][2]
            eijnext3 =   EdgeEnvironment[i,jnext][3];eijnext4 =   EdgeEnvironment[i,jnext][4]
            a = UnitCell[i,j];b = UnitCell[i,jnext]
            #---- Update Corner Environment
            P,Pdagger = constructprojector(cij1,cij2,cij3,cij4,eij1,eij2,eij3,eij4,a,chimax)

            #------   Update CornerEnvironment
            @tensor CornerEnvironment[inext,j][4][:] := CornerEnvironment[i,j][4][2,1]*EdgeEnvironment[i,j][4][-1,3,4,2]*Pdagger[1,3,4,-2]
            @tensor CornerEnvironment[inext,jnext][1][:] :=CornerEnvironment[i,jnext][1][1,2]*EdgeEnvironment[i,jnext][2][2,3,4,-2]*P[1,3,4,-1]
            CornerEnvironment[inext,j][4][:]= CornerEnvironment[inext,j][4]/maximum(CornerEnvironment[inext,j][4])
            CornerEnvironment[inext,jnext][1][:] = CornerEnvironment[inext,jnext][1]/maximum(CornerEnvironment[inext,jnext][1])
            append!(isometry,[P])
            append!(isometrydagger,[Pdagger])

            if j > 1 
                @tensor  EdgeEnvironment[inext,j][1][-1,-2,-3,-4] := EdgeEnvironment[i,j][1][1,4,2,7]*isometry[end][1,5,3,-1]*isometrydagger[end-1][7,8,9,-4]*
                                            UnitCell[i,j][6,9,2,3,-3]*UnitCell[i,j][6,8,4,5,-2]
                EdgeEnvironment[inext,j][1][:] = EdgeEnvironment[inext,j][1]/maximum(EdgeEnvironment[inext,j][1])
            end
            if j == y
                @tensor  EdgeEnvironment[inext,1][1][-1,-2,-3,-4] := EdgeEnvironment[i,1][1][1,4,2,7]*isometry[1][1,5,3,-1]*isometrydagger[end][7,8,9,-4]*
                                UnitCell[i,1][6,9,2,3,-3]*UnitCell[i,1][6,8,4,5,-2]
                EdgeEnvironment[inext,1][1][:] = EdgeEnvironment[inext,1][1]/maximum(EdgeEnvironment[inext,1][1])

            end
        end
    end


    #!    Test: The same tensor in the unit cell has the same environment
    #=
    param.CornerEnvironment[2,2] = deepcopy(param.CornerEnvironment[1,1])
    param.EdgeEnvironment[2,2] = deepcopy(param.EdgeEnvironment[1,1])
    param.CornerEnvironment[2,1] = deepcopy(param.CornerEnvironment[1,2])
    param.EdgeEnvironment[2,1] = deepcopy(param.EdgeEnvironment[1,2])
    =#
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
function mover(param::PEPSParam,paramc::PEPSParam,taoind::Int64,chiind::Int64,Dind::Int64;direction=1::Int64)
    x = size(param.UnitCell,1); y = size(param.UnitCell,2)
    #!   permarg stores the permute parameter that is used to different direction movers
    @match direction begin
        #
        1 => (permarg= [[1,2],[1:1:x,1:1:y],[1,2,3,4,5],[1,2,3,4],[1,2,3,4,5],[1,2,3,4]])
        2 => (permarg= [[2,1],[x:-1:1,1:1:y],[1,5,2,3,4],[2,3,4,1],[1,3,4,5,2],[4,1,2,3]])
        3 => (permarg= [[1,2],[x:-1:1,y:-1:1],[1,4,5,2,3],[3,4,1,2],[1,4,5,2,3],[3,4,1,2]])
        4 => (permarg= [[2,1],[1:1:x,y:-1:1],[1,3,4,5,2],[4,1,2,3],[1,5,2,3,4],[2,3,4,1]])
        #
    end

    #=
    UnitCell = deepcopy(param.UnitCell)
    permutedims!(paramc.UnitCell,param.UnitCell,permarg[1])
    paramc.UnitCell[:] = paramc.UnitCell[permarg[2]...]
    permutedims!(paramc.CornerEnvironment,param.CornerEnvironment,permarg[1])
    paramc.CornerEnvironment[:] = paramc.CornerEnvironment[permarg[2]...]
    permutedims!(paramc.EdgeEnvironment,param.EdgeEnvironment,permarg[1])
    paramc.EdgeEnvironment[:] = paramc.EdgeEnvironment[permarg[2]...]
    =#

    #! permutedims the environment such that they are compatible with mover directions
    paramc.UnitCell[:] = permutedims(param.UnitCell,permarg[1])[permarg[2]...]
    paramc.CornerEnvironment[:] = permutedims(param.CornerEnvironment,permarg[1])[permarg[2]...]
    paramc.EdgeEnvironment[:] = permutedims(param.EdgeEnvironment,permarg[1])[permarg[2]...]
    PermuteEnvironment(paramc,permarg[3],permarg[4])
    #paramc.UnitCell,paramc.CornerEnvironment,paramc.EdgeEnvironment = PermuteEnvironment(paramc.UnitCell,paramc.CornerEnvironment,
    #                                                paramc.EdgeEnvironment,permarg[3],permarg[4])

    #! apply moves in specific directions according to which unitcell and projector types
    if param.MoverType == "2x2UnitCell"
        CornerEnvironmenttemp,EdgeEnvironmenttemp= LeftMover2x2UnitCell(paramc.UnitCell,paramc.CornerEnvironment,paramc.EdgeEnvironment,
                                            paramc.chimax[chiind],param.ProjectorFunction)
    elseif param.MoverType == "Isometry"
        CornerEnvironmenttemp,EdgeEnvironmenttemp= LeftMoverUCIsometry(UnitCelltemp,CornerEnvironmenttemp,
                                            EdgeEnvironmenttemp,param.chimax,param.Dp,param.ProjectorFunction)
    elseif param.MoverType == "SingleLayer"
        LeftMoverSingleLayer(paramc.UnitCell,paramc.CornerEnvironment,
                        paramc.EdgeEnvironment,paramc.chimax[chiind],paramc.Dlink[Dind],paramc.ProjectorFunction)
    elseif param.MoverType == "2x1UnitCell"
        LeftMover2x1UnitCell(paramc.UnitCell,paramc.CornerEnvironment,
                        paramc.EdgeEnvironment,paramc.chimax[chiind],paramc.Dlink[Dind],paramc.ProjectorFunction) 
    end


    paramc.UnitCell,paramc.CornerEnvironment,paramc.EdgeEnvironment = PermuteEnvironment(paramc.UnitCell,paramc.CornerEnvironment,
                                            paramc.EdgeEnvironment,permarg[5],permarg[6])
    #permutedims!(param.CornerEnvironment,paramc.CornerEnvironment[permarg[2]...],permarg[1])
    #permutedims!(param.EdgeEnvironment,paramc.EdgeEnvironment[permarg[2]...],permarg[1])
    
    #! for movers, we don't need to change the param.UnitCell since the unit cell will not change in this step
    #PermuteEnvironment(paramc,permarg[5],permarg[6])
    param.CornerEnvironment[:] = permutedims(paramc.CornerEnvironment[permarg[2]...],permarg[1])
    param.EdgeEnvironment[:] = permutedims(paramc.EdgeEnvironment[permarg[2]...],permarg[1])     
    
end
















"""
    Update environment after increase the virtual bond dimension of iPEPS
"""
function Mover_bond_increase(UnitCell,CornerEnvironment,EdgeEnvironment,chimax) 

    # unit cell
    A = UnitCell[1,1]
    B = UnitCell[1,2]

    # environment
    C1 = CornerEnvironment[1,1][1]
    C2 = CornerEnvironment[1,1][2]
    C3 = CornerEnvironment[1,2][3]
    C4 = CornerEnvironment[1,2][4]

    E1 = EdgeEnvironment[1,2][1]
    E2 = EdgeEnvironment[1,1][1]        
    E3 = EdgeEnvironment[1,1][2]
    E5 = EdgeEnvironment[1,1][3]
    E6 = EdgeEnvironment[1,2][3]
    E8 = EdgeEnvironment[1,2][4]

    chitemp = chimax
    # compute projectors to update environment
    P_up_left_1,P_up_left_2 = compute_projector(C1,E2,E1,C4,E8,C3,E6,chitemp)
    P_dn_left_1,P_dn_left_2 = compute_projector(permutedims(C4,[2,1]),permutedims(E1,[4,2,3,1]),permutedims(E2,[4,2,3,1]),
                    permutedims(C1,[2,1]),permutedims(E3,[4,2,3,1]),permutedims(C2,[2,1]),permutedims(E5,[4,2,3,1]),chitemp)
    P_up_right_1,P_up_right_2 = compute_projector(permutedims(C2,[2,1]),permutedims(E5,[4,2,3,1]),permutedims(E6,[4,2,3,1]),
                    permutedims(C3,[2,1]),permutedims(E8,[4,2,3,1]),permutedims(C4,[2,1]),permutedims(E1,[4,2,3,1]),chitemp)
    P_dn_right_1,P_dn_right_2 = compute_projector(C3,E6,E5,C2,E3,C1,E2,chitemp)


    @tensor C1new[:] := C1[1,2]*E2[-1,5,3,1]*P_up_left_1[2,3,4]*P_up_left_2[4,5,-2]
    @tensor E2new[:] := E3[1,8,3,4]*A[9,3,2,-3,5]*A[9,8,7,-2,11]*P_up_left_1[1,2,6]*P_up_left_2[6,7,-1]*P_up_right_1[4,5,10]*P_up_right_2[10,11,-4]
    @tensor C2new[:] := C2[2,1]*E5[1,5,3,-2]*P_up_right_1[2,3,4]*P_up_right_2[4,5,-1]


    @tensor C3new[:] := C3[3,1]*E6[-1,5,2,3]*P_dn_right_1[1,2,4]*P_dn_right_2[4,5,-2]
    @tensor C4new[:] := C4[2,1]*E1[1,5,3,-2]*P_dn_left_1[2,3,4]*P_dn_left_2[4,5,-1]
    @tensor E4new[:] := E8[4,8,2,1]*B[9,-3,3,2,5]*B[9,-2,11,8,7]*P_dn_right_1[4,5,6]*P_dn_right_2[6,7,-1]*P_dn_left_1[1,3,10]*P_dn_left_2[10,11,-4]

    CornerEnvironment[1,2][1] = deepcopy(C1new)
    CornerEnvironment[1,2][2] = deepcopy(C2new)
    EdgeEnvironment[1,2][2] = deepcopy(E2new)

    CornerEnvironment[1,1][3] = deepcopy(C3new) 
    CornerEnvironment[1,1][4] = deepcopy(C4new)
    EdgeEnvironment[1,1][4] = deepcopy(E4new)

    CornerEnvironment[2,1] = deepcopy(CornerEnvironment[1,2])
    CornerEnvironment[2,2] = deepcopy(CornerEnvironment[1,1])
    EdgeEnvironment[2,1] = deepcopy(EdgeEnvironment[1,2])
    EdgeEnvironment[2,2] = deepcopy(EdgeEnvironment[1,1])

    return CornerEnvironment,EdgeEnvironment

end



"""
    mover_increase(param::PEPSParam;direction=1::Int64)

Mover for computing the CTM environment.
    direction == 1   :    Right Mover
    direction == 2   :    Up Mover
    direction == 3   :    Left Mover
    direction == 4   :    Down Mover
"""
function mover_increase(param::PEPSParam,taoind::Int64,chiind::Int64,Dind::Int64;direction=1::Int64)
    x = size(param.UnitCell,1); y = size(param.UnitCell,2)
    #   permarg stores the permute parameter that is used to different direction movers
    @match direction begin
        4 => (permarg= [[1,2],[1:1:x,1:1:y],[1,2,3,4,5],[1,2,3,4],[1,2,3,4,5],[1,2,3,4]])
        1 => (permarg= [[2,1],[x:-1:1,1:1:y],[1,5,2,3,4],[2,3,4,1],[1,3,4,5,2],[4,1,2,3]])
        2 => (permarg= [[1,2],[x:-1:1,y:-1:1],[1,4,5,2,3],[3,4,1,2],[1,4,5,2,3],[3,4,1,2]])
        3 => (permarg= [[2,1],[1:1:x,y:-1:1],[1,3,4,5,2],[4,1,2,3],[1,5,2,3,4],[2,3,4,1]])
    end

    UnitCelltemp = permutedims(param.UnitCell,permarg[1])[permarg[2]...]
    CornerEnvironmenttemp = permutedims(param.CornerEnvironment,permarg[1])[permarg[2]...]
    EdgeEnvironmenttemp = permutedims(param.EdgeEnvironment,permarg[1])[permarg[2]...]
    UnitCelltemp,CornerEnvironmenttemp,EdgeEnvironmenttemp = PermuteEnvironment(UnitCelltemp,CornerEnvironmenttemp,
                                            EdgeEnvironmenttemp,permarg[3],permarg[4])

    CornerEnvironmenttemp,EdgeEnvironmenttemp = Mover_bond_increase(UnitCelltemp,CornerEnvironmenttemp,EdgeEnvironmenttemp,
                                                param.chimax[chiind])
    UnitCelltemp,CornerEnvironmenttemp,EdgeEnvironmenttemp = PermuteEnvironment(UnitCelltemp,CornerEnvironmenttemp,
                                            EdgeEnvironmenttemp,permarg[5],permarg[6])
    param.CornerEnvironment = permutedims(CornerEnvironmenttemp[permarg[2]...],permarg[1])
    param.EdgeEnvironment = permutedims(EdgeEnvironmenttemp[permarg[2]...],permarg[1])


    #return CornerEnvironment,EdgeEnvironment
end





function ConstructProjectorTwoToOne(c1,c2,c3,c4,e1,e2,e3,e4,chitemp)

    D = size(e2,2);
    chitemp = 2*D;
    @tensor uptemp[:] := c4[2,1]*c4[3,1]*e4[6,4,5,2]*e4[7,4,5,3]*c3[-1,6]*c3[-2,7];
    @tensor dntemp[:] := e1[1,2,3,4]*e1[1,2,3,5]*c1[4,-1]*c1[5,-2];
    uptemp = uptemp/maximum(uptemp);    
    dntemp = dntemp/maximum(dntemp);

    @tensor Env_E2[:] := uptemp[1,2]*dntemp[9,8]*e3[5,3,4,1]*e3[6,3,4,2]*c2[7,5]*c2[10,6]*e2[9,-1,-2,7]*e2[8,-3,-4,10];
    sizeEnv_E2 = size(Env_E2);
    Rlt = svd(reshape(Env_E2,prod(sizeEnv_E2[1:2]),prod(sizeEnv_E2[3:4])));
    P2 = reshape(Rlt.U[:,1:chitemp],sizeEnv_E2[1],sizeEnv_E2[2],chitemp);


    @tensor Env_E3[:] := uptemp[9,8]*dntemp[1,2]*e3[7,-1,-2,9]*e3[10,-3,-4,8]*c2[5,7]*c2[6,10]*e2[1,3,4,5]*e2[2,3,4,6];
    sizeEnv_E3 = size(Env_E3);
    Rlt = svd(reshape(Env_E3,prod(sizeEnv_E3[1:2]),prod(sizeEnv_E3[3:4])));
    P3 = reshape(Rlt.U[:,1:chitemp],sizeEnv_E3[1],sizeEnv_E3[2],chitemp);

    #
    # compute another projector
    #
    P = rand(size(P3,3),size(P2,3),4*D);
    for j in 1:15
        @tensor Env_P[:] := uptemp[4,18]*dntemp[7,17]*e3[3,1,2,4]*c2[8,3]*e2[7,5,6,8]*
                    e3[13,11,12,18]*c2[16,13]*e2[17,14,15,16]*P3[1,2,9]*P3[11,12,-1]*
                    P2[5,6,10]*P2[14,15,-2]*P[9,10,-3]
        Rlt = svd(reshape(Env_P,prod(size(Env_P)[1:2]),size(Env_P,3)))
        P = reshape(Rlt.U*Rlt.V',size(Env_P)...)
    end

    #=
    @tensor phiphi[:] := uptemp[4,8]*dntemp[12,15]*e2[12,10,11,17]*c2[17,3]*e3[3,1,2,4]*e2[15,13,14,18]*
                    c2[18,7]*e3[7,5,6,8]*P2[10,11,16]*P2[13,14,16]*P3[1,2,9]*P3[5,6,9];
    @tensor psipsi[:] := uptemp[10,19]*dntemp[9,21]*e2[9,4,5,6]*c2[6,3]*e3[3,1,2,10]*e2[21,14,15,16]*
                    c2[16,13]*e3[13,11,12,19]*P[7,8,20]*P[17,18,20]*P2[4,5,8]*P2[14,15,18]*P3[1,2,7]*P3[11,12,17];
    =#

    @tensor DATemp[:] := A[7,1,-2,-4,2]*A[7,5,-1,-3,6]*P3[6,2,4]*P2[5,1,3]*P[4,3,-5]
    @tensor TEMP[:] := e1[-1,11,12,10]*c1[10,9]*e2[9,4,5,6]*c2[6,3]*e3[3,1,2,-2]*P[8,7,13]*
                    DATemp[11,12,-3,-4,13]*P3[1,2,8]*P2[4,5,7]
    TEMP = TEMP/maximum(TEMP)
    @tensor Env_Proj_1[:] := TEMP[9,10,-1,-2]*TEMP[9,8,-3,-4]*c3[10,6]*e4[6,4,5,2]*c4[2,1]*
                    c3[8,7]*e4[7,4,5,3]*c4[3,1]
    Rlt = svd(reshape(Env_Proj_1,prod(size(Env_Proj_1)[1:2]),prod(size(Env_Proj_1)[3:4])))
    Proj_1 = reshape(Rlt.U[:,1:4*D],size(Env_Proj_1,1),size(Env_Proj_1,2),4*D)
    
    @tensor Env_Proj_2[:] := TEMP[-1,13,11,12]*TEMP[-3,10,8,9]*c3[13,6]*e4[6,4,5,2]*c4[2,1]*
                    c3[10,7]*e4[7,4,5,3]*c4[3,1]*Proj_1[11,12,-2]*Proj_1[8,9,-4]
    Rlt = svd(reshape(Env_Proj_2,prod(size(Env_Proj_2[1:2]),prod(size(Env_Proj_2)[3:4]))))
    Proj_2 = reshape(Rlt.U[:,1:chimax],size(Env_Proj_2,1),size(Env_Proj_2,2),chimax)

    Proj_1_dagger = deepcopy(Proj_1)
    Proj_2_dagger = deepcopy(Proj_2)
    #
    return Proj_1,Proj_1_dagger,Proj_2,Proj_2_dagger

end






function LeftMoverTwoToOne(UnitCell::Array{Array{Float64}},CornerEnvironment::Array{Array{Array{Float64}}},
    EdgeEnvironment::Array{Array{Array{Float64}}},chimax::Int64,Dp::Int64,constructprojector::Function)


    x = size(UnitCell,1);y = size(UnitCell,2)
    for i in 1:x
        isometryd1 = Array{Array{Float64}}(undef,0)
        isometrydaggerd1 = Array{Array{Float64}}(undef,0)
        isometryd2 = Array{Array{Float64}}(undef,0)
        isometrydaggerd2 = Array{Array{Float64}}(undef,0)
        for j in 1:y
            i == x ? inext = 1 : inext = i+1; j == y ? jnext = 1 : jnext = j+1 # define inext and jnext
            #----------- Single Layer
            #---- Construct Projector: will be deleted
            #---- Extract Environment Tensor: cij and cinextj
            cij1 = CornerEnvironment[i,j][1];cij2 = CornerEnvironment[i,j][2]
            cij3 = CornerEnvironment[i,j][3];cij4 = CornerEnvironment[i,j][4]
            eij1 = EdgeEnvironment[i,j][1];eij2 = EdgeEnvironment[i,j][2]
            eij3 = EdgeEnvironment[i,j][3];eij4 = EdgeEnvironment[i,j][4]

            cijnext1 = CornerEnvironment[i,jnext][1];cijnext2 = CornerEnvironment[i,jnext][2]
            cijnext3 = CornerEnvironment[i,jnext][3];cijnext4 = CornerEnvironment[i,jnext][4]
            eijnext1 =   EdgeEnvironment[i,jnext][1];eijnext2 =   EdgeEnvironment[i,jnext][2]
            eijnext3 =   EdgeEnvironment[i,jnext][3];eijnext4 =   EdgeEnvironment[i,jnext][4]

            a = UnitCell[i,j];b = UnitCell[i,jnext]
            #---- Construct Projector : Approximate the environment 
            P1,Pdagger1,P2,Pdagger2 = ConstructProjectorSingleLayer(cij1,cij2,cij3,cij4,eij1,eij2,eij3,eij4,
                                    cijnext1,cijnext2,cijnext3,cijnext4,eijnext1,eijnext2,eijnext3,eijnext4,
                                    a,b,chimax)
            #---- Update Corner Environment

            @tensor CornerEnvironment[inext,j][4][-1,-2] := CornerEnvironment[i,j][4][3,4]*EdgeEnvironment[i,j][4][-1,1,2,3]*Pdagger1[1,2,5]*Pdagger2[4,5,-2]
            @tensor CornerEnvironment[inext,jnext][1][-1,-2] := CornerEnvironment[i,jnext][1][4,3]*EdgeEnvironment[i,jnext][2][3,1,2,-2]*P1[1,2,5]*P2[4,5,-1]
            CornerEnvironment[inext,j][4][:]= CornerEnvironment[inext,j][4]/maximum(CornerEnvironment[inext,j][4])
            CornerEnvironment[inext,jnext][1][:] = CornerEnvironment[inext,jnext][1]/maximum(CornerEnvironment[inext,jnext][1])

            append!(isometryd1,[P1])
            append!(isometryd2,[P2])
            append!(isometrydaggerd1,[Pdagger1])
            append!(isometrydaggerd2,[Pdagger2])
            
            #---- Update Edge Environment
            if j > 1
                #
                @tensor EdgeEnvironment[inext,j][1][-1,-2,-3,-4] := EdgeEnvironment[i,j][1][8,6,7,10]*isometryd1[end][3,2,9]*isometryd2[end][8,9,-1]*
                                isometrydaggerd1[end-1][4,1,11]*isometrydaggerd2[end-1][10,11,-4]*UnitCell[i,j][5,1,7,2,-3]*UnitCell[i,j][5,4,6,3,-2]
                EdgeEnvironment[inext,j][1][:] = EdgeEnvironment[inext,j][1]/maximum(EdgeEnvironment[inext,j][1])
            end
            if j == y
                @tensor EdgeEnvironment[inext,1][1][-1,-2,-3,-4] := EdgeEnvironment[i,1][1][8,6,7,10]*isometryd1[1][3,2,9]*isometryd2[1][8,9,-1]*
                                isometrydaggerd1[end][4,1,11]*isometrydaggerd2[end][10,11,-4]*UnitCell[i,1][5,1,7,2,-3]*UnitCell[i,1][5,4,6,3,-2]
                EdgeEnvironment[inext,1][1][:] = EdgeEnvironment[inext,1][1]/maximum(EdgeEnvironment[inext,1][1])
            end
        end
    end


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
