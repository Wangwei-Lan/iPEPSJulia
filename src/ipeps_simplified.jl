function iPEPS(param::PEPSParam,paramc::PEPSParam)
    # SECTION : Simple update
    if param.SimpleUpdateOn == "true"
        println("Start Simple Update")
        #--- Initial tensor; Only works for AB sub-lattice case; # TODO : should rewrite for general case
        A = rand(Float64,(param.Dlink.*(1,1,1,1)...,(param.dphy)))*1.0e-2
        B = rand(Float64,(param.Dlink.*(1,1,1,1)...,(param.dphy)))*1.0e-2
        A[1,1,1,1,1] = 1.0
        B[1,1,1,1,2] = 1.0
        
        A,B = SimpleUpdate(A,B,param.g[1],param.MaxSimpLoop,param.SimpTol)
        # permutedims to make simple update formalism and full update formalism match
        # TODO : This should rewrite later
        A = permutedims(A,[5,1,2,3,4])
        B = permutedims(B,[5,1,2,3,4])
        param.UnitCell[1,1] = deepcopy(A)
        param.UnitCell[2,2] = deepcopy(A)
        param.UnitCell[1,2] = deepcopy(B)
        param.UnitCell[2,1] = deepcopy(B)
    end

    ComputeEnvironmentUC(param,100,1.0e-6)
    energy = ComputeEnergy(param)
    append!(param.EnergyArray,[sum(energy)/4])
    #energy = ComputeEnergy(A,B,param.param.CornerEnvironment,param.param.EdgeEnvironment,param.Hamiltonian)
    #append!(param.EnergyArray,[sum(energy)/4])
    #print("  Simple Update Energy: $(sum(energy)/4/2)")
    #
    println("Start Optimization")
    # SECTION: full update
    for k in 1:param.MaxLoop
        println("-------------------------RG step $k : -----------------------------")
        ind = argmax(k .< param.step) # check which tao we should use
        
        x = size(param.UnitCell,1),y = size(param.UnitCell,2)                           # UnitCell size
        #--- Loop over the entire UnitCell
        for i in 1:x
            isometry = Array{Array{Float64}}(undef,0)
            isometrydagger = Array{Array{Float64}}(undef,0)
            for j in 1:y
                i == x ? inext = 1 : inext = i+1; j == y ? jnext = 1 : jnext = j+1       # define inext and jnext
                i == 1 ? ipre = x: ipre = i-1; j == 1 ? jpre = y : jpre = j-1            # define ipre and jpre
                # NOTE:  use 2x2 sub unit cell to update or construct the environment
                # extract the environment tensor for 2x2 sub unit cell
                # c1 = param.CornerEnvironment[i,j][1];c2=param.CornerEnvironment[inext,j][2];c3=param.CornerEnvironment[inext,jnext][3];c4=param.CornerEnvironment[i,jnext][4]
                # e1 = param.EdgeEnvironment[i,jnext][1];e2 = param.EdgeEnvironment[i,j][1];e3 = param.EdgeEnvironment[i,j][2];e4 = param.EdgeEnvironment[inext,j][2];
                # e5 = param.EdgeEnvironment[inext,j][3];e6 = param.EdgeEnvironment[inext,jnext][3];e7 = param.EdgeEnvironment[inext,jnext][4];e8 = param.EdgeEnvironment[i,jnext][4]
                a = UnitCell[i,j];b = UnitCell[i,jnext]
                c1 = param.CornerEnvironment[i,j][1];c2 = param.CornerEnvironment[i,j][2]
                c3 = param.CornerEnvironment[i,jnext][3];c4 = param.CornerEnvironment[i,jnext][4]
                e1 = param.EdgeEnvironment[i,jnext][1];e2 = param.EdgeEnvironment[i,j][1]
                e3 = param.EdgeEnvironment[i,j][2];e5 = param.EdgeEnvironment[i,j][3]
                e6 = param.EdgeEnvironment[i,jnext][3]; e8 = param.EdgeEnvironment[i,jnext][4]
                #----- Down (relative to A) Update
                update(a,b,c1,c2,c3,c4,e1,e2,e3,e5,e6,e8,param.g[ind])
                
                
                #---- Construct Projector
                @tensor param.EdgeEnvironment[i,jnext][2][:] =  param.EdgeEnvironment[i,j]

                @tensor param.EdgeEnvironment[i,j][4][:] =
                


                #----- Right (relative to A) Update
                update()

                @tensor param.EdgeEnvironment[]


                @tensor param.EdgeEnvironment[]

                
                #=
                P,Pdagger = constructprojector(c1,c2,c3,c4,e1,e2,e3,e4,e5,e6,e7,e8,a,b,chimax)
                @tensor param.CornerEnvironment[inext,j][4][-1,-2] = param.CornerEnvironment[i,j][4][2,1]*param.EdgeEnvironment[i,j][4][-1,3,4,2]*Pdagger[1,3,4,-2]
                @tensor param.CornerEnvironment[inext,jnext][1][-1,-2] =param.CornerEnvironment[i,jnext][1][1,2]*param.EdgeEnvironment[i,jnext][2][2,3,4,-2]*P[1,3,4,-1]
                param.CornerEnvironment[inext,j][4][:]= param.CornerEnvironment[inext,j][4]/maximum(param.CornerEnvironment[inext,j][4])
                param.CornerEnvironment[inext,jnext][1][:] = param.CornerEnvironment[inext,jnext][1]/maximum(param.CornerEnvironment[inext,jnext][1])
                append!(isometry,[P])
                append!(isometrydagger,[Pdagger])
                if j > 1 
                    @tensor param.EdgeEnvironment[inext,j][1][-1,-2,-3,-4] = param.EdgeEnvironment[i,j][1][1,4,2,7]*isometry[end][1,5,3,-1]*isometrydagger[end-1][7,8,9,-4]*
                                    UnitCell[i,j][6,9,2,3,-3]*UnitCell[i,j][6,8,4,5,-2]
                    param.EdgeEnvironment[inext,j][1][:] = param.EdgeEnvironment[inext,j][1]/maximum(param.EdgeEnvironment[inext,j][1])
                end
                if j == y
                    @tensor param.EdgeEnvironment[inext,1][1][-1,-2,-3,-4] = param.EdgeEnvironment[i,1][1][1,4,2,7]*isometry[1][1,5,3,-1]*isometrydagger[end][7,8,9,-4]*
                                    UnitCell[i,1][6,9,2,3,-3]*UnitCell[i,1][6,8,4,5,-2]
                    param.EdgeEnvironment[inext,1][1][:] = param.EdgeEnvironment[inext,1][1]/maximum(param.EdgeEnvironment[inext,1][1])
                end
                =#

            end
        end
        
        
        
        @time begin
        for i in 1:3
            for j in 1:4
                mover(param,direction=j)
            end
            update(param,paramc,ind,direction=i)
        end
        for i in 4:-1:1
            for j in 1:4
                mover(param,direction=j)
            end
            update(param,paramc,ind,direction=i)
        end
        end




        if k%20 ==0
            ComputeEnvironmentUC(param,50,1.0e-6)
            energy = ComputeEnergy(param)
            append!(param.EnergyArray,[sum(energy)/4])
        end
    end
    #
    return param
end
