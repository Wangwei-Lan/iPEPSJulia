function iPEPS(param::PEPSParam,paramc::PEPSParam)

    #
    #----- Simple update
    if param.SimpleUpdateOn == "true"
        println("Start Simple Update")
        #
        A = rand(Float64,(param.Dlink[1].*(1,1,1,1)...,(param.dphy)))*1.0e-2
        B = rand(Float64,(param.Dlink[1].*(1,1,1,1)...,(param.dphy)))*1.0e-2
        #
        #A[1,1,1,1,1] = 1.0
        #B[1,1,1,1,1] = 1.0
        #
        #g = reshape(reshape(param.Hamiltonian,4,4))
        @time A,B = SimpleUpdate(A,B,param.g[1],param.MaxSimpLoop,param.SimpTol)
        #

        #
        g = reshape((reshape(param.g[1],param.dphy^2,param.dphy^2)^(1/10)),param.dphy,param.dphy,param.dphy,param.dphy)[:,:,:,:]
        @time A,B = SimpleUpdate(A,B,g,param.MaxSimpLoop,param.SimpTol)
        #
        g = reshape((reshape(param.g[1],param.dphy^2,param.dphy^2)^(1/100)),param.dphy,param.dphy,param.dphy,param.dphy)[:,:,:,:]
        @time A,B = SimpleUpdate(A,B,g,param.MaxSimpLoop,param.SimpTol)
        #
        #
        g = reshape((reshape(param.g[1],param.dphy^2,param.dphy^2)^(1/200)),param.dphy,param.dphy,param.dphy,param.dphy)[:,:,:,:]
        @time A,B = SimpleUpdate(A,B,g,param.MaxSimpLoop,param.SimpTol)
        #

        A = permutedims(A,[5,1,2,3,4])
        B = permutedims(B,[5,1,2,3,4])
        param.UnitCell[1,1] = deepcopy(A)
        param.UnitCell[2,2] = deepcopy(A)
        param.UnitCell[1,2] = deepcopy(B)
        param.UnitCell[2,1] = deepcopy(B)
        #
    end
    #
    ComputeEnvironmentUC(param,paramc,1,1,1,15,1.0e-6)
    energy = compute_physical(param)
    mag = compute_physical(param,Operator="Sz")
    append!(Magnetization,sum(abs.(mag))/4)
    println("Energy : ",energy )
    println("Energy: ",sum(energy)/4)
    println("magnetism : ",mag)
    append!(param.EnergyArray,[sum(energy)/4])
    plot!(EnergyPlot,[0],[abs.((sum(energy)/4 .-Energy)./Energy)],yscale=:log10,markershape=:x,linestyle=:dashdot,minorgrid=true,grid=true,minorticks=9)
    gui();




    if param.FullUpdateOn == "true"
    

        println("Start Optimization")
        for k in 1:param.MaxLoop
            print("RG step $k :  ")
            print("Bond: $(size(param.UnitCell[1,1])[5])  ")

            taoind = argmax(k .< param.step) # check which tao we should use
            chiind = argmax(k .< param.chistep)
            Dind = argmax(k .< param.Dstep)
            print(taoind," ",chiind," ",Dind," ")
            print(param.tao[taoind],"  ")
            print(param.chimax[chiind],"  ")
            print(param.Dlink[Dind],"  ")
            @time begin

                update_order = [1:1:4...]
                update_order = append!(update_order,[3:-1:1...])


                #TODO: update_order =[1,3,2,4,2,3,1], then we don't need to recalculate the environment too much. we could make our code at least 2 times faster (may be 4 times faster) 
                #! NORMAL FAST FULL UPDATE
                #
                repeat = 1 
                for i in update_order

                    #--------------------------------------------------------------------------------------------
                    #
                    #
                    # Fast Full Update the environment: basically just do mover a few times to update the environment
                    # such that the environment has been updated with nearest sites.
                    #
                    #
                    #--------------------------------------------------------------------------------------------
                    for l in 1:6
                        for j in 1:4
                            mover(param,paramc,taoind,chiind,Dind,direction=j)
                        end
                    end


                    #----------------------------------------------------
                    #
                    #
                    # updat bond according to bond direction
                    #
                    #
                    #----------------------------------------------------
                    update(param,paramc,taoind,chiind,Dind,param.Dlink[Dind],direction=i)



                    #------------------------------------------------------------------------------------
                    
                    #! If Virtual bond dimension increases, recompute environment
                    #! ways to check whether the bond dimension is increased or not: 
                    #! check the environment bond dimension with local tensor bond dimension 
                    #! if they match, then the virtual bond dimension has not been increased 
                    #! otherwise the bond dimension has been incrased!
                    #
                    #
                    #-----------------------------------------------------------------------------------
                    if size(param.UnitCell[1,1],2) != size(param.EdgeEnvironment[1,1][2],2) ||
                                size(param.UnitCell[1,1],3) != size(param.EdgeEnvironment[1,1][1],2) ||
                                    size(param.UnitCell[1,1],4) != size(param.EdgeEnvironment[1,1][4],2) ||
                                        size(param.UnitCell[1,1],5) != size(param.EdgeEnvironment[1,1][3],2)
                        mover_increase(param,taoind,chiind,Dind,direction=i)
                        for m in 1:15
                        for j in 1:4
                            mover(param,paramc,taoind,chiind,Dind,direction=j)
                        end
                        end
                    end
                end
            end



            #! print out result during update steps 
            if k%10 == 0
                if param.chimax[chiind] != size(param.CornerEnvironment[1,1][1],1)
                     reset_parameter(param,param.chimax[chiind],param.Dlink[Dind])
                end
                paramc = deepcopy(param)
                ComputeEnvironmentUC(param,paramc,taoind,chiind,Dind,10,1.0e-6)
                #energy = ComputePhysical(param,Operator="Energy")

                energy = compute_physical(param,Operator="Energy")
                #mag = compute_physical(param,Operator="Sz")
                mag = ComputePhysical(param,Operator="Mag")


                append!(Magnetization,[sum(abs.(mag))/4])
                append!(param.EnergyArray,[sum(energy)/4])
                println(Magnetization)
                println("magnetism: ",mag)
                #println("Energy: ",sum(energy)/4)
                #println(abs.((sum(energy)/4 .-Energy)./Energy))

                push!(EnergyPlot,1,k,abs.((sum(energy)/4 .-Energy)./Energy))
                println(param.EnergyArray)
                gui()
            end

        end
    end
    #
    return param
end
