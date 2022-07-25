function ComputeEnvironmentPlane(a,b,chiini,chimax,envloop,envtol)
    dphy = size(a,1);Dlink = size(a,2)
    c1 = rand(chiini,chiini)
    c2 = c3 = c4 = c1
    e1 = rand(chiini,Dlink,Dlink,chiini)
    e2 = e3 = e4 = e5 = e6 = e7 = e8 = e1
    k=0;numcvg = 0;
    s1last =rand(chi); s2last = rand(chi);s3last = rand(chi);s4last = rand(chi)
    s1last2 = rand(chi);s2last2 = rand(chi);s3last2 = rand(chi);s4last2 = rand(chi)
    for i in 1:envloop
        println("LOOP $i")
        #
        # Directional corner Transfer
        #
        #  Step 1
        c1new,c4new,e1new,e2new = LeftMover(c1,c2,c3,c4,e1,e2,e3,e4,e5,e6,e7,e8,a,b,chimax)
        c3new,c2new,e5new,e6new = RightMover(c3,c4,c1,c2,e5,e6,e7,e8,e1,e2,e3,e4,a,b,chimax)
        c1=c1new;c2=c2new;c3=c3new;c4=c4new;e1=e1new;e2=e2new;e5=e5new;e6=e6new

        c2new,c1new,e4new,e3new = UpMover(c2,c3,c4,c1,e4,e3,e5,e6,e8,e7,e1,e2,a,b,chimax)
        c4new,c3new,e8new,e7new = DownMover(c4,c1,c2,c3,e8,e7,e1,e2,e4,e3,e5,e6,a,b,chimax)
        c1 = c1new;c2 = c2new;c3=c3new;c4=c4new;e3=e3new;e4=e4new;e7=e7new;e8=e8new

        c1new,c4new,e2new,e1new = LeftMover(c1,c2,c3,c4,e2,e1,e4,e3,e6,e5,e8,e7,a,b,chimax)
        c3new,c2new,e6new,e5new = RightMover(c3,c4,c1,c2,e6,e5,e8,e7,e2,e1,e4,e3,a,b,chimax)
        c1 = c1new;c2 = c2new;c3=c3new;c4=c4new;e1=e1new;e2=e2new;e5=e5new;e6=e6new

        c2new,c1new,e3new,e4new = UpMover(c2,c3,c4,c1,e3,e4,e6,e5,e7,e8,e2,e1,a,b,chimax)
        c4new,c3new,e7new,e8new = DownMover(c4,c1,c2,c3,e7,e8,e2,e1,e3,e4,e6,e5,a,b,chimax)
        c1 = c1new;c2 = c2new;c3=c3new;c4=c4new;e3=e3new;e4=e4new;e7=e7new;e8=e8new

        k+=1
        (u1,s1,v1) = LinearAlgebra.svd(c1);(u2,s2,v2) = LinearAlgebra.svd(c2);(u3,s3,v3) = LinearAlgebra.svd(c3);(u4,s4,v4) = LinearAlgebra.svd(c4)
        #append!(S1matrix,[s1/maximum(s1)])
        #append!(S2matrix,[s2/maximum(s2)])
        #append!(S3matrix,[s3/maximum(s3)])
        #append!(S4matrix,[s4/maximum(s4)])
        if length(s1)==length(s1last)&&length(s2)==length(s2last)&&length(s3)==length(s3last)&&length(s4)==length(s4last)&&
            length(s1)==length(s1last2)&&length(s2)==length(s2last2)&&length(s3)==length(s3last2)&&length(s4)==length(s4last2)
            cvgctr1 =norm(abs.(s1/maximum(s1)-s1last))/norm(s1last); cvgctr2 =norm(abs.(s2/maximum(s2)-s2last))/norm(s2last);
            cvgctr3 =norm(abs.(s3/maximum(s3)-s3last))/norm(s3last); cvgctr4 =norm(abs.(s4/maximum(s4)-s4last))/norm(s4last);
            cvgctr12 =norm(abs.(s1/maximum(s1)-s1last2))/norm(s1last2); cvgctr22 =norm(abs.(s2/maximum(s2)-s2last2))/norm(s2last2);
            cvgctr32 =norm(abs.(s3/maximum(s3)-s3last2))/norm(s3last2); cvgctr42 =norm(abs.(s4/maximum(s4)-s4last2))/norm(s4last2);

            print("This is convergence criteria cvgctr1 $k : $cvgctr1 $cvgctr12 \n")
            print("This is convergence criteria cvgctr2 $k : $cvgctr2 $cvgctr22\n")
            print("This is convergence criteria cvgctr3 $k : $cvgctr3 $cvgctr32\n")
            print("This is convergence criteria cvgctr4 $k : $cvgctr4 $cvgctr42\n")
            cvgctr12 < envtol && cvgctr22 < envtol && cvgctr32 < envtol && cvgctr42 < envtol ? (numcvg+=1;break) : (numcvg=0;)
        end
        s1last2 = s1last;s2last2 = s2last;s3last2 = s3last;s4last2=s4last
        s1last = s1/maximum(s1);s2last = s2/maximum(s2);s3last=s3/maximum(s3);s4last=s4/maximum(s4);
        numcvg> 2 ? break : continue
    end
    #push!(updateenvnum,k)
    k == envLoop ? cvgchk = false : cvgchk = true
    print("This is environment step $k   \n")
    return c1,c2,c3,c4,e1,e2,e3,e4,e5,e6,e7,e8,cvgchk
end


function NComputeEnvironment(a,b,chiini,chimax,envloop,envtol)
    dphy = size(a,1);Dlink = size(a,2)
    c1 = rand(chiini,chiini)
    c2 = c3 = c4 = c1
    e1 = rand(chiini,Dlink,Dlink,chiini)
    e2 = e3 = e4 = e5 = e6 = e7 = e8 = e1
    k=0;numcvg = 0;
    s1last =rand(chi); s2last = rand(chi);s3last = rand(chi);s4last = rand(chi)
    s1last2 = rand(chi);s2last2 = rand(chi);s3last2 = rand(chi);s4last2 = rand(chi)
    for i in 1:envloop
        #println("LOOP $i")
        #
        # Directional corner Transfer
        #
        #  Step 1
        c1new,c4new,e1new,e2new = NLeftMoverWithH(c1,c2,c3,c4,e1,e2,e3,e4,e5,e6,e7,e8,a,b,chimax)
        c3new,c2new,e5new,e6new = NRightMoverWithH(c3,c4,c1,c2,e5,e6,e7,e8,e1,e2,e3,e4,a,b,chimax)
        c1=c1new;c2=c2new;c3=c3new;c4=c4new;e1=e1new;e2=e2new;e5=e5new;e6=e6new
        c2new,c1new,e4new,e3new = NUpMoverWithH(c2,c3,c4,c1,e4,e3,e5,e6,e8,e7,e1,e2,a,b,chimax)
        c4new,c1new,e8new,e7new = NDownMoverWithH(c4,c1,c2,c3,e8,e7,e1,e2,e4,e3,e5,e6,a,b,chimax)
        c1 = c1new;c2 = c2new;c3=c3new;c4=c4new;e3=e3new;e4=e4new;e7=e7new;e8=e8new

        c1new,c4new,e2new,e1new = NLeftMoverWithH(c1,c2,c3,c4,e2,e1,e4,e3,e6,e5,e8,e7,a,b,chimax)
        c3new,c2new,e6new,e5new = NRightMoverWithH(c3,c4,c1,c2,e6,e5,e8,e7,e2,e1,e4,e3,a,b,chimax)
        c1 = c1new;c2 = c2new;c3=c3new;c4=c4new;e1=e1new;e2=e2new;e5=e5new;e6=e6new

        c2new,c1new,e3new,e4new = NUpMoverWithH(c2,c3,c4,c1,e3,e4,e6,e5,e7,e8,e2,e1,a,b,chimax)
        c4new,c1new,e7new,e8new = NDownMoverWithH(c4,c1,c2,c3,e7,e8,e2,e1,e3,e4,e6,e5,a,b,chimax)
        c1 = c1new;c2 = c2new;c3=c3new;c4=c4new;e3=e3new;e4=e4new;e7=e7new;e8=e8new

        k+=1
        (u1,s1,v1) = svd(c1);(u2,s2,v2) = svd(c2);(u3,s3,v3) = svd(c3);(u4,s4,v4) = svd(c4)
        #append!(S1matrix,[s1/maximum(s1)])
        #append!(S2matrix,[s2/maximum(s2)])
        #append!(S3matrix,[s3/maximum(s3)])
        #append!(S4matrix,[s4/maximum(s4)])
        if length(s1)==length(s1last)&&length(s2)==length(s2last)&&length(s3)==length(s3last)&&length(s4)==length(s4last)&&
            length(s1)==length(s1last2)&&length(s2)==length(s2last2)&&length(s3)==length(s3last2)&&length(s4)==length(s4last2)
            cvgctr1 =norm(abs.(s1/maximum(s1)-s1last))/norm(s1last); cvgctr2 =norm(abs.(s2/maximum(s2)-s2last))/norm(s2last);
            cvgctr3 =norm(abs.(s3/maximum(s3)-s3last))/norm(s3last); cvgctr4 =norm(abs.(s4/maximum(s4)-s4last))/norm(s4last);
            cvgctr12 =norm(abs.(s1/maximum(s1)-s1last2))/norm(s1last2); cvgctr22 =norm(abs.(s2/maximum(s2)-s2last2))/norm(s2last2);
            cvgctr32 =norm(abs.(s3/maximum(s3)-s3last2))/norm(s3last2); cvgctr42 =norm(abs.(s4/maximum(s4)-s4last2))/norm(s4last2);
            #print("This is convergence criteria cvgctr1 $k : $cvgctr1 $cvgctr12 \n")
            #print("This is convergence criteria cvgctr2 $k : $cvgctr2 $cvgctr22\n")
            #print("This is convergence criteria cvgctr3 $k : $cvgctr3 $cvgctr32\n")
            #print("This is convergence criteria cvgctr4 $k : $cvgctr4 $cvgctr42\n")
            cvgctr12 < envtol && cvgctr22 < envtol && cvgctr32 < envtol && cvgctr42 < envtol ? (numcvg+=1;break) : (numcvg=0;)
        end
        s1last2 = s1last;s2last2 = s2last;s3last2 = s3last;s4last2=s4last
        s1last = s1/maximum(s1);s2last = s2/maximum(s2);s3last=s3/maximum(s3);s4last=s4/maximum(s4);
        numcvg> 2 ? break : continue
    end
    k == envLoop ? cvgchk = false : cvgchk = true
    print("This is environment step $k   \n")
    return c1,c2,c3,c4,e1,e2,e3,e4,e5,e6,e7,e8,cvgchk
end





function ComputeEnvironmentWithH(a,b,c1,c2,c3,c4,e1,e2,e3,e4,e5,e6,e7,e8,mpo,chiini,chimax,envloop,envtol)
    dphy = size(a,1);Dlink = size(a,2)
    k=0;numcvg = 0;
    s1last =rand(chi); s2last = rand(chi);s3last = rand(chi);s4last = rand(chi)
    s1last2 = rand(chi);s2last2 = rand(chi);s3last2 = rand(chi);s4last2 = rand(chi)
    e1Hm=e1;e2Hm=e2;e3Hm=e3;e4Hm=e4;e5Hm=e5;e6Hm=e6;e7Hm=e7;e8Hm=e8
    for i in 1:envloop
        #println("LOOP $i")
        #
        # Directional corner Transfer
        #
        #  Step 1, initial step including hamiltonian term
        if i == 1
            c1new,c4new,e1new,e2new,e1Hmnew,e2Hmnew = NLeftMoverWithH(c1,c2,c3,c4,e1,e2,e3,e4,e5,e6,e7,e8,a,b,chimax,
                                    caltype="HENV",calstep="Initial",e1hm=e1Hm,e2hm=e2Hm,MPO=mpo[1])
            c3new,c2new,e5new,e6new,e5Hmnew,e6Hmnew = NRightMoverWithH(c3,c4,c1,c2,e5,e6,e7,e8,e1,e2,e3,e4,a,b,chimax,
                                    caltype="HENV",calstep="Initial",e1hm=e5Hm,e2hm=e6Hm,MPO=mpo[3])
            c1=c1new;c2=c2new;c3=c3new;c4=c4new;e1=e1new;e2=e2new;e5=e5new;e6=e6new;
            e1Hm=e1Hmnew;e2Hm=e2Hmnew;e5Hm=e5Hmnew;e6Hm=e6Hmnew
            c2new,c1new,e4new,e3new,e4Hmnew,e3Hmnew = NUpMoverWithH(c2,c3,c4,c1,e4,e3,e5,e6,e8,e7,e1,e2,a,b,chimax,
                                    caltype="HENV",calstep="Initial",e1hm=e4Hm,e2hm=e3Hm,MPO=mpo[1])
            c4new,c1new,e8new,e7new,e8Hmnew,e7Hmnew = NDownMoverWithH(c4,c1,c2,c3,e8,e7,e1,e2,e4,e3,e5,e6,a,b,chimax,
                                    caltype="HENV",calstep="Initial",e1hm=e8Hm,e2hm=e7Hm,MPO=mpo[3])
            c1 = c1new;c2 = c2new;c3=c3new;c4=c4new;e3=e3new;e4=e4new;e7=e7new;e8=e8new;
            e4Hm = e4Hmnew;e3Hm = e3Hmnew;e8Hm=e8Hmnew;e7Hm=e7Hmnew
        else
            c1new,c4new,e1new,e2new,e1Hmnew,e2Hmnew = NLeftMoverWithH(c1,c2,c3,c4,e1,e2,e3,e4,e5,e6,e7,e8,a,b,chimax,
                                    caltype="HENV",calstep="Normal",e1hm=e1Hm,e2hm=e2Hm,MPO=mpo[2])
            c3new,c2new,e5new,e6new,e5Hmnew,e6Hmnew = NRightMoverWithH(c3,c4,c1,c2,e5,e6,e7,e8,e1,e2,e3,e4,a,b,chimax,
                                    caltype="HENV",calstep="Normal",e1hm=e5Hm,e2hm=e6Hm,MPO=mpo[2])
            c1=c1new;c2=c2new;c3=c3new;c4=c4new;e1=e1new;e2=e2new;e5=e5new;e6=e6new;
            e1Hm=e1Hmnew;e2Hm=e2Hmnew;e5Hm=e5Hmnew;e6Hm=e6Hmnew
            c2new,c1new,e4new,e3new,e4Hmnew,e3Hmnew = NUpMoverWithH(c2,c3,c4,c1,e4,e3,e5,e6,e8,e7,e1,e2,a,b,chimax,
                                    caltype="HENV",calstep="Normal",e1hm=e4Hm,e2hm=e3Hm,MPO=mpo[2])
            c4new,c1new,e8new,e7new,e8Hmnew,e7Hmnew = NDownMoverWithH(c4,c1,c2,c3,e8,e7,e1,e2,e4,e3,e5,e6,a,b,chimax,
                                    caltype="HENV",calstep="Normal",e1hm=e8Hm,e2hm=e7Hm,MPO=mpo[2])
            c1 = c1new;c2 = c2new;c3=c3new;c4=c4new;e3=e3new;e4=e4new;e7=e7new;e8=e8new;
            e4Hm = e4Hmnew;e3Hm = e3Hmnew;e8Hm=e8Hmnew;e7Hm=e7Hmnew
        end
        # Step 2, renormalization steps including Hamiltonian terms
        c1new,c4new,e2new,e1new,e2Hmnew,e1Hmnew = NLeftMoverWithH(c1,c2,c3,c4,e2,e1,e4,e3,e6,e5,e8,e7,a,b,chimax,
                                caltype="HENV",calstep="Normal",e1hm=e2Hm,e2hm=e1Hm,MPO=mpo[2])
        c3new,c2new,e6new,e5new,e6Hmnew,e5Hmnew = NRightMoverWithH(c3,c4,c1,c2,e6,e5,e8,e7,e2,e1,e4,e3,a,b,chimax,
                                caltype="HENV",calstep="Normal",e1hm=e6Hm,e2hm=e5Hm,MPO=mpo[2])
        c1=c1new;c2=c2new;c3=c3new;c4=c4new;e1=e1new;e2=e2new;e5=e5new;e6=e6new;
        e1Hm=e1Hmnew;e2Hm=e2Hmnew;e5Hm=e5Hmnew;e6Hm=e6Hmnew
        c2new,c1new,e3new,e4new,e3Hmnew,e4Hmnew = NUpMoverWithH(c2,c3,c4,c1,e3,e4,e6,e5,e7,e8,e2,e1,a,b,chimax,
                                caltype="HENV",calstep="Normal",e1hm=e3Hm,e2hm=e4Hm,MPO=mpo[2])
        c4new,c1new,e7new,e8new,e7Hmnew,e8Hmnew = NDownMoverWithH(c4,c1,c2,c3,e7,e8,e2,e1,e3,e4,e6,e5,a,b,chimax,
                                caltype="HENV",calstep="Normal",e1hm=e7Hm,e2hm=e8Hm,MPO=mpo[2])
        c1 = c1new;c2 = c2new;c3=c3new;c4=c4new;e3=e3new;e4=e4new;e7=e7new;e8=e8new;
        e4Hm = e4Hmnew;e3Hm = e3Hmnew;e8Hm=e8Hmnew;e7Hm=e7Hmnew

        k+=1
        (u1,s1,v1) = svd(c1);(u2,s2,v2) = svd(c2);(u3,s3,v3) = svd(c3);(u4,s4,v4) = svd(c4)
        if length(s1)==length(s1last)&&length(s2)==length(s2last)&&length(s3)==length(s3last)&&length(s4)==length(s4last)&&
            length(s1)==length(s1last2)&&length(s2)==length(s2last2)&&length(s3)==length(s3last2)&&length(s4)==length(s4last2)
            cvgctr1 =norm(abs.(s1/maximum(s1)-s1last))/norm(s1last); cvgctr2 =norm(abs.(s2/maximum(s2)-s2last))/norm(s2last);
            cvgctr3 =norm(abs.(s3/maximum(s3)-s3last))/norm(s3last); cvgctr4 =norm(abs.(s4/maximum(s4)-s4last))/norm(s4last);
            cvgctr12 =norm(abs.(s1/maximum(s1)-s1last2))/norm(s1last2); cvgctr22 =norm(abs.(s2/maximum(s2)-s2last2))/norm(s2last2);
            cvgctr32 =norm(abs.(s3/maximum(s3)-s3last2))/norm(s3last2); cvgctr42 =norm(abs.(s4/maximum(s4)-s4last2))/norm(s4last2);
            #print("This is convergence criteria cvgctr1 $k : $cvgctr1 $cvgctr12 \n")
            #print("This is convergence criteria cvgctr2 $k : $cvgctr2 $cvgctr22\n")
            #print("This is convergence criteria cvgctr3 $k : $cvgctr3 $cvgctr32\n")
            #print("This is convergence criteria cvgctr4 $k : $cvgctr4 $cvgctr42\n")
            #cvgctr1 < envtol && cvgctr2 < envtol && cvgctr3 < envtol && cvgctr4 < envtol ? (numcvg+=1;break) : (numcvg=0;)
        end
        s1last2 = s1last;s2last2 = s2last;s3last2 = s3last;s4last2=s4last
        s1last = s1/maximum(s1);s2last = s2/maximum(s2);s3last=s3/maximum(s3);s4last=s4/maximum(s4);
        #numcvg> 2 ? break : continue
    end
    k == envLoop ? cvgchk = false : cvgchk = true
    print("This is environment step $k   \n")
    return c1,c2,c3,c4,e1,e2,e3,e4,e5,e6,e7,e8,e1Hm,e2Hm,e3Hm,e4Hm,e5Hm,e6Hm,e7Hm,e8Hm,cvgchk
end




"""
    Compute Environment for Unit Cell
"""
function ComputeEnvironmentUC(param::PEPSParam,paramc::PEPSParam,taoind::Int64,chiind::Int64,Dind::Int64,envloop::Int64,envtol::Float64)
    Slast = [[rand(param.chimax[chiind]) for k in 1:4] for i in 1:2, j in 1:2]
    #---- Update Tensors
    l= 0
    for k in 1:envloop
        
        #print("-------ComputeEnvironmentUC Loop $k:   ")
        #----- Left ; Right Movers
        for j in 1:4
            mover(param,paramc,taoind,chiind,Dind,direction=j)
        end
        SMatrix = [[(x=svd(param.CornerEnvironment[i,j][m]).S;x=x/maximum(x)) for m in 1:4] for i in 1:2, j in 1:2]
        relediff = vcat([abs.(norm(SMatrix[i,j][m]-Slast[i,j][m])/norm(Slast[i,j][m])) for m in 1:4, i in 1:2,j in 1:2]...)
        
        sum(relediff .< envtol) > 4.0 ? (l+=1;break) : (l+=1)
        Slast = copy(SMatrix)
    end
    
end







"""
    Initial_Environment
"""

function Initial_Environment(param)

    CornerEnvironment =[[rand(1,1) for k in 1:4] for i in 1:2, j in 1:2]
    CornerEnvironment[2,2] = deepcopy(CornerEnvironment[1,1])
    CornerEnvironment[2,1] = deepcopy(CornerEnvironment[1,2])
    EdgeEnvironment =[[rand(1,Dlink[1],Dlink[1],1) for k in 1:4] for i in 1:2, j in 1:2]
    EdgeEnvironment[2,2] = deepcopy(EdgeEnvironment[1,1])
    EdgeEnvironment[2,1] = deepcopy(EdgeEnvironment[1,2])

    param.CornerEnvironment = deepcopy(CornerEnvironment)
    param.EdgeEnvironment = deepcopy(EdgeEnvironment)

end

#=
"""
    ComputeEnvironmentUCIsometry
"""
function ComputeEnvironmentUCIsometry(param::PEPSParam,envloop::Int64,envtol::Float64)
    Slast = [[rand(param.chimax) for k in 1:4] for i in 1:2, j in 1:2]
    #---- Update Tensors
    l= 0
    for k in 1:envloop
        println("-------ComputeEnvironmentUC Loop $k ----------------------------------")
        #----- Movers
        for j in 1:4
            println("mover $j")
            mover(param,direction=j)
        end
        SMatrix = [[(x=svd(param.CornerEnvironment[i,j][k]).S;x=x/maximum(x)) for k in 1:4] for i in 1:2, j in 1:2]
        relediff = vcat([abs.(sum(SMatrix[i,j][k]-Slast[i,j][k])) for k in 1:4, i in 1:2,j in 1:2]...)
        sum(relediff .< envtol) > 4.0 ? (l+=1;break) : (l+=1)
        Slast = copy(SMatrix)
        append!(EnergyIteration,sum(ComputeEnergy(paramc))/4/2)
    end
    println("Environment Step $l")

end
=#
