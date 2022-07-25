"""
    LeftUpdate consistent with 2x2 unit cell
"""
function LeftUpdate(a,b,g,c1,c2,c3,c4,e1,e2,e3,e5,e6,e8,updateloop,updatetol,chimax)
    Dlink = size(a,2)
    dphy = size(a,1)

    #--- Update
    #@tensor TMP[-1,-2,-3,-4] := aU[3,1,-1]*bD[2,1,-3]*g[2,3,-4,-2]
    #F = LinearAlgebra.svd(reshape(TMP,dphy*aUbond,dphy*bDbond))
    #U = copy(F.U);S = copy(F.S);V = copy(F.V);Vt = copy(F.Vt)
    @tensor TMP[-1,-2,-3,-4,-5,-6,-7,-8] := a[3,-3,-4,1,-2]*b[2,1,-8,-7,-6]*g[-5,-1,2,3]
    F = LinearAlgebra.svd(reshape(TMP,dphy*Dlink^3,dphy*Dlink^3))
    atilt = reshape(F.U*Matrix(Diagonal(sqrt.(F.S))),dphy,Dlink,Dlink,Dlink,Dlink^3*dphy)
    btilt = reshape(Matrix(Diagonal(sqrt.(F.S))),Dlink^3*dphy,dphy,Dlink,Dlink,Dlink)
    atilt = permutedims(atilt,[1,3,4,5,2])
    btilt = permutedims(btilt,[2,1,5,4,3])
    atilt = atilt[:,:,:,1:Dlink,:]
    btilt = btilt[:,1:Dlink,:,:,:]
    #--- Initial of aUtilt and bDtilt (Another easier way is to use bD and aU directly)
    @tensor begin
        Env[-1,-2,-3,-4,-5,-6,-7,-8,-9,-10,-11,-12] := c1[1,2]*c2[3,4]*c3[5,6]*c4[7,8]*e1[8,-9,-10,9]*e2[9,-3,-4,1]*e3[2,-1,-2,3]*e5[4,-5,-6,10]*
                            e6[10,-11,-12,5]*e8[6,-7,-8,7]
    end
    Env = Env/maximum(Env)
    costfunc = [[1.0]];numcvg = 0;k=0
    for i in 1:updateloop
        #println(" Update Loop $i ")
        #------------------------- Update
        #println("------------Update Step $i--------------------")
        @tensor begin
            la[-1,-2,-3,-4,-5,-6,-7,-8,-9,-10] := Env[-2,-7,-3,-8,-5,-10,5,2,4,1,6,3]*btilt[7,-9,1,2,3]*conj(btilt)[7,-4,4,5,6]*Matrix(1.0I,dphy,dphy)[-1,-6]
            ra[-1,-2,-3,-4,-5] := Env[-2,9,-3,10,-5,11,5,2,4,1,6,3]*a[12,9,10,13,11]*b[8,13,1,2,3]*g[7,-1,8,12]*conj(btilt)[7,-4,4,5,6]
        end
        la = reshape(la,Dlink^4*dphy,Dlink^4*dphy)
        ra = reshape(ra,Dlink^4*dphy)
        atilt = reshape(\(la,ra),dphy,Dlink,Dlink,Dlink,Dlink)
        @tensor begin
            lb[-1,-2,-3,-4,-5,-6,-7,-8,-9,-10] := Env[5,2,6,3,4,1,-4,-9,-3,-8,-5,-10]*atilt[7,2,3,-7,1]*conj(atilt)[7,5,6,-2,4]*Matrix(1.0I,dphy,dphy)[-1,-6]
            rb[-1,-2,-3,-4,-5] := Env[5,2,6,3,4,1,-4,10,-3,9,-5,11]*a[7,2,3,12,1]*b[13,12,9,10,11]*g[-1,8,13,7]*conj(atilt)[8,5,6,-2,4]
        end
        lb = reshape(lb,Dlink^4*dphy,Dlink^4*dphy)
        rb = reshape(rb,Dlink^4*dphy)
        btilt = reshape(\(lb,rb),dphy,Dlink,Dlink,Dlink,Dlink)
        cost,daboo,dabnn,dabno,dabon = ConvergeCheck(Env,b,a,btilt,atilt,g)
        abs.((cost .- costfunc[end])[1]) < updatetol ?  (numcvg+=1;k+=1) : (numcvg=0;k+=1)
        println("cost difference ",abs.((cost - costfunc[end])[1]))
        println("cost : ",cost )
        println("daboo : ",daboo)
        #println("dabnn : ",dabnn)
        #println("dabon : ",dabon)
        #println("dabno : ",dabno)
        #aUtilt = aUtilt/maximum(aUtilt)
        #bDtilt = bDtilt/maximum(bDtilt)
        atilt = atilt/maximum(atilt)
        btilt = btilt/maximum(btilt)
        push!(costfunc,[cost])
        numcvg > 2 ? break : continue
    end
    k == updateloop ? cvgchk = false : cvgchk = true
    #--- Next Few step seems not necessary! But I still include it here!
    #=
    aUtmp = reshape(permutedims(aUtilt,[2,1,3]),Dlink,dphy*aUbond)
    bDtmp = reshape(permutedims(bDtilt,[3,1,2]),bDbond*dphy,Dlink)
    (Q1,R) = qr(bDtmp);(L,Q2) = lq(aUtmp)
    F = svd(R*L)
    bDtilt = Q1*F.U*Matrix(Diagonal(sqrt.(F.S)))
    aUtilt = Matrix(Diagonal(sqrt.(F.S)))*F.Vt*Q2
    bDtilt = permutedims(reshape(bDtilt,bDbond,dphy,Dlink),[2,3,1])
    aUtilt = permutedims(reshape(aUtilt,Dlink,dphy,aUbond),[2,1,3])
    =#
    #@tensor a[-1,-2,-3,-4,-5]:= Xtilt[1,-5,-2,-3]*aUtilt[-1,-4,1]
    #@tensor b[-1,-2,-3,-4,-5]:= Ytilt[-3,-4,-5,1]*bDtilt[-1,-2,1]

    #@tensor a[-1,-2,-3,-4,-5]:= X[1,-5,-2,-3]*aUtilt[-1,-4,1]
    #@tensor b[-1,-2,-3,-4,-5]:= Y[-3,-4,-5,1]*bDtilt[-1,-2,1]

    #atilt = atilt/maximum(atilt)
    #btilt = btilt/maximum(btilt)
    return atilt,btilt,cvgchk

end

function RightUpdate(a,b,g,c3,c4,c1,c2,e5,e6,e7,e1,e2,e4,updateloop,updatetol,chimax)
    a = permutedims(a,[1,4,5,2,3])
    b = permutedims(b,[1,4,5,2,3])
    a,b,cvgchk=LeftUpdate(a,b,g,c3,c4,c1,c2,e5,e6,e7,e1,e2,e4,updateloop,updatetol,chimax)
    a = permutedims(a,[1,4,5,2,3])
    b = permutedims(b,[1,4,5,2,3])
    return a,b,cvgchk
end

function DownUpdate(b,a,g,c4,c1,c2,c3,e7,e8,e1,e3,e4,e6,updateloop,updatetol,chimax)
    b = permutedims(b,[1,3,4,5,2])
    a = permutedims(a,[1,3,4,5,2])
    b,a,cvgchk = LeftUpdate(b,a,g,c4,c1,c2,c3,e7,e8,e1,e3,e4,e6,updateloop,updatetol,chimax)
    b = permutedims(b,[1,5,2,3,4])
    a = permutedims(a,[1,5,2,3,4])
    return b,a,cvgchk
end

function UpUpdate(b,a,g,c2,c3,c4,c1,e3,e4,e5,e7,e8,e2,updateloop,updatetol,chimax)
    b = permutedims(b,[1,5,2,3,4])
    a = permutedims(a,[1,5,2,3,4])
    b,a,cvgchk = LeftUpdate(b,a,g,c2,c3,c4,c1,e3,e4,e5,e7,e8,e2,updateloop,updatetol,chimax)
    b = permutedims(b,[1,3,4,5,2])
    a = permutedims(a,[1,3,4,5,2])
    return b,a,cvgchk
end

"""
    ConvergeCheck calculate the difference of ||Ψ_old - Ψ_new ||
"""
function ConvergeCheck(Env,b,a,btilt,atilt,g)
    @tensor begin
        #daboo[] := Env[8,12,1,3]*bD[4,2,1]*conj(bD)[9,10,8]*conj(aU)[11,10,12]*aU[5,2,3]*g[6,7,4,5]*conj(g)[6,7,9,11]
        #dabnn[] := Env[2,8,1,5]*bDtilt[3,4,1]*conj(bDtilt)[3,6,2]*aUtilt[7,4,5]*conj(aUtilt)[7,6,8]
        #dabno[] := Env[6,10,1,3]*bD[4,2,1]*conj(bDtilt)[7,8,6]*aU[5,2,3]*conj(aUtilt)[9,8,10]*g[7,9,4,5]
        #dabon[] := Env[6,10,1,3]*bDtilt[4,2,1]*conj(bD)[7,8,6]*aUtilt[5,2,3]*conj(aU)[9,8,10]*conj(g)[4,5,7,9]
        daboo[] := Env[16,2,17,3,20,1,13,7,14,6,12,4]*b[8,5,6,7,4]*b[15,18,14,13,12]*a[9,2,3,5,1]*a[19,16,17,18,20]*g[10,11,8,9]*g[10,11,15,19]
        dabnn[] := Env[13,9,14,10,12,8,5,2,6,3,4,1]*btilt[7,11,3,2,1]*btilt[7,15,6,5,4]*atilt[16,9,10,11,8]*atilt[16,13,14,15,12]
        dabno[] := Env[15,10,16,11,14,9,5,2,6,3,4,1]*b[8,12,3,2,1]*btilt[7,17,6,5,4]*a[13,10,11,12,9]*atilt[18,15,16,17,14]*g[7,18,8,13]
        dabon[] := Env[15,10,16,11,14,9,5,2,6,3,4,1]*btilt[8,12,3,2,1]*b[7,17,6,5,4]*atilt[13,10,11,12,9]*a[18,15,16,17,14]*g[8,13,7,18]
    end
    return (daboo[1]+dabnn[1]-dabno[1]-dabon[1]),daboo,dabnn,dabno,dabon
end
