using TensorOperations
function rupdate(gammaa,gammab,lambda1,lambda2,lambda3,lambda4,U,g)
    Dlink = size(gammaa,1);dphy = size(gammaa,5)

    # r-link
    #
    @tensor Left[:] := gammaa[1,2,3,-4,-5]*lambda1[-1,1]*lambda2[-2,2]*lambda3[-3,3]
    @tensor Right[:] := gammab[1,-2,3,2,-5]*lambda1[-3,3]*lambda2[2,-4]*lambda3[-1,1]
    Left = reshape(Left,Dlink^3,Dlink*dphy)
    Right = reshape(permutedims(Right,[1,4,3,2,5]),Dlink^3,Dlink*dphy)
    Fleft = svd(Left)
    Fright = svd(Right)    

    @tensor TEMP[-1,-2,-3,-4] := reshape(Fleft.V*diagm(Fleft.S),Dlink,dphy,min(Dlink*dphy,Dlink^3))[1,3,-2]*reshape(Fright.V*diagm(Fright.S),Dlink,dphy,min(Dlink*dphy,Dlink^3))[2,4,-4]*
                                g[-1,-3,3,4]*lambda4[1,2]
    TEMP = reshape(TEMP,dphy*min(Dlink^3,dphy*Dlink),dphy*min(Dlink^3,dphy*Dlink))
    F = svd(TEMP)
    gauge = diagm(sign.(diag(U'*F.U)));

    gammaa = reshape(Fleft.U,Dlink,Dlink,Dlink,min(Dlink*dphy,Dlink^3))
    gammab = reshape(Fright.U,Dlink,Dlink,Dlink,min(Dlink*dphy,Dlink^3))
    
    @tensor gammaa[:] := gammaa[-1,-2,-3,1]*reshape(F.U*gauge,dphy,min(dphy*Dlink,Dlink^3),min(Dlink^3,dphy*Dlink)*dphy)[-5,1,-4]
    @tensor gammab[:] := gammab[-1,-2,-3,1]*reshape(F.V*gauge,dphy,min(dphy*Dlink,Dlink^3),min(Dlink^3,dphy*Dlink)*dphy)[-5,1,-4]
    #@tensor gammaa[:] := gammaa[-1,-2,-3,1]*reshape(F.U,dphy,min(dphy*Dlink,Dlink^3),min(Dlink^3,dphy*Dlink)*dphy)[-5,1,-4]   
    #@tensor gammab[:] := gammab[-1,-2,-3,1]*reshape(F.V,dphy,min(dphy*Dlink,Dlink^3),min(Dlink^3,dphy*Dlink)*dphy)[-5,1,-4]
     
    gammab = permutedims(gammab,[1,4,3,2,5])
    gammaa = gammaa[:,:,:,1:Dlink,:]
    gammab = gammab[:,1:Dlink,:,:,:]

    @tensor gammaa[:] := gammaa[1,2,3,-4,-5]*inv(lambda1)[-1,1]*inv(lambda2)[-2,2]*inv(lambda3)[-3,3]
    @tensor gammab[:] := gammab[1,-2,3,2,-5]*inv(lambda1)[-3,3]*inv(lambda2)[2,-4]*inv(lambda3)[-1,1]

    gammaa = gammaa/maximum(gammaa)
    gammab = gammab/maximum(gammab)
    lambda4=Matrix(Diagonal(F.S[1:Dlink]))/norm(F.S[1:Dlink])
    #


    #=
    @tensor begin
        TMP[-1,-2,-3,-4,-5,-6,-7,-8] := gammaa[1,2,3,4,9]*gammab[5,8,7,6,10]*g[-4,-8,9,10]*lambda1[-1,1]*
                                        lambda2[-2,2]*lambda3[-3,3]*lambda4[4,8]*lambda2[-6,6]*lambda1[-7,7]*lambda3[-5,5]
    end
    TMP = reshape(TMP,Dlink^3*dphy,Dlink^3*dphy)
    F = LinearAlgebra.svd(TMP/norm(TMP))
    @tensor gammaa[-1,-2,-3,-4,-5] := reshape(F.U,Dlink,Dlink,Dlink,dphy,Dlink^3*dphy)[1,2,3,-5,-4]*inv(lambda1)[-1,1]*inv(lambda2)[-2,2]*inv(lambda3)[-3,3]
    @tensor gammab[-1,-2,-3,-4,-5] := reshape(copy(F.V),Dlink,Dlink,Dlink,dphy,Dlink^3*dphy)[1,2,3,-5,-2]*inv(lambda1)[-3,3]*inv(lambda2)[2,-4]*inv(lambda3)[-1,1]
    gammaa = gammaa[:,:,:,1:Dlink,:]
    gammab = gammab[:,1:Dlink,:,:,:]
    gammaa = gammaa/maximum(gammaa)
    gammab = gammab/maximum(gammab)
    lambda4=Matrix(Diagonal(F.S[1:Dlink]))/norm(F.S[1:Dlink])
    =#

    return gammaa,gammab,lambda4,F.S,F.U*gauge
end


function lupdate(gammaa,gammab,lambda1,lambda2,lambda3,lambda4,U,g)
    gammaa = permutedims(gammaa,[3,4,1,2,5])
    gammab = permutedims(gammab,[3,4,1,2,5])
    gammaa,gammab,lambda2,S2,U2= rupdate(gammaa,gammab,lambda3,lambda4,lambda1,lambda2,U,g)
    gammaa = permutedims(gammaa,[3,4,1,2,5])
    gammab = permutedims(gammab,[3,4,1,2,5])
    return gammaa,gammab,lambda2,S2,U2
end

function dupdate(gammaa,gammab,lambda1,lambda2,lambda3,lambda4,U,g)
    gammaa = permutedims(gammaa,[2,3,4,1,5])
    gammab = permutedims(gammab,[2,3,4,1,5])
    gammab,gammaa,lambda3,S3,U3= rupdate(gammab,gammaa,lambda4,lambda1,lambda2,lambda3,U,g)
    gammaa = permutedims(gammaa,[4,1,2,3,5])
    gammab = permutedims(gammab,[4,1,2,3,5])
    return gammaa,gammab,lambda3,S3,U3
end

function uupdate(gammaa,gammab,lambda1,lambda2,lambda3,lambda4,U,g)
    gammaa = permutedims(gammaa,[4,1,2,3,5])
    gammab = permutedims(gammab,[4,1,2,3,5])
    gammab,gammaa,lambda1,S1,U1 = rupdate(gammab,gammaa,lambda2,lambda3,lambda4,lambda1,U,g)
    gammaa = permutedims(gammaa,[2,3,4,1,5])
    gammab = permutedims(gammab,[2,3,4,1,5])
    return gammaa,gammab,lambda1,S1,U1
end

"""
Simple Update based on arxiv0912.0646v2
"""
function SimpleUpdate(a,b,g,updateloop,tol)
    Dlink = size(a,1);dphy = size(a,5)
    #lambda1 = Matrix(Diagonal(rand(Dlink)));lambda2=Matrix(Diagonal(rand(Dlink)));lambda3=Matrix(Diagonal(rand(Dlink)));lambda4=Matrix(Diagonal(rand(Dlink)));
    lambda1 = Matrix(1.0I,Dlink,Dlink);lambda2 = Matrix(1.0I,Dlink,Dlink);lambda3 = Matrix(1.0I,Dlink,Dlink);lambda4 = Matrix(1.0I,Dlink,Dlink)
    @tensor GammaA[-1,-2,-3,-4,-5] :=inv(sqrt.(lambda1))[-1,1]*a[1,2,3,4,-5]*inv(sqrt.(lambda2))[-2,2]*inv(sqrt.(lambda3))[-3,3]*inv(sqrt.(lambda4))[-4,4]
    @tensor GammaB[-1,-2,-3,-4,-5] :=inv(sqrt.(lambda3))[-1,1]*b[1,2,3,4,-5]*inv(sqrt.(lambda4))[-2,2]*inv(sqrt.(lambda1))[-3,3]*inv(sqrt.(lambda2))[-4,4]
    s1tmp = rand(min(Dlink*dphy^2,Dlink^3*dphy)); s2tmp=rand(min(Dlink*dphy^2,Dlink^3*dphy));s3tmp=rand(min(Dlink*dphy^2,Dlink^3*dphy));s4tmp = rand(min(Dlink*dphy^2,Dlink^3*dphy))
    #s1tmp = rand(Dlink^3*dphy); s2tmp=rand(Dlink^3*dphy);s3tmp=rand(Dlink^3*dphy);s4tmp = rand(Dlink^3*dphy)
    numcvg = 0;k=0;
    GammaA_old = deepcopy(GammaA);GammaB_old = deepcopy(GammaB)
    lambda1_old = deepcopy(lambda1)    
    lambda2_old = deepcopy(lambda2)
    lambda3_old = deepcopy(lambda3)
    lambda4_old = deepcopy(lambda4)
    U1 = Matrix(1.0I,dphy^2*Dlink,dphy^2*Dlink)
    U2 = Matrix(1.0I,dphy^2*Dlink,dphy^2*Dlink)
    U3 = Matrix(1.0I,dphy^2*Dlink,dphy^2*Dlink)
    U4 = Matrix(1.0I,dphy^2*Dlink,dphy^2*Dlink)
    for i in 1:updateloop
        #println("Simple Update Loop $i")
        GammaA,GammaB,lambda4,S4,U4= rupdate(GammaA,GammaB,lambda1,lambda2,lambda3,lambda4,U4,g)
        GammaA,GammaB,lambda2,S2,U2= lupdate(GammaA,GammaB,lambda1,lambda2,lambda3,lambda4,U2,g)
        GammaA,GammaB,lambda1,S1,U1= uupdate(GammaA,GammaB,lambda1,lambda2,lambda3,lambda4,U1,g)
        GammaA,GammaB,lambda3,S3,U3= dupdate(GammaA,GammaB,lambda1,lambda2,lambda3,lambda4,U3,g)
        #
        GammaA,GammaB,lambda3,S3,U3= dupdate(GammaA,GammaB,lambda1,lambda2,lambda3,lambda4,U3,g)
        GammaA,GammaB,lambda1,S1,U1= uupdate(GammaA,GammaB,lambda1,lambda2,lambda3,lambda4,U1,g)
        GammaA,GammaB,lambda2,S2,U2= lupdate(GammaA,GammaB,lambda1,lambda2,lambda3,lambda4,U2,g)
        GammaA,GammaB,lambda4,S4,U4= rupdate(GammaA,GammaB,lambda1,lambda2,lambda3,lambda4,U4,g)

        #append!(Amatrix,[GammaA])
        #append!(Bmatrix,[GammaB])
        sum(abs.(S1-s1tmp)/norm(S1)) < tol && sum(abs.(S2-s2tmp)/norm(S2)) < tol && sum(abs.(S3-s3tmp)/norm(S3)) < tol && sum(abs.(S4-s4tmp)/norm(S4)) < tol ? numcvg+=1 : numcvg=0
        #println(sum(abs.(S1-s1tmp)/norm(S1)))
        #norm(abs.(GammaA_old) - abs.(GammaA))/norm(GammaA) < tol && norm(abs.(GammaB_old)-abs.(GammaB))/norm(GammaB) < tol ? numcvg += 1 : numcvg = 0
        #norm(abs.(lambda1_old) - abs.(lambda1))/norm(lambda1) < tol && norm(abs.(lambda2_old)-abs.(lambda2))/norm(lambda2) < tol ? numcvg += 1 : numcvg = 0
        #=
        println(norm(abs.(GammaA_old) - abs.(GammaA))/norm(GammaA))
        println(norm(abs.(GammaB_old) - abs.(GammaB))/norm(GammaB))
        println(norm(lambda1_old -lambda1)/norm(lambda1))
        println(norm(lambda2_old -lambda2)/norm(lambda2))
        println(norm(lambda3_old -lambda3)/norm(lambda3))
        println(norm(lambda4_old -lambda4)/norm(lambda4))
        =#
        GammaA_old = deepcopy(GammaA)
        GammaB_old = deepcopy(GammaB)
        lambda1_old = deepcopy(lambda1)    
        lambda2_old = deepcopy(lambda2)
        lambda3_old = deepcopy(lambda3)
        lambda4_old = deepcopy(lambda4)

        #=
        println(norm(S1-s1tmp)/norm(S1))
        println(norm(S2-s2tmp)/norm(S2))
        println(norm(S3-s3tmp)/norm(S3))
        println(norm(S4-s4tmp)/norm(S4))
        #
        println( sum(abs.(S1-s1tmp)/norm(S1)) )
        println( sum(abs.(S2-s2tmp)/norm(S2)) )
        println( sum(abs.(S3-s3tmp)/norm(S3)) )
        println( sum(abs.(S4-s4tmp)/norm(S4)) )
        =#
        s1tmp = S1;s2tmp =S2;s3tmp=S3;s4tmp=S4;k+=1
        numcvg > 5 && i >250 ? break : continue
        #numcvg > 5 ? break : continue
    end
    print("Simple Update convergence step $k \n")
    #----- rewrite to get a and b tensor
    @tensor begin
        a[-1,-2,-3,-4,-5] := sqrt.(lambda1)[-1,1]*sqrt.(lambda2)[-2,2]*sqrt.(lambda3)[-3,3]*
                                sqrt.(lambda4)[-4,4]*GammaA[1,2,3,4,-5]
        b[-1,-2,-3,-4,-5] := sqrt.(lambda1)[-3,3]*sqrt.(lambda2)[-4,4]*sqrt.(lambda3)[-1,1]*
                                sqrt.(lambda4)[-2,2]*GammaB[1,2,3,4,-5]
    end
    a = a/maximum(a)
    b = b/maximum(b)
    return a,b
end
