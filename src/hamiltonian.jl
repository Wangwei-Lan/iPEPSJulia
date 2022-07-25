function ConstructHamiltonian(Sx::Array{Float64},Sy::Array{Complex{Float64}},Sz::Array{Float64},hamitype::String;Magnetism=1.0)
    if hamitype == "Heisenberg"
        return 1/2*(kron(Sx,Sx)+kron(Sy,Sy)+kron(Sz,Sz))
    elseif hamitype == "Ising"
        return -kron(Sz,Sz)-Magnetism/4*(kron(Matrix(1.0I,2,2),Sx)+kron(Sx,Matrix(1.0I,2,2)))
        #return -kron(Sz,Sz)-Magnetism/4*(kron(Matrix(1.0I,2,2),Sx)+kron(Sx,Matrix(1.0I,2,2)))
    elseif hamitype == "AKLT"
        SdotS = real(kron(Sz,Sz)+kron(Sx,Sx)+kron(Sy,Sy))
        Hamiltonian = 1/14*(SdotS+7/10*SdotS^2+7/45*SdotS^3+1/90*SdotS^4)
        #Hamiltonian = reshape(Hamiltonian,5,5,5,5)
        return Hamiltonian
    end
end
