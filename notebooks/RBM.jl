#---RBM----
#clear
using Sobol, Random, Distributions
using JLD2
using JLD
using FileIO
using CSV
using LaTeXStrings
using Plots

L = 2 * pi/1000;
T = pi;
h = 0.2;
loop_number = Int(floor(T/h));
particle_number = 1500;
X_0 = zeros(3,particle_number);
V_0 = zeros(3,particle_number);
S = zeros(3*particle_number, 3*particle_number);
B = [0,0,1];
Bhat = [[0, -B[3], B[2]] [B[3], 0, -B[1]]   [-B[2], B[1], 0]   ]
#---generate particles with the same weights.
rng_sobol = SobolSeq(3) 
for i = 1 : particle_number
     X_0[:,i] = Sobol.next!(rng_sobol).* L;
 end
d = Normal()

 for i = 1 : particle_number
     V_0[:,i] .= rand!(d,V_0[:,i]) .- 1;
 end
X = X_0;
V = V_0;
exact_X = X_0;
exact_V = V_0;
M = 1;
a = L/M/1;
constant = 1/(a^6)*L^3/(particle_number);
#---exact_solution---
#=
  momentum_0 =  sum(V');  
    
      for j = 1 : particle_number
         for k = 1 : particle_number
             factor = prod(a - min(a*ones(3,1),abs(X(:,j)-X(:,k))));          
             S[3*(j-1)+1:3*(j-1)+3, 3*(k-1)+1:3*(k-1)+3] = factor*Bhat;
         end
      end
      
      for step = 1 : loop_number
      exact = exp(h*constant*S)*reshape(exact_V,3*particle_number,1);
      exact_V = reshape(exact,3,particle_number);
      momentum_exact[:,step] =  sum(exact_V');
      step
      end
=#    
    



#----RBM----

number = particle_number;
SS = zeros(3*number,3*number);
answer = zeros(3*number,1)
first_particle = zeros(3,Int(loop_number))
momentum = zeros(3,Int(loop_number))
norm_v = zeros(1,Int(loop_number))
for step = 1 : Int(loop_number)
   index =  randperm(particle_number);
   for group = 1 : Int(particle_number/number)
       ii = index[number*(group-1)+1 : number*group];
       for j = 1 : number
            for k = 1 : number
                
                SS[3*(j-1)+1:3*(j-1)+3, 3*(k-1)+1:3*(k-1)+3] .= prod(a .- min.(a*ones(3,1),abs.(X[:,ii[j]]-X[:,ii[k]])))*Bhat;
               
            end
       end
       answer .= exp(h*constant*SS)*reshape(V[:,index[number*(group-1)+1 : number*group]],3*number,1);#%[X(:,index(2*group-1));X(:,index(2*group))];
       V[:,index[number*(group-1)+1 : number*group]] .= reshape(answer,3,number);
   end
   first_particle[:,step] = V[:,1];
   norm_v[step] = sum(V.^2);
   momentum[:,step] =  [sum(V[1,:]), sum(V[2,:]), sum(V[3,:]) ] ;
   print(step)    
end
save("momentum.jld2", "momentum", momentum)
save("X_0.jld2", "X_0", X_0)
save("V_0.jld2", "V_0", V_0)
