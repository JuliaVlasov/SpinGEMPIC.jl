using JLD2
using JLD
using FileIO
using CSV
using LaTeXStrings
using Plots
using FFTW

frame = CSV.read("frame.csv");
tmp1 = frame[!, :KineticEnergy];
ftmp1 = fft(tmp1) ./ 50000;
ftmp1[1] = 0.0;
plot(
    abs.(ftmp1[1:1000]),
    xlabel = "freq",
    ylabel = "spectrum kinetic energy",
    legend = false,
)
savefig("Kinetic_spectrum_dirac.png")

tmp2 = frame[!, :PotentialEnergyE2];
ftmp2 = fft(tmp2) ./ 50000;
ftmp2[1] = 0.0;
plot(
    abs.(ftmp2[1:1000]) ./ 50000,
    xlabel = "freq",
    ylabel = "spectrum Ex energy",
    legend = false,
)
savefig("Ex_spectrum_dirac.png")

tmp3 = frame[!, :Kineticspin];
ftmp3 = fft(tmp3) ./ 50000;
ftmp3[1] = 0.0;
plot(abs.(ftmp3[1:1000]), xlabel = "freq", ylabel = "spectrum Spin energy", legend = false)
savefig("Spin_spectrum_dirac.png")

tmp1 = frame[!, :Momentum1];
ftmp1 = fft(tmp1) ./ 50000;
ftmp1[1] = 0.0;
plot(abs.(ftmp1[1:1000]), xlabel = "freq", ylabel = "Ex energy", legend = false)
savefig("Ex_spectrum_dirac.png")

tmp1 = frame[!, :PotentialEnergyE2];
ftmp1 = fft(tmp1) ./ 50000;
ftmp1[1] = 0.0;
plot(abs.(ftmp1[1:1000]), xlabel = "freq", ylabel = "Ey energy", legend = false)
savefig("Ey_spectrum_dirac.png")

tmp1 = frame[!, :PotentialEnergyE3];
ftmp1 = fft(tmp1) ./ 50000;
ftmp1[1] = 0.0;
plot(abs.(ftmp1[1:1000]), xlabel = "freq", ylabel = "Ez energy", legend = false)
savefig("Ez_spectrum_dirac.png")

tmp1 = frame[!, :PotentialEnergyB2];
ftmp1 = fft(tmp1) ./ 50000;
ftmp1[1] = 0.0;
plot(abs.(ftmp1[1:1000]), xlabel = "freq", ylabel = "By energy", legend = false)
savefig("By_spectrum_dirac.png")

tmp1 = frame[!, :PotentialEnergyB3];
ftmp1 = fft(tmp1) ./ 50000;
ftmp1[1] = 0.0;
plot(abs.(ftmp1[1:1000]), xlabel = "freq", ylabel = "Bz energy", legend = false)
savefig("Bz_spectrum_dirac.png")
