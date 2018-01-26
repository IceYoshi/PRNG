#-----------------------------------------------------------------------------------------------------------------
# Execution speed comparison between differeng RNGs (OpenMPI) with constant m = 0 and n = 100000000000
x = c(1,5,10,15,20, 25, 28, 28, 30, 35, 40, 45, 50, 55, 56)
t0 = c(1647.502,353.919,187.934,125.536,94.402,75.641,68.281,68.617,64.288,55.046,48.566,43.301,39.051,35.734,35.4)
t1 = c(491.299,105.46,56.2,37.605,28.347,22.793,20.696,41.57,38.667,34.747,29.273,26.085,23.721,22.92,22.471)
t2 = c(2420.335,519.241,276.556,200.97,200.315,198.732,201.233,102.202,96.672,81.1,72.419,63.716,58.615,52.125,52.539)

plot(x, t0, log="y", type="o", pch=19, las=0, ylim=c(20, 2500), xlim=c(1,56), 
    main="Execution speed comparison between differeng RNGs (OpenMPI) \n with constant m = 0 and n = 100000000000",
    xlab="Number of parallel processes", 
    ylab="Execution time in s (log scale)"
)

lines(x, y=t1, col="red", type="o", pch=6)
lines(x, y=t2, col="blue", type="o", pch=8)

grid(col = "lightgray", lty = "dotted", lwd = par("lwd"), equilogs = TRUE)
legend(44, 2500, legend=c("rand()", "mt19937", "MRG32k3a"), col=c("black", "red", "blue"), pch=c(19,6, 8))

abline(v = 28, col="black")

#-----------------------------------------------------------------------------------------------------------------
# Execution speed comparison between differeng parallelism methods for mt19937 generator and n = 100000000000
x = c(1,5,10,15,20)
t0 = c(39214,7316,3329,2542,1915)
t1 = c(41023,8568,3669,2346,2041)
t2 = c(39467,8661,4975,3666,4010)

plot(x, t0, log="y", type="o", pch=19, las=0, ylim=c(1500, 45000), xlim=c(1,20), 
    main="Execution speed comparison between differeng parallelism methods \n for mt19937 generator and n = 100000000",
    xlab="Number of parallel processes", 
    ylab="Execution time in ms (log scale)"
)

lines(x, y=t1, col="red", type="o", pch=6)
lines(x, y=t2, col="blue", type="o", pch=8)

grid(col = "lightgray", lty = "dotted", lwd = par("lwd"), equilogs = TRUE)
legend(12.5, 45000, legend=c("Different parameter sets", "Block-splitting", "Leap-frogging"), col=c("black", "red", "blue"), pch=c(19,6, 8))