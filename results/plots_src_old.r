par(mfrow=c(1,3)) # Show all graphs simultaneously in a row

# Execution speed comparison between differeng RNGs (sequential)
x = c(1000,10000,100000,1000000,10000000,100000000,1000000000,10000000000)
t0 = c(21,23,42,158,978,9175,91271,912230)
t1 = c(21,22,41,156,953,8924,88389,883718)
t2 = c(20,23,45,181,1145,10855,108012,1079419)

plot(x, t0, log="xy", type="o", pch=19, las=0, ylim=c(20, 1200000), xlim=c(1000,10000000000), 
    main="Execution speed comparison between differeng RNGs (sequential)", 
    xlab="Number of generated random 32-bit numbers (log scale)", 
    ylab="Execution time in ms (log scale)"
)

lines(x, y=t1, col="red", type="o", pch=6)
lines(x, y=t2, col="blue", type="o", pch=8)

grid(col = "lightgray", lty = "dotted", lwd = par("lwd"), equilogs = TRUE)
legend(1000, 1200000, legend=c("rand()", "mt19937", "MRG32k3a"), col=c("black", "red", "blue"), pch=c(19,6, 8))

#-----------------------------------------------------------------------------------------------------------------
# Execution speed comparison between differeng RNGs (OpenMPI) with constant m = 0 and n = 100000000
x = c(1,5,10,15,20)
t0 = c(49229,8409,4202,2652,2028)
t1 = c(39214,7316,3329,2542,1915)
t2 = c(39066,8318,3932,2448,1873)

plot(x, t0, log="y", type="o", pch=19, las=0, ylim=c(1500, 55000), xlim=c(1,20), 
    main="Execution speed comparison between differeng RNGs (OpenMPI) \n with constant m = 0 and n = 100000000",
    xlab="Number of parallel processes", 
    ylab="Execution time in ms (log scale)"
)

lines(x, y=t1, col="red", type="o", pch=6)
lines(x, y=t2, col="blue", type="o", pch=8)

grid(col = "lightgray", lty = "dotted", lwd = par("lwd"), equilogs = TRUE)
legend(15.5, 55000, legend=c("rand()", "mt19937", "MRG32k3a"), col=c("black", "red", "blue"), pch=c(19,6, 8))

#-----------------------------------------------------------------------------------------------------------------
# Execution speed comparison between differeng parallelism methods for mt19937 generator and n = 100000000
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