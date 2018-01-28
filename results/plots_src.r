#-----------------------------------------------------------------------------------------------------------------
# Execution speed comparison between different RNGs (OpenMPI) with constant m = 0 and n = 100000000000
x = c(1,5,10,15,20, 25, 28, 28, 30, 35, 40, 45, 50, 55, 56)
t0 = c(1647.502,353.919,187.934,125.536,94.402,75.641,68.281,68.617,64.288,55.046,48.566,43.301,39.051,35.734,35.4)
t1 = c(491.299,105.46,56.2,37.605,28.347,22.793,20.696,41.57,38.667,34.747,29.273,26.085,23.721,22.92,22.471)
t2 = c(2420.335,519.241,276.556,200.97,200.315,198.732,201.233,102.202,96.672,81.1,72.419,63.716,58.615,52.125,52.539)

plot(x, t0, log="y", type="o", pch=19, las=0, ylim=c(20, 2500), xlim=c(1,56), 
    main="Execution speed comparison between different RNGs (OpenMPI) \n with constant m = 0 and n = 100000000000",
    xlab="Number of parallel processes", 
    ylab="Execution time in s (log scale)"
)

lines(x, y=t1, col="red", type="o", pch=6)
lines(x, y=t2, col="blue", type="o", pch=8)

grid(col = "lightgray", lty = "dotted", lwd = par("lwd"), equilogs = TRUE)
legend(44, 2500, legend=c("rand()", "mt19937", "MRG32k3a"), col=c("black", "red", "blue"), pch=c(19,6, 8))

abline(v = 28, col="black")

#-----------------------------------------------------------------------------------------------------------------
# Execution speed comparison between different parallelism methods for mt19937 generator and n = 100000000000
x = c(1,5,10,15,20, 25, 28, 28, 30, 35, 40, 45, 50, 55, 56)
t0 = c(559.071,112.034,56.464,44.222,41.804,41.45,41.854,21.616,20.439,16.915,15.009,13.47,12.596,11.114,11.133)
t1 = c(494.762,111.903,65.223,45.677,37.46,32.297,31.207,30.596,29.263,26.465,24.867,23.029,21.215,19.169,19.323)
t2 = c(2250.072,1374.589,1475.45,1482.447,1811.979,2315.967,2624.61,1281.97,1274.911,1265.315,1267.943,1259.394,1251.983,1245.622,1246.712)

plot(x, t0, log="y", type="o", pch=19, las=0, ylim=c(10, 5000), xlim=c(1,56), 
    main="Execution speed comparison between different parallelism methods \n for MT19937 generator and n = 100000000000",
    xlab="Number of parallel processes", 
    ylab="Execution time in s (log scale)"
)

lines(x, y=t1, col="red", type="o", pch=6)
lines(x, y=t2, col="blue", type="o", pch=8)

grid(col = "lightgray", lty = "dotted", lwd = par("lwd"), equilogs = TRUE)
legend(35, 4500, legend=c("Different parameter sets", "Block-splitting", "Leap-frogging"), col=c("black", "red", "blue"), pch=c(19,6, 8))

abline(v = 28, col="black")

#-----------------------------------------------------------------------------------------------------------------
# Execution speed between different sequential PRNGs
x = c(100,1000,10000,100000,1000000,10000000,100000000,1000000000,10000000000, 100000000000)
t0 = c(0.010,0.008,0.008,0.009,0.018,0.101,0.936,9.207,91.863,876.140)
t1 = c(0.010,0.008,0.009,0.009,0.014,0.066,0.585,5.677,56.796,540.635)
t2 = c(0.010,0.008,0.008,0.011,0.035,0.280,2.717,27.118,267.944,2511.625)

plot(x, t0, log="xy", type="o", pch=19, las=0, ylim=c(0.01, 3000), xlim=c(100,100000000000), 
    main="Execution speed between different sequential PRNGs",
    xlab="Number of random numbers (log scale)", 
    ylab="Execution time in s (log scale)"
)

lines(x, y=t1, col="red", type="o", pch=6)
lines(x, y=t2, col="blue", type="o", pch=8)

grid(col = "lightgray", lty = "dotted", lwd = par("lwd"), equilogs = TRUE)
legend(100, 2500, legend=c("rand()", "mt19937", "MRG32k3a"), col=c("black", "red", "blue"), pch=c(19,6, 8))