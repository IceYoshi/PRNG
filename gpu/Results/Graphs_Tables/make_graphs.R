#par(mfrow=c(1,3)) # Show all graphs simultaneously in a row

# Execution speed comparison between differeng RNGs on CUDA
x = c(163840,1638400,16384000,163840000,1638400000,16384000000,163840000000)
#t0 = c(219.8,209.2,833.8,7794.3,76265.8,760931.8,7586595.3)
#t1 = c(67.3,83.6,417.7,3729.2,35982.4,371058.9,3608595.7)
#t2 = c(334.6,331,548.7,4822.4,46065.6,472828.1,4586871.3)
#t3 = c(218.7,211.3,978.8,9602,90761.7,910492.4,9089844.7)
#t4 = c(46.5,720.7,8163,81626,818941.9,7982884.3,7839147.7)
#t5 = c(81.7,187.2,1182,11187.1,108345,1081088.4,10824432.5)
#t6 = c(105.4,596.1,5643.6,59112.4,565444.6,5308391.7,53272623)
#t7 = c(77.6,150.3,1007.8,9167,90180.4,891344.5,8887951.2)
#t8 = c(81.6,160.9,1192.9,11701.9,111826.2,1122143.1,11182539.5)
#t9 = c(46.2,1324.8,12947.5,122757.6,1278499.5,12386966.5,886654.1)
#t10 = c(58.8,210,1431,12976,125956.8,1180428.6,11836315.7)

t0 = c(219.8,209.2,833.8,7794.3,76265.8,760931.8,7592197.9)
t1 = c(67.3,83.6,417.7,3729.2,35982.4,371058.9,3619383)
t2 = c(334.6,331,548.7,4822.4,46065.6,472828.1,4595440.9)
t3 = c(218.7,211.3,978.8,9602,90761.7,910492.4,9108394.5)
t4 = c(46.5,720.7,8163,81626,818941.9,7982884.3,9468088.1)
t5 = c(81.7,187.2,1182,11187.1,108345,1081088.4,10828346.4)
t6 = c(105.4,596.1,5643.6,59112.4,565444.6,5308391.7,53502059.1)
t7 = c(77.6,150.3,1007.8,9167,90180.4,891344.5,8916521.7)
t8 = c(81.6,160.9,1192.9,11701.9,111826.2,1122143.1,11184420)
t9 = c(46.2,1324.8,12947.5,122757.6,1278499.5,12386966.5,890310.5)
t10 = c(58.8,210,1431,12976,125956.8,1180428.6,11834016.6)

  
plot(x, t0, log="xy", type="o", pch=19, las=0, ylim=c(20, 60000000), xlim=c(163840,163840000000), main="Execution speed comparison between different RNGs", xlab="Number of generated random numbers (log scale)", ylab="Execution time in microseconds (log scale)")
  
lines(x, y=t1, col="red", type="o", pch=6)
lines(x, y=t2, col="blue", type="o", pch=8)
lines(x, y=t3, col="green", type="o", pch=10)
lines(x, y=t4, col="yellow", type="o", pch=12)
lines(x, y=t5, col="orange", type="o", pch=14)
lines(x, y=t6, col="brown", type="o", pch=16)
lines(x, y=t7, col="pink", type="o", pch=18)
lines(x, y=t8, col="magenta", type="o", pch=20)
lines(x, y=t9, col="lightgreen", type="o", pch=22)
lines(x, y=t10, col="lightblue", type="o", pch=24)

  
grid(col = "lightgray", lty = "dotted", lwd = par("lwd"), equilogs = TRUE)
legend(100000, 100000000, legend=c("combinedLCGTaus", "drand48gpu", "kiss07", "lfsr113", "md5", "mtgp", "park_miller", "precompute", "ranecu", "tea", "tt800"), col=c("black", "red", "blue", "green", "yellow", "orange", "brown", "pink", "magenta", "lightgreen", "lightblue"), pch=c(19,6, 8, 10, 12, 14, 16, 18, 20, 22, 24))
  
