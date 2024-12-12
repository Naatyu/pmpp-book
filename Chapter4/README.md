#### ⚠️ DISCLAIMER ⚠️ - These are my answers but were never checked. They might be all false (or right). I also encourage you to try the exercises first and then look at the answers.

**Q1.A**
- 1 warp is 32 threads and there is 128 threads so 4 warps per blocks.

**Q1.B**
- The grid consist of $\frac{1024+128-1}{128}=8.99$ blocks which is 8 blocks. Since there is 4 warps per block we have 32 warps 

**Q1.C**
 - It can be seen that threads between 40 and 104 will be inactive. Here, only the warp $[64,96]$ is inactive. Since the condition depends solely on the threads, this means it will be replicated in each block, therefore 1 warp disabled in each block. This results in 8 disabled warps, leaving 24 active warps.
- 2 warps are divergent in a block $[32,64]$ and $[96,128]$ so in total $8\times2=16$ warps are divergent.
- The efficiency is the ratio of active threads compare to total threads. For warp 0 of block 0 we have an efficiency of 1 so 100%.
- In warp 1, threads 32 to 39 are actives. Which is $\frac{8}{32}=0.25$ so 25% efficiency.
- In warp 3, threads 104 to 127 are actives, $\frac{24}{32}=0.75$ so 75% efficiency.

**Q1.D**
- The if statement is active for pair threads so only half of the threads are active. So all warps are active
- Since only pair threads are active it means only half of the threads in a warp are active so all warps are divergent.
- Since half of threads are active, the efficiency is 50%.
1.E
- The iteration will run between 3 and 5 iterations So there is 3 iterations without divergence.
- There will be 2 iterations with divergence. 

**Q2** 
- since there is 512 threads per block and we need 2000 threads the total will be 2048 threads.

**Q3** 
- Only one warp will have divergence, the one between $[1984,2016]$

**Q4**
- First we compute the difference between the longest threads and all threads which is 4.1ms is total. Since all threads spent 3ms we have total execution of 24ms. So the percentage is 17.08%.

**Q5** 
- His answer assume that each warp is 32 threads so by not calling `__syncthreads`, it will still be waiting since there are in the same warp. But if in future implementations a warp is more than 32 threads, then his code will not work.

**Q6** 
- Answer is 512 since 3 blocks of 512 threads is 1536 which is the maximum number of threads in the SM.

**Q7** 
- a is possible since $8 < 64$ and $8\times128=1024 < 2048$, the occupancy is 50%. b is possible, occupancy is 50%. c is possible with 50% occupancy. d is possible with 100% occupancy. e is possible with 100% occupancy. 

**Q8**
- Answer *a*, there is 128 threads  per block so we can have 16 blocks to get 2048 threads. The number of register is below 64K. This can achieve full occupancy. b, this gives 64 block but we can have only 32 blocks. The number of register is below 64K, we achieve 50% occupancy and the limiting factor is the number of blocks. c, this give 8 blocks and 69K registers which is too much. The occupancy is $64000/(256\times34)=7.34$ so we can launch 7 blocks which is 1792 threads, 87.5% occupancy.
9 - This is not possible since he claims that a block can have 1024 threads but the device only accept 512 threads per block. 