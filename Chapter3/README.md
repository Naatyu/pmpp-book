#### ⚠️ DISCLAIMER ⚠️ - These are my answers but were never checked. They might be all false (or right). I also encourage you to try the exercises first and then look at the answers.

**Q1.A**
- see `matmul.c`

**Q1.B**
- see `matmul.c`

**Q1.C**
- In the row version, we access one row and all columns of the other matrix but in the col wise kernel we access one column and all rows of the second matrix. Both kernels advantages depends on the memory layout. If we have a row major memory layout, the col wise kernel will be better (memory access will be better since it access contiguous elements for a row). If we have a column major layout then the row wise kernel will be better (same argument as before but for columns). 

**Q2**
- see `matmul.c`

**Q3.A**
- The number of threads per block is $16\times32=512$ 

**Q3.B**
- The total number of threads is $512\times20\times6=61440$

**Q3.C**
- The total number of blocks is $20\times6=120$

**Q3.D**
- The number of threads that executes the code in line 05 is $150*300=45000$

**Q4.A**
- In a row major order it will be $20\times400+10=8010$

**Q4.B**
- In a column major order it wil be $10\times500+20=5020$

**Q5**
- In a row major order it will be $10\times500\times300+20\times300+5=1506005$