MODEL: GPT5.2 medium reasoning effort

R = 10
For L = 2^i for i in range(2,12).
Repeat R times:
Generate two 2^L bit primes. 
Write them in decimal.
Ask the model to multiply them out **by hand** as a demonstration of multiplication.

Make a folder called generations where you store the model outputs. 
Store them as 
bits=xxx_idx=k
where idx is which number generation it is, you'll have to read the folder to
see if there are already generations and then go based on that. 

Start with a test run where you use R=1. 
Write the outputs as you go instead of doing it all at the end.