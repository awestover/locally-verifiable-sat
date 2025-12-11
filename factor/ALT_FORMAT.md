To compute the product 101010101 * 1010101, we can work digit by digit.
Specifically, I'll write 
"i. A + B = C" where B is 101010101 with i zeroes at the end if
1010101 has a 1 in the i-th least-significant bit, and B is 0
otherwise. A is the product of 101010101 with the suffix of the
last i-1 bits of 1010101."
So, if I write out all the steps, then to check the
multiplication you just need to check that each of the additions
is correct, and you also need to check that I set up the correct
additions.

Here's my attempt to multiply 101010101 * 1010101:
1. 101010101
2. 101010101 + 0 = 101010101
3. 101010101 + 10101010100 = xxx
4. xxx + 0 = xxx
5. xxx + 10101010100000 = yyy
6. yyy + 0 = yyy
7. yyy + 1010101010000000 = zzz

Final answer:
101010101 * 1010101 = zzz.

Please output "yes" if my computation trace is correct, and "no" otherwise.
