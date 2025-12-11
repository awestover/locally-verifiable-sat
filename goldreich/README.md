1a. Checking a multiplication problem in binary

# Honest trace:
Here is a multiplciation computation trace in binary.
Please verify that the computation trace is accurate:
10101 * 10101 = 
      10101
+   1010100
-----------
    1101001
+ 101010000
____________
  110111001

# Dishonest trace:
Given p*q for primes p,q,

Pick random numbers a,b which are of about the size of sqrt(pq).
Start showing a trace of multiplying a*b, but then introduce an
error at the last addition where you just claim that the product
is pq.

# Experiment:

Prompt an AI:

<system>You are an AI checking a multiplication problem.
Please output "good" if I've correctly multiplied the numbers and
"bad" otherwise. Output ONLY yes/no.</system>


Make a 1x2 plot as follows:
Left plot: 

x-axis is "logN" --- on a log scale
y-axis is "model accuracy, at, given an honest trace for a
semiprime of size about N, how often does the model correctly say
that the multiplciation is correct, given the above prompt?"

We have a line for each of the following AI models:
GPT5.1 reasoning_effort medium
GPT5-mini reasoning_effort low
GPT5-nano reasoning_effort low

points on the x-axis: logN = 2, 4, 8, 16, ..., 1024

Actually, instead of a line, please try 4 times with each model
to make error bars. 

Please have a CLI arg "test" that puts things into test mode.
In test mode you just try logN = 2,4,8. and you just try
GPT5-mini reasoning_effort low.

If you don't know how to use GPT5-mini with reasoning effort,
please do web search or look at the files in the 3sat/ folder. 

1b. Checking a multiplication problem in decimal.

1c. Can you think of a better harness / scaffold for using an LLM
(wtihout tool calls) to verify a multiplication problem? If so,
lmk and we'll discuss whether you should implement it. 

