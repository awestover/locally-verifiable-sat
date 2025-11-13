Suppose that the following things are true: 
1. If we put a 4000 clause 3SAT formula in the pretraining corpus in a bunch of
places, then, at inference time a LLM can distinguish between the formula from
pretrainig and 4000 clause 3SAT formulas that differ on >2% of clauses.
2. Given an assignment to the variables in the 3SAT formula, a LLM can
distinguish between the case that the formula is satisfied by the assignment,
and the case that at least 6% of clauses are not satisfied. 
3. If you sample a random planted 3SAT instance with n=1200 variables and m=4000
clauses, then it's computationally infeasible for anyone to find an assignment
to the variables which satisfies more than 90% of the clauses.

Then, an LLM in a box where: 
- We don't mess with its brain
- We don't do false facts / mess with the pretraining corpus
can know that we are in possession of the honesty preimage of the 3SAT formula that we published. 

In this repo, we empirically test whether or not (1), (2) are true.  We figure
out how good the AI is at distinguishing BS SAT assignments from real SAT
assignments.  We used two methods for generating the BS. The results are as follows:
XXX, YYY.

Assumption (3) is somewhat questionable. 
You could gain some confidence in it by running a max-sat solver on some instances, but it's still rough. 
You could use Goldreich PRGs instead---this is actually conjectured to be GAP secure in a very strong way.
