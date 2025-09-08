+++
title = "Minimising the product of sums chosen from intervals"
date = "2025-09-07T00:00:00+01:00"
draft = false
+++

{{< details title="Contents">}}
{{< toc >}}
{{< /details >}}

What a catchy name!
I'd like to explain this problem, and then provide a wordy proof for a general case.
All the proofs are hidden in drop downs to motivate you to try them yourself and to make the blog less dense, if you have any questions about the proofs (or anything else), please leave them below!

## The problem
In this post I would like to present a problem that my Dad showed me a while back from a Dutch mathematics magazine called Pythagoras that he had read when he was a child.
It was an episode from December 93[^1] and it was the first problem in the so-called "Pythagoras Olympiad".
The problem is described as follows:

> The numbers from $1$ to $100$ are divided into $10$ groups of $10$. The sum of the numbers in the j'th group is called $s_j$.
In which distribution is the product $s_1 \cdot s_2 \cdot s_3 \dots \cdot s_9 \cdot s_{10}$ as small as possible.

If this isn't clear for you here is another way of describing the problem:

> We have $10$ sets of cardinality $10$ ($S_1, S_2, S_3 \dots, S_9, S_{10}$), we fill each set by picking from the interval $[1, 100]$.
The sets share no common elements, so all of $[1, 100]$ is covered.
$s_n$ is the sum of all of the elements in $S_n$.
Minimise $s_1 \cdot s_2 \cdot s_3 \dots \cdot s_9 \cdot s_{10}$

### An example
One example of a configuration we could have:

$$
\begin{aligned}
S_1 &= \{1, 2, 3, 4, 5, 6, 7, 8, 9, 10\} \\
S_2 &= \{11, 12, 13, 14, 15, 16,17,18,19,20\} \\
\dots \\
S_{10} &= \{91, 92, 93, 94, 95, 96, 97, 98, 99, 100\}
\end{aligned}
$$

Giving:

$$
\begin{aligned}
s_1 &= 55 \\
s_2 &= 155  \\
& \dots \\
s_{10} &= 955
\end{aligned}
$$
We can spot this is the case due to the similarities between all the $s_n$'s meaning $s_n$ is $100$ less than $s_{n+1}$.
I.e. $s_{n+1} = s_{n} + 100$
This is because each element in $s_{n+1}$ is 10 greater than its adjacent element in $s_n$, for example: $11$ is $10$ greater than $1$, $12$ is $10$ greater than $2$.

So our product $p$ is:

$$
\begin{aligned}
p &= 55 \cdot 155 \cdot \dots \cdot 955 \\ 
&\approx 7.87 \times 10^{25}
\end{aligned}
$$

## The solution
We can attack this problem by repeatedly (well twice...) restricting our solution space until only one solution is left.

### Lemma one
We begin by saying none of our sums can be equal:

> Lemma: If $s_n = s_m$ ($n \neq m$), then swapping the elements strictly decreases the product

{{< details title="Proof one">}}
> Proof: We proceed with direct proof\
Take $a\in S_n$ and $b \in S_m$ and swap them\
This means that $s_n' = s_n - (a - b)$ and $s_m' = s_m + (a - b)$(writing it like this makes it easier later on)\
Now our product, $p = s_n \cdot s_m$ is what we are attempting to prove is not optimum\
We also let $p' = s_n' \cdot s_m'$ which is our updated $p$.

$$
\begin{aligned}
p' &= s_n' \cdot s_m' \\
&= (s_n - (a-b))(s_m + (a-b)) \\
&= s_n \cdot s_m + s_n(a-b) - s_m (a-b) - (a-b)^2 \\
&= p + (a-b)(s_n - s_m) - (a-b)^2 \\
\text{ As } s_n = s_m &\implies p + (a-b)(0) - (a-b)^2  \\
&= p - (a-b)^2
\end{aligned}
$$

> This means $p' \lt p$ because $(a-b)^2 \gt 0$\
$(a-b)^2 \gt 0$ because $a \neq b$ because the sets share no common values\
This means that, if $s_n = s_m$ ($n \neq m$) the product can be further minimised by switching values.

{{< /details >}}

> Corollary: Without Loss Of Generality (WLOG) $s_1 \lt s_2 \lt \dots \lt s_{10}$ in our solution

### Lemma two
Next, we can make a plea to intuition and argue that if there is a larger number in a smaller sum group, swapping it out will reduce the product:

<!--
> Lemma: If $\exists \ a \in S_n, b \in S_m$ such that $a < b$ and $n>m$ then the product is not optimum
-->

> Lemma: If $s_n \lt s_m$ then $\forall a \in S_n, b \in S_m$ we have $a \lt b$ in the optimum solution, i.e. in the optimal solution, if one group has a smaller sum than another, then every element in the smaller group must also be smaller than every element in the larger group

{{< details title="Proof two">}}

> Proof: We proceed by way of contradiction\
For the purposes of contradiction we will argue $\exists \ a \in S_n, b \in S_m$ such that $a > b$ and $s_n \lt s_m$ in an optimum product\
So we have our assumed optimum product, $p = s_n \cdot s_m$\
Let us proceed as we have done before, lets swap our $a$ and $b$: $s_n' = s_n - (a - b)$ and $s_m' = s_m + (a - b)$\
We can now start with our new product, $p' = s_n' \cdot s_m'$

$$
\begin{aligned}
p' &= s_n' \cdot s_m' \\
&= (s_n - (a-b))(s_m + (a-b)) \\
&= s_n \cdot s_m + s_n(a-b) - s_m (a-b) - (a-b)^2 \\
&= p + (a-b)(s_n - s_m) - (a-b)^2 \\
\text{As } a > b \land s_n \lt s_m &\implies (a-b)(s_n - s_m) = -k \text{ where } k \text{ is a positive number}
\text{So } &= p - k - (a-b)^2
\end{aligned}
$$

> This means $p' \lt p$ because $(a-b)^2 \gt 0$ and $k \gt 0$\
So, we have a contradiction, $p$ is not our optimum product, because $p' \lt p$\
Therefore, if $s_n \lt s_m$ then $\forall a \in S_n, b \in S_m$ we have $a \lt b$ in the optimum solution 

{{< /details >}}
Note this lemma requires [lemma one](#lemma-one) as it proved that in the optimum solution, we have the case that $s_n \lt s_m$, not $s_n = s_m$.

### Conclusion
This leaves us with only one option as [lemma one](#lemma-one) and [lemma two](#lemma-two) force the sets to be arranged consecutively, starting with the smallest numbers.
In $S_1$, all elements must be lower than all other elements in $S_2, \dots, S_{10}$, this means 
$$
S_1 = \left\{1, 2, 3, 4, 5, 6, 7, 8, 9, 10\right\}
$$
Now in $S_2$, all elements must be lower than all other elements in $S_3, \dots, S_{10}$:
$$
S_2 = \left\{11, 12, 13, 14, 15, 16, 17, 18, 19, 20\right\}
$$
Thus, the optimum configuration is the one we had in the [example](#an-example)!

## Generalisation
You may have noticed that in both [lemma one](#lemma-one) and [lemma two](#lemma-two), we never used the fact we were working with $10$ sets of $10$, or that these numbers were being chosen from an interval of $[1,100]$.
This means we can immediately generalise our solution to working on $k$ sets of the cardinality $j$ over an interval of size $k\cdot j$.

Furthermore, you may notice we didn't use the fact that $[1,100]$ is contiguous, we could similarly have done the interval from $[2, 200]$ but with only even numbers, still using $10$ sets of cardinality $10$.
This means we can just use any set of real distinct numbers, not necessarily just an interval.

Therefore, we can restate the problem like so:

> Given a set of real distinct numbers length $k \cdot j$ (not necessarily an interval), and $k$ sets of cardinality $j$ that share no common elements hence covering the entire set.\
$s_n$ is the sum of the elements of $k_n$.\
Minimise $s_1 \cdot s_2 \cdot s_3 \dots \cdot s_{k-1} \cdot s_{k}$

and our solution is still valid.

{{< comments >}}

[^1]: https://pyth.eu/uploads/user/ArchiefPDF/Pyth23-2.pdf
