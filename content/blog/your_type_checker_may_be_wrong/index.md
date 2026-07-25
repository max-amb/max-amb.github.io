+++
title = "Your type checker may be wrong - an introduction to formal proof verification and the Curry-Howard Correspondence"
date = "2026-07-25T09:50:01+01:00"
+++

{{< details title="Contents">}}
{{< toc >}}
{{< /details >}}

When writing code many of us have been saved time and time again by type checkers: the useful piece of software that ensures you aren't adding a string to an integer [^1] or returning a reference to a value instead of the owned value. However, while useful, and sometimes annoying, it seems that the humble type checker has limited capability beyond its noble task of saving our asses...

Therefore, one may be surprised by the fact that type checkers also form the backbone of proof assistants - languages like Lean and Rocq. They use the structure gained from types to definitively check whether some statement follows from some other statements - or more colloquially, to verify mathematical proofs.

In this blog I would like to first introduce some of the more basic elements of the Curry-Howard correspondence, followed by how it is used in proof assistants and eventually, why this may mean your type checker is *wrong* (or just, may not know you are right).

## The Curry-Howard correspondence
A light definition of the Curry-Howard (CH) correspondence could be:
>proofs can be represented as programs, (...), proofs can be run[^2]

This definition certainly doesn't give much away, however one possible line of thought may derive that as proofs can be represented as programs, we need a way for programs to return proofs. But what does this even mean to return a proof?

Reverting back to fundamentals: in order for a program to return an integer, we say that it returns the type $\text{int}$, which could be any integer. So $\text{int}$ represents the set of integers. Similarly, for a program to return $\text{True}$ or $\text{False}$, we say it returns the type $\text{bool}$, which contains both possibilities. This approximation of types as sets isn't strictly correct, but it is enough for the purposes of this post.

Attempting to extend this to proofs, one may say that when a program returns a proof of some fact, it returns an element of the type $P(X)$. Where $P(X)$ is the set of all proofs of fact $X$. 

To seriously work with our new proof object, we must first translate a few logical operations from propositional and predicate logic to our new paradigm. We begin with the most simple:
$$
X \text{ is true}
$$
In our case, for $X$ to be true, we must have a proof of $X$, i.e.
$$
\exists p : p \in P(X)
$$
The way we say this is: $P(X)$ is inhabited [^3]. For example, $P(5=5)$ is inhabited but $P(5+2=6)$ is not (as there is no proof of this in peano arithmetic, it is false).

The next logical operation to represent is "and" ($\wedge$). For those unfamiliar, we have $X \wedge Y$ if and only if both $X$ and $Y$ are true. So we have that both $P(X)$ and $P(Y)$ are inhabited, i.e. $\exists p: p \in P(X)$  and $\exists p\prime : p\prime \in P(Y)$. This means we can construct an object $(p, p\prime)$, so we have that
$$
P(X) \times P(Y)
$$
(where $\times$ represents the cartesian product) is inhabited ($(p, p\prime) \in  P(X) \times P(Y)$).

Next, we want to represent the implication operation. If $X \implies Y$ then either $X$ is false, or $X$ is true and $Y$ is true. We represent this as the existence of a function from $P(X)$ to $P(Y)$:
$$
P(X) \to P(Y)
$$
If this function exists, then whenever we have a proof of $X$, we can derive a proof of $Y$. If $X$ is false (so $P(X)$ is not inhabited), then the function does not have its input, so $P(Y)$ may or may not hold.

In the interests of conciseness, I omit discussions about the rest of the standard logical operations. They translate to set theory like so (but are not relevant for the remainder of the post):

| Logical operation                 | Set theory                                            |
| --------------------------------- | ----------------------------------------------------- |
| $X \lor Y$                        | $P(X) + P(Y)$ (Where $+$ represents a disjoint union) |
| $(\forall(n \in \mathbb{N})X(n))$ | $(n: \mathbb{N}) \to P(X(n))$                         |
| $(\exists(n \in \mathbb{N})X(N))$ | $(n: \mathbb{N}) \times P(X(n))$                      |
[^4]
## How do proof assistants use the CH correspondence
Proof assistants use this correspondence and their type checkers to validate proofs. But how do they do this? To show how, let us prove a simple theorem using [lean](https://lean-lang.org/) notation.

### A theorem
Consider the below theorem named `blog`:
```lean4
theorem blog (X Y Z: Prop) (h_1: X) (h_2: Y) (h_3: Y → Z) : X ∧ Z
```
It begins by stating that $X, Y$ and $Z$ are logical statements, i.e. they are things that may or may not be true. This is like constructing the sets $P(X), P(Y)$ and $P(Z)$, but not saying if they are inhabited yet.

Next we have a hypothesis, $h_1 : X$, which serves as a proof of $X$. Thinking back to our earlier discussion, you may remember that having a proof of $X$, is equivalent to $X$ being true, so $h_1 : X$ simply gives, $X$ is true. This is similar with $h_2$, but $h_2$ gives a proof of $Y$.

Then we have final hypothesis, which using unbounded creativity and intellect, I have named $h_3$. It is of type
$$
Y \to Z
$$
This means that a there is a function that from a proof of $Y$, allows us to get a proof of $Z$, this is equivalent to $Y \implies Z$ (similarly discussed before).

The final component of the theorem is the desired result, $X \land Z$. To show this, we must construct an element that inhabites $P(X) \times P(Z)$, and to do this, we have to construct proofs of both $X$ and $Z$.

Phew, that was a lot, and all just to understand the theorem statement 😨. However, I hope you can see now how the earlier discussion of mapping logical operations to set theory has allowed us to transpose the theorem from the language of types, to the land of logic.
### A proof of the theorem
Now we are tasked with proving the theorem... there really is no rest for the mathematic (it sounds like wicked!).

The keen eyed among you may have noted that we already have one of our desired components. We need elements to inhabit both $P(X)$ and $P(Z)$, and we have $h_1$, which is a proof of $X$ (and hence is suitable to inhabit $P(X)$). What a slam dunk.

Now we need to show $P(Z)$. Well, we have $h_2 : Y$ and $h_3$ which is a function which takes a proof of $Y$ and gives us a proof of $Z$. Hold on, that just works ™. So, by passing in our proof of $Y$ ($h_2$) to $h_3$, we get a proof of $Z$ which can inhabit $P(Z)$.

Now, how to write this in lean? There are many ways, but one such way follows (please try another way yourself and leave it in the comments):

We begin by deconstructing our desired result into two parts which we must fill in succession. Using the statement `constructor` we request lean tells us what we need to do to achieve our desired result. Faithfully, lean spits out two goals, one to fill $X$ and one to fill $Z$.

To fill $X$ we can just tell lean it is $h_1$, with `exact h_1`. To fill $Z$ we need to apply $h_3$ to $h_2$ (remember, $h_3$ is a function). To do this in lean it is simply `h_3 h_2` (or you can write it `h_3 (h_2)` to make it resemble more non-functional languages). So, we define a variable `z`, to be of type `Z`:
```
have z := h_3 h_2
```
and finish off with `exact` once again.

Lean congratulates us with a "Goals accomplished 🎉" message and we can once again show our faces in public. The full code is
```lean
theorem blog (X Y Z: Prop) (h_1: X) (h_2: Y) (h_3: Y → Z) : X ∧ Z := by
  constructor
  exact h_1
  have z: Z := h_3 (h_2)
  exact z
```
## The type checker
At the beginning of the post I promised to argue why your type checker maybe wrong, and now I shall do so. As discussed before, in order for a proof assistant to verify you have proven your desired result, it checks if you have managed to output the correct type (something to inhabit $P(\text{whatever you wanted to prove}$)).

### Limitations of the type checker
In order for the type checker to safely assert you have done this, it needs to evaluate the types at each expression in the sequence of expressions you have provided it. There is a problem with this requirement. It requires all the expressions to finish evaluating. There are two ways an expression may never finish evaluating:

A bit of a vacuous case is `exit()` in c or python, escaping the requirement to finish evaluating by just quitting the program. What is the type checker to do with this? The way we solve this is by restricting the expressions allowed in the programming language itself (and this is what languages like Lean and Agda have done!).

The next way an expression may never finish evaluating is harder to resolve (I mean it is actually impossible but I do not wish to dampen the jovial mood we have going on here). We must ensure that the sequence of expressions does not get caught up in some sort of infinite loop as then they would never finish. So, we just need a way to check if our sequence of expressions will ever halt with the input we give it. 

This... is unfortunately impossible to do in finite time (oh look now I have stomped on the vibe, I am sorry dear reader :( ). In 1936, Alan Turing proved that the problem of whether a program will halt in a finite time (known as the halting problem) is not decidable (which means not computable in finite time). If you wish to gain some intuition for the Turing machines he used to prove this you may be interested in my post [here]({{< relref "../an_introduction_to_turing_machines_and_computation/index.md" >}}) about the topic.

The way formal languages attempt to sidestep this fact is by further restricting the language of computation. Recursion is provably finite in some cases, for example recursion downwards on the naturals [^7] . So by only allowing recursion that can be proven to stop, we ensure the type checker always finishes in finite time. Note that this is not a limitation of current technology or software, this is a fundamental limitation of proof assistants which cannot be fixed. There will always exist results which are impossible to validate a proof to.

### Mathematical consequences (and proof)
This restriction limits the expressiveness of these languages sufficiently to mean that they are unable to represent every possible proof. But why is this?

Let us assume (for the purposes of contradiction) that the restricted language is expressive enough to represent a proof or disproof for every propositional statement, i.e. $\forall S$ we can prove either $S$ or $\lnot S$ within our language. Now, since we have assumed our language to be omniscient, lets have some fun...

Consider an arbitrary program $P$ with some finite collection of arbitrary inputs $A$ and the proposition $H$ that $P$ will halt (stop) on inputs $A$. Now, using our language, we know we can write a proof (or disproof) of this and verify it in finite time. 

The way we do this is by generating all possible proofs [^8] of the $H$ and $\lnot H$, and using our type checker to check if one is valid. Since the type checker runs in finite time and one of the proofs is correct (as our language is sufficiently expressive), this process is finite. Since we can do this for any program, we have now managed to check whether an arbitrary program will halt in finite time!

This is however, as previously discussed, impossible (Turing's halting problem again). So, our assumptions must be incorrect, and therefore the restricted language is not expressive enough to represent a proof or disproof for every propositional statement.

The CH-correspondence has given us that proofs and programs are equivalent, but this now leads to a disturbing fact. If our language, which is necessarily restricted, cannot prove or disprove some propositions, then it suggest it may be impossible to do so in general...

This is a famous problem in mathematics. Gödel's first incompleteness theorem states that any consistent formal system within which a certain amount of elementary arithmetic can be carried out is incomplete [^5]. Colloquially, for any formal system useful for calculation, of which every formal system in mathematics is, there exist statements which can neither be proven or disproven.

Now, we have proved that the negation of Gödel's first incompleteness theorem implies the negation of Turings halting problem, and therefore, via the contraposative the Curry-Howard correspondence has suggested an absolutely stunning result.

**Turing's halting problem implies Gödel's incompleteness theorem.**

To gain some intuition for the contraposative perhaps read my fish focused approach to the contraposative [here]({{< relref "../some_intuition_behind_the_contrapositive/index.md" >}}).

### Your type checker maybe wrong
We have now seen that the type checker is not perfect, in fact, it is provably imperfect. Therefore... in some cases... your type checker maybe... **WRONG** [^9]. There may be a case where your type checker has rejected your code (in order to not run forever), but it actually was correct.

Of course this is unlikely, the type checker recursion limit in rust for example is 128 [^6]. But it is possible, and therefore, when your colleagues are complaining about your code not type checking, just know... you may be correct (although unlikely :) )[^10].

## Conclusion
In this post we have shown that the type checker may not be able to verify your code is correct, but in a very roundabout way. It will however never accept incorrect code, so if it accepts your code, you can be safe in the knowledge that it is correct. Now just to find that logic bug...
{{< comments >}}

[^1]: Notably excluding javascript where masterpieces like $5 + \text{'five'}$ are possible

[^2]: https://en.wikipedia.org/wiki/Curry%E2%80%93Howard_correspondence

[^3]: https://en.wikipedia.org/wiki/Inhabited_set

[^4]: https://cklixx.people.wm.edu/teaching/math400/Wesley-P1.pdf

[^5]: https://en.wikipedia.org/wiki/G%C3%B6del%27s_incompleteness_theorems#First_incompleteness_theorem

[^6]: https://doc.rust-lang.org/reference/attributes/limits.html

[^7]: https://lean4.dev/language/control-flow/recursion

[^8]: By a possible proof we mean any possible sequence of syntactic expressions in the language.

[^9]: Well... actually we call this incomplete, not wrong. It is not going to accept something incorrect, it just may not accept something that is correct.

[^10]: The process the Rust type checker goes through to check your code is of course different to the one undertaken by the checker in Lean or Agda, but the fundamental limitation is the same (so the joke works 😁).
