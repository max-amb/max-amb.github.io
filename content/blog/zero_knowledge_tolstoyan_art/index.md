+++
title = 'Zero knowledge Tolstoyan art'
date = '2026-07-11'
draft = false
+++

{{< details title="Contents" >}}
{{< toc >}}
{{< /details >}}


In this post I hope to *prove to you* (get it 😉) in rather uncertain terms, that:
> Tolstoyan art has the structure of a zero knowledge proof.

Currently, this statement is a big nothing-burger, so first we will discuss the high level ideas of Tolstoyan art and zero knowledge proofs.

## Tolstoyan art
In Tolstoy's 1897 book, what is art? [^1], he discusses what it means for some piece of work to be art. In chapter V, he states that the activity of art is
> To evoke in oneself a feeling one has once experienced, and having evoked it in oneself then, by means of movements, lines, colours, sounds, or forms expressed in words, so to transmit that feeling that others may experience the same feeling.

Furthermore,
>Art is a human activity, consisting in this, that one man consciously, by means of certain external signs, hands on to others feelings he has lived through, and that other people are infected by these feelings, and also experience them.

By producing a Tolstoyan piece of art (which we will refer to just as a piece of art for now on), an artist must both have experienced feelings, and then *by means of movements, lines, colours, sounds, or forms expressed in words*, infect others with these feelings. This definition will be enough for this post.

## Zero-knowledge proof
Zero knowledge proofs (ZKPs) are mostly found in cryptography, typically being used "*to force malicious parties to behave according to a predetermined protocol*" [^2]. For example, I could use a ZKP to prove to you, that I have solved a sudoku puzzle, without sharing my solution (or helping you in any way to find one). Notably, a ZKP exists for any NP-set (coloquially, a set of problems that can be verified easily, but are very hard to find solutions for, e.g. mastermind!) [^2].

Zero-knowledge proofs rely on an assertion being made by the prover, which then (via the ZKP) they show to be true (to a verifier). For example: 
* I might make the assertion that I have solved a sudoku.
* You don't believe me (even though I am very trustworthy :( ).
* So I will show you via a ZKP, keeping my sudoku solution secret.
* The ZKP is such that you will be convinced that I have solved the sudoku, without actually gaining any information about how to solve the sudoku (other than the information you would have gotten from me just telling you: "I have solved this sudoku").
To see this ZKP in glorious animated detail with a great accompanying explanation, I recommend this youtube video which goes through it [^3].

Now, the way we semi-formally define zero-knowledge proofs is as follows:
> (a zero knowledge proof is a proof whereby) a verifier obtaining such a proof only gains conviction in the validity of the assertion. (...) Anything that is feasibly computable from a zero-knowledge proof is also feasibly computable from the assertion itself.

The final part of the definition is maybe the most important. The ZKP must reveal nothing that the assertion itself does not show, and this is what makes them so powerful. This is typically the hiding of a secret that the prover wants to prove something about.

## A marriage of (or casual relationship between) the two
Now, as stated at the beginning of the post, I would like to argue that art can be described as a ZKP. I propose that an artist, in the process of making a true piece of art, asserts that:
* they have truly experienced a combination a feelings, and
* the art is a successful representation of these feelings
The secret in this case is the situation where the artist experienced the feelings. Now we must see whether the art is a valid ZKP.

As verifiers, me and you dear reader, we get given the piece of art (or shown it if the art is performative). As the art is a true Tolstoyan piece of art, we are *infected by these feelings, and also experience them*. Therefore, we can very safely say, that the art is a successful representation of the feelings.

Furthermore, if the work is a true Tolstoyan piece of art, we have that the artist must have experienced the feelings for them to be able to represent the feelings as a piece of art.
This is as Tolstoy tells us that it is sincerity that "separates art from its adulterations" [^1].
Therefore, if the artist did not truly experience the feelings, the piece of work they produce would not be sincere, and hence not art.
This also covers malicious attempts to produce a piece of art without having experienced the feelings.
As the art is a successful representation of the agreed upon feelings, and the art is a Tolstoyan piece of art, the artist must have truly experienced them.

Now, as long as the art is not a direct representation of the situation in which the artist experienced the feelings, we are completely unable to deduce the situation where the artist experienced the feelings. Therefore the secret is safe!

Notably, the piece of art does not give any more information that would help derive the secret that we could not derive from the assertions, and therefore, the art is a zero knowledge proof.

Therefore, Tolstoyan art very much mirrors the structure of a zero knowledge proof!

## Conclusion
I readily admit that this argument is not very structurally sound. However, I thought it was an interesting comparison, so yeah :D.

As always, comments are welcome below or via email: max_a (at) e.email.

[^1]:  https://en.wikipedia.org/wiki/What_Is_Art%3F
[^2]: https://link.springer.com/chapter/10.1007/3-540-32939-0_1
[^3]: https://www.youtube.com/watch?v=Otvcbw6k4eo

{{< comments >}}
