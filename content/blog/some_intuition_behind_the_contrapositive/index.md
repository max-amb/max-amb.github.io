+++
title = 'Some intuition behind the contrapositive'
date = '2026-06-27'
draft = false
+++

{{< details title="Contents" >}}
{{< toc >}}
{{< /details >}}

# Introduction
While quite a fundamental law in logic, the contrapositive is certainly not immediately intuitive (or at least it was not to me).
After someone else expressed a similar feeling, I decided to try and derive some intuition to explain the rule, and I think I have succeeded.
This blog could also be treated as an introduction to the contrapositive.

Just so everyone is on the same page, here is the rule in all its formal glory:
$$
P \to Q \leftrightarrow (\lnot Q) \to (\lnot P)
$$
where $P, Q$ are logical propositions and $\lnot$ represents the following proposition being false.
Stated colloquially, if having one thing ($P$) gives us that we have another thing ($Q$), then if we do not have this other thing ($\lnot Q$), then we cannot have our original thing ($\lnot P$).

# The explanation
Consider a fish, an alive and healthy fish. To spark your imagination, I have used all of my artistic talent to provide a diagram:

{{<
    figure-dynamic
    dark-src="./images/fish_dark.svg"
    light-src="./images/fish_light.svg"
    alt="An outline drawing of a fish"
>}}

Those who are expert marine biologists, may have noted, that my diagram (while beautiful) does not consider the full situation of a healthy fish.
If we have a healthy fish, then we must have a body of water (I will call it a lake for conciseness), for the fish, otherwise it will die :( [^1].
I have worked hard to produce another diagram to show this relation:

{{<
    figure-dynamic
    dark-src="./images/fish_in_lake_dark.svg"
    light-src="./images/fish_in_lake_light.svg"
    alt="An outline drawing of a fish in a lake"
>}}

Therefore, if we have a healthy fish, then we have a lake, or in notation:
$F \to L$
where $F$ represents having a fish, and $L$ represents having a lake.
Take a moment to really make sure this makes sense to you, both the notational and picture form:

{{<
    figure-dynamic
    dark-src="./images/fish_impl_dark.svg"
    light-src="./images/fish_impl_light.svg"
    alt="An outline drawing of a happy fish, followed by a right pointing arrow (representing implication), followed by an outline drawing of a lake"
>}}

Now, fish lovers may wish to stop reading here, and I implore them to do so!
Bathe in the enjoyment that we have a healthy fish, and worry no more my friend.

However, in the unrelenting interest of science, we must now do something quite cruel... we must... take away the water :((. Expressed in picture form,

{{<
    figure-dynamic
    dark-src="./images/no_lake_dark.svg"
    light-src="./images/no_lake_light.svg"
    alt="An outline drawing of a lake with a big red cross overlaying it"
>}}

So, after quite the monumentous task of taking away all the lakes (how did you do it reader??) we must now consider the consequences of our actions.
It seems, dear reader, that you were so preoccupied with whether or not you could, you didn't stop to think if you should [^2].

The situation for our healthy fish has now changed significantly.
Earlier we noted that in order to have a healthy fish, we needed a body of water for the fish.
However, now we have no body of water, therefore, our fish is no longer healthy (it is likely dead).
I provide a graphic (in two ways!!!) representation of this fact below:

{{<
    figure-dynamic
    dark-src="./images/dead_fish_dark.svg"
    light-src="./images/dead_fish_light.svg"
    alt="An outline drawing of a fish with crosses for eyes to indicate, it is dead"
>}}

While cruel, our actions have now illuminated the right hand side of the contrapositive rule.
If we have no lakes, then our fish is not healthy, or
$$
\lnot L \to \lnot F
$$
Drawn in all of its horrifying detail:

{{<
    figure-dynamic
    dark-src="./images/dead_fish_impl_dark.svg"
    light-src="./images/dead_fish_impl_light.svg"
    alt="An outline drawing of a lake with a big red cross overlaying it, followed by a right pointing arrow (representing implication), followed by an outline drawing of a dead fish"
>}}

Though our methods were unconventional, we have now managed to demonstrate an example of the contrapositive!
If we have a healthy fish, then we have lakes.
If we have no lakes, then the fish are not healthy.
$$
F \to L \leftrightarrow  \lnot L \to \lnot F
$$

# Conclusion
I hope this example was somewhat illuminating and entertaining. As always, any questions/comments are welcome below or by email: max_a (at) e.email!

[^1]: Every animal needs water to survive (see this comprehensive source: https://www.bbc.co.uk/bitesize/articles/z343f82), but it's easier to draw how fish depend on water.
[^2]: https://knowyourmeme.com/memes/your-scientists-were-so-preoccupied-with-whether-or-not-they-could-they-didnt-stop-to-think-if-they-should
{{< comments >}}
