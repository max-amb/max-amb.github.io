---
title: "The main components of the MLP"
date: 2025-08-14T21:41:40+01:00
series: 'The making of a MLP'
draft: true
toc: true 
---

This is the second iteration in the [series](https://max-amb.github.io/series/making-a-mlp-in-rust-from-scratch-for-mnist/) where we are building a Multi-Layer-Perceptron (MLP) from scratch!
If you have not read the previous post in the series 
This post will consider the main sections of the code in the MLP, which are the:
* `training` function (to train the network),
* `backpropagation` function (to perform backpropagation),
* `forward_pass` function (for performing the forward pass),
* `new` function (to create a new MLP for training/use)

I intend to walk through the entirety of the code in these functions as well as explain each lines function.
This should also make the operations more intuitive.
