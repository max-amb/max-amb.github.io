+++
title = "The main components of the MLP"
date = "2025-08-17T00:00:00+01:00"
series =  'The making of a MLP'
draft = false
+++

{{< details title="Contents">}}
{{< toc >}}
{{< /details >}}

This is the second iteration in the [series]({{< ref "series/the-making-of-a-mlp" >}}) where we are building a Multi-Layer-Perceptron (MLP) from scratch!
This post will consider the main sections of the code in the MLP, which are the:
* `new` function (to create a new MLP for training/use)
* `forward_pass` function (for performing the forward pass),
* `backpropagation` function (to perform backpropagation),
* `training` function (to train the network),
and it will be majorly focused on implementation details (as well as a small bit of fun added maths).

I intend to walk through the entirety of the code in these functions as well as explain each lines function.
This should also make some of the operations discussed in the first iteration of the series ([the maths post]({{< ref "blog/the_maths_behind_the_MLP" >}})) more intuitive.

It is worth noting that throughout this code the [nalgebra](https://nalgebra.rs/) crate is used extensively so if something containing a `DMatrix` or a `DVector` looks really quite weird, check their [documentation](https://docs.rs/nalgebra/latest/nalgebra/index.html) as it will help.

## The structure of the network
We must begin with a very short exploration of the structure that contains the neural network. While simple, it is used everywhere in the following functions so understanding it can only be beneficial. It is as follows:

```rust
#[derive(Debug, Clone)]
pub struct NN {
    pub layers: Vec<DVector<f32>>,
    pub weights: Vec<DMatrix<f32>>,
    pub biases: Vec<DVector<f32>>,
    pub alpha: f32,
}
```

Firstly, note the fact that this is a `pub` struct.
This is because the entirety of the neural network logic is stored in its own file (`neural_network.rs`), so for it to be accessed in `main.rs` the structure needs to be public.
See more on `pub` [here](https://doc.rust-lang.org/stable/book/ch07-02-defining-modules-to-control-scope-and-privacy.html).

The first line adds two [attributes](https://doc.rust-lang.org/stable/rust-by-example/attribute.html) to the neural network struct. These tell the compiler to add functions to the code that relate to this struct.
The `Debug` attribute allows for the programmer (me) to print out information about the network, like the layers, quickly and easily.
This attribute does not actually add anything to the end user experience but it makes development using the structure much simpler.

The next attribute is the `Clone` attribute. This allows for an instantiation of the `NN` struct to be `.clone()`'d, making a copy of the structure in memory.
This only happens once, in the `training` function, where the network needs to be cloned for a reference to be passed to each thread that is running in parallel (more on that later).

## The new function
This is the first function (obviously after `main`) that is run in the program and it initialises the neural network. Following this the neural network can be trained or used on data. It is worth noting that the `generate_model_from_file` function will also create a neural network but it will have been pre-trained in a previous iteration of the program and then stored in a file.

The `new` function is as follows:
```rust
pub fn new(
    layer_sizes: &[i32],
    initialisation: InitialisationOptions,
    alpha_value: Option<f32>,
) -> NN {
    let number_of_layers = layer_sizes.len();
    let mut rng = rand::rng();

    let layers: Vec<DVector<f32>> = (0..number_of_layers)
        .map(|x| DVector::from_element(layer_sizes[x] as usize, 0.0))
        .collect();

    let weights: Vec<DMatrix<f32>> = match initialisation {
        InitialisationOptions::Random => (1..number_of_layers)
            .map(|x| {
                DMatrix::from_fn(
                    layer_sizes[x] as usize,
                    layer_sizes[x - 1] as usize,
                    |_, _| rng.random_range(-1.0..=1.0),
                )
            })
            .collect(),
        InitialisationOptions::He => (1..number_of_layers)
            .map(|x| {
                let normal_dist =
                    Normal::new(0.0, (2.0_f32 / (layer_sizes[x - 1] as f32)).sqrt())
                        .unwrap();
                DMatrix::from_fn(
                    layer_sizes[x] as usize,
                    layer_sizes[x - 1] as usize,
                    |_, _| {
                        normal_dist.sample(&mut rng)
                    },
                )
            })
            .collect(),
    };

    let biases: Vec<DVector<f32>> = (1..number_of_layers)
        .map(|x| DVector::from_fn(layer_sizes[x] as usize, |_, _| rng.random_range(-1.0..=1.0)))
        .collect();

    let alpha = alpha_value.unwrap_or(0.01);
    Self {
        layers,
        weights,
        biases,
        alpha,
    }
}
```

### Arguments
An example input to the function could be `NN::new(&[784, 128, 128, 128, 10], InitialisationOptions::He, None)` (this example was taken from `main` in `main.rs`).
This would create a network with an input layer of $784$ neurons, three hidden layers each with $128$ neurons, and an output layer of $10$ neurons.
The next argument specifies that the network's weights should be initialised with [He initialisation](https://doi.org/10.48550/arXiv.1502.01852). More on this [later](#initialisation-options)...
The final argument (which is optional demonstrated by the `Option<>` type) specifies the alpha value to be used for the leaky ReLU activation function (if in use).

### Network setup
```rust{lineNoStart = 6}
let number_of_layers = layer_sizes.len();
let mut rng = rand::rng();

let layers: Vec<DVector<f32>> = (0..number_of_layers)
    .map(|x| DVector::from_element(layer_sizes[x] as usize, 0.0))
    .collect();
```

To begin we attain the number of layers for use in our iterators and then we initialise a Random-Number-Generator (RNG) generator using the [rand crate](https://crates.io/crates/rand).

I find the RNG interesting so I talk a bit about it in the [appendix](#appendix), but this is (really quite) far from relevancy in this post so feel free to ignore it!

Finally, we initialise the layers - the vectors containing actual neurons - to vectors full of $0.0$.
We represent the layers as a vector of `DVector`s as we use `DVector`s from the `nalgebra` crate for linear algebra calculations.
The value we choose doesn't matter as the layers neurons (obviously) have their values changed with each forward pass to represent the neural networks calculation.

### Weight initialisation
```rust{lineNoStart = 13}
let weights: Vec<DMatrix<f32>> = match initialisation {
    InitialisationOptions::Random => (1..number_of_layers)
        .map(|x| {
            DMatrix::from_fn(
                layer_sizes[x] as usize,
                layer_sizes[x - 1] as usize,
                |_, _| rng.random_range(-1.0..=1.0),
            )
        })
        .collect(),
    InitialisationOptions::He => (1..number_of_layers)
        .map(|x| {
            let normal_dist =
                Normal::new(0.0, (2.0_f32 / (layer_sizes[x - 1] as f32)).sqrt())
                    .unwrap();
            DMatrix::from_fn(
                layer_sizes[x] as usize,
                layer_sizes[x - 1] as usize,
                |_, _| {
                    normal_dist.sample(&mut rng)
                },
            )
        })
        .collect(),
};
```

Initially it is clear that weights depends on the `initialisation` variable and by association the `InitialisationOptions` enum.

#### Initialisation options
The initialisation options enum is simple:
```rust
#[derive(Default)]
pub enum InitialisationOptions {
    Random,
    #[default]
    He,
}
```
We derive `Default` so we can set our default value, in this case `He`.
These seperate values represent two different ways to initialise weights.

We can initialise the weights completely randomly, with `InitialisationOptions::Random`, and we can also initialises them with `InitialisationOptions::He` for He initialisation.

But what is He initialisation?
He initialisation is the method of initialisating weights written about by [Kaiming He](https://en.wikipedia.org/wiki/Kaiming_He) in his [paper on ReLU activation for image classification](https://doi.org/10.48550/arXiv.1502.01852).
It aims to minimise vanishing or exploding activations by keeping the variance of the layers values approximately the same at $\frac{2}{n_{in}}$ where $n_{in}$ is the number of neurons in the input layer of that weights matrix.
That is, for $\omega^{[l]}$ we use $\mathcal{N}(0, \frac{2}{\text{size of }l-1})$.
I may do a short blog post on this in the future, so please let me know in the comments if you would be interested in that!

Once we determine which initialisation option is being used via the `match` statement we apply this initialisation option.
Random initialisation, as promised, fills the weights vector with a selection of `DMatrix`s filled with random numbers between $-1$ and $1$ (via the `rng.random_range()` function).
He initialisation creates a normal distribution for each layer, using the layer size of the last layer as $n_{in}$. It then repeatedly samples from that distribution to fill that weights matrix.

> The keen eyed among you may have spotted that we use `(2.0_f32 / (layer_sizes[x - 1] as f32)).sqrt()` in the normal distribution initialisation.
> This is because of the input to `Normal::new()` expects a standard deviation and obviously standard deviation is the square root of variance.

### Bias initialisation
The final integral part of the neural network making process is bias initialisation:
```rust{lineNoStart=39}
let biases: Vec<DVector<f32>> = (1..number_of_layers)
    .map(|x| DVector::from_fn(layer_sizes[x] as usize, |_, _| rng.random_range(-1.0..=1.0)))
    .collect();
```
This simply randomly generates biases to fill the bias vectors for each layer (excluding the input layer via `(1..`).

### Returning
We now need to return all the amazing work we have done (as well as store the alpha value):
```rust{lineNoStart=43}
let alpha = alpha_value.unwrap_or(0.01);
Self {
    layers,
    weights,
    biases,
    alpha,
}
```

The first line checks if the the function has been passed in an explicit alpha value (or just `None`).
If so, this explicit alpha value is used (for example during tests in `tests.rs`) but otherwise the `unwrap_or()` ensures that that default alpha value is $0.01$.
Just to reiterate, the alpha value is the value used in leaky ReLU for the gradient $<0$. In tests $0.2$ is used but generally a lower value (like $0.01$) is used.

Finally, we return a `NN` struct that contains all our newly generated layers, weights, biases and alpha values.
An interesting thing we have done here is to not explicitly state the components of the struct, i.e.:
```rust
Self {
    layers: layers,
    weights: weights,
    biases: biases,
    alpha: alpha,
}
```

This is because the variable names are exactly the same as the structures component names, meaning it is what rust calls [redundant field names](https://rust-lang.github.io/rust-clippy/master/index.html#redundant_field_names).

## Conclusion
This concludes the post as of now, I am going on holiday and aim to come back early September when I will finish the post. I just kinda wanted to get something out, so here it is, unpolished and ugly.

{{< comments >}}

## Appendix
### The RNG tangent
Just to jog your memory here is the line of code we were focusing on:
```rust
let mut rng = rand::rng();
```

The RNG generator samples the operating systems source of randomness (in my case `/dev/urandom`) at initialisation and it does this using the `OsRng` struct.
It then uses this $32$ byte sample as a seed for a cryptographically secure pseudo-random generator (CSPRNG).

CSPRNG is quite the loaded term so lets explore it a bit more.
The first sub-term (word?) is generator. `rng` is a generator, we can sample it and it returns values.
Next we have pseudo-random, this simply means that we have a seed - a string of random numbers - and we repeatedly apply functions like `XOR` or bit shifts for example, to the seed.
Then, an output number is generated from the seed. One way this could be done is just selecting a 32 bit area in the number output by the functions.
To generate another number, the same functions are applied to the output of the previous application of the functions and another number is taken from the output of this application of the functions.
Finally, what makes a pseudo-random generator cryptographically secure?
This means that the next number in a series of generated numbers cannot be reasonably guessed by considering the previously generated numbers.

It is worth noting that while our CSPRNG is cryptographically secure, it is still deterministic.
Meaning given the seed we can say with $100\%$ accuracy what the sequence of numbers will be.

