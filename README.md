## Grove

This is a Rust implementation of the GloVe (Global Vectors for Word Representation) model
for learning word vector representations. 

### GloVe

See 
the [GloVe project page](http://nlp.stanford.edu/projects/glove/) or 
the [original paper](http://nlp.stanford.edu/pubs/glove.pdf) 
for more information on glove vectors.

The [original GloVe implementation](https://github.com/stanfordnlp/GloVe/tree/master) is written 
in simple, concise, and beautiful C code. 
It contains an elegant solution to the memory vs. speed trade-off of the GloVe preprocessing algorithm,
and has great scaling properties.

### Why the reimplementation

I chose to implement Grove = Glove + Rust as a perfect way of learning Rust,
which is often compared to C in both speed and elegance.

