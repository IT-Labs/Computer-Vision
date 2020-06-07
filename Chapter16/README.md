## 1. Learning Rate Schedulers
By adjusting our learning rate on an epoch-to-epoch basis, we can reduce loss, increase accuracy, and even in certain
situtations reduce the total amount of time it takes to train a network.

#### 1.1 Dropping our learning rate
The most simple and heavily used learning rate schedulers are ones that progressively reduce learning rate over time.

```W += -alpha * gradient```

The learning rate `alpha` controls the 'step' we make along the gradient. Larger values of alpha imply that we are
 taking bigger steps, while smaller values of alpha will make tiny steps.
 When training our network, we are trying to find some location along our loss landscape where the network obtains
 reasonable accuracy.

 If we consistently keep a learning rate high, we could overshoot these areas of low loss as we'll be taking too large
 of steps to descend into these areas. Instead, what we can do is decrease our learning rate, thereby allowing our
 network to take smaller steps - this decreased rate enables our network to descend into areas of the loss landscape
 that are 'more optimal' and would have otherwise been missed by our larger learning rate.
##### 1.1.1 The standard Decay Schedule in Keras
The `keras` library ships with a time-based learning rate scheduler - it is controlled via the `decay` parameter of the
optimizer classes (such as SGD).

```SGD(lr=0.01, decay=0.1/40, momentum ...)```

Internally `keras` applies the following learning rate schedule to adjust the learning rate after
_**every batch update**_ :

```lr = init_lr * (1.0 / (1.0 + decay * iterations))```

##### 1.1.2 Step-based Decay
Another popular learning rate scheduler is step-based decay where we systematically drop the learning rate after
specific epochs during training.

When applying step decay, we often drop our learning rate by either (1) half or (2) an order of magnitude after every
fixed number of epochs.
##### 1.2 Summary
Two primary types of learning rate schedulers:

1. Time-based schedulers that gradually decrease based on epoch number.
2. Drop-based schedulers that drop on a specific epoch, similar to the behaviour of a piecewise function.

Exactly which scheduler should we use is part of experimentation process. Typically, your first expieriment would not
use any type of decay or learning rate scheduling so you can obtain a baseline accuracy and loss/accuracy curve. 