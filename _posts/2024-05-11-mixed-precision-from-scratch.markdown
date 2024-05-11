---
layout: post
comments: true
title:  'Mixed Precision Training from Scratch'
date:   2024-05-11 00:00:00 -0100
categories: jekyll update
---
<style>
{% include blogposts.css %}
</style>

To really understand the whole picture of Mixed Precision Training, you need to go down to the CUDA level. Staying in 
Pytorch land can only get you so far:

```python
# Just because you can import, doesn't mean you understand.
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
for input, target in data:
    
    # What is happening here?
    with autocast(device_type='cuda', dtype=torch.float16):
        output = model(input) 
        loss = loss_fn(output, target)

    # ...and here?
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```
While `torch.cuda.amp` offers a seamless way to apply mixed precision training,
it also hides away the most important details. Especially how it makes your model run faster. 
The answer, as the library's name suggests, lies in CUDA. 

I decided to rewrite my [own](https://github.com/tspeterkim/mixed-precision-from-scratch) 
from scratch, going down to the CUDA level, and write this guide. 
I hope this will help you really understand mixed precision training.

## What is Mixed Precision Training?

Mixed precision training is training neural nets with 
[half-precision floats](https://en.wikipedia.org/wiki/Half-precision_floating-point_format) (FP16).
This means all of our model weights, activations, and gradients during our forward and backward pass are now
FP16, instead of single-precision floats (FP32). Now our memory requirement is halved! Unfortunately, just doing this 
causes issues with our gradient updates:
```python
for weight, grad in zip(weights, grads):
    update = -learning_rate * grad
    weight.data += update
```
Since `grad` is now FP16, it can become too small to be represented in FP16 
(any value smaller than 2^-24). When the gradient is less than 2^-24, it underflows to 0 and makes `update` = 0, causing 
our model to stop learning.

Remember, we also made `weight` FP16 too. So even if we somehow preserved `update` to express the too-small-for-FP16 
values, adding it to an FP16 tensor like `weight` will cause no change in our weight values.

FP16 sacrifices numerical stability (fancy word for over/underflowing easily) for reduced memory footprint. So how does
Mixed Precision Training overcome this trade-off? As the [original paper](https://arxiv.org/pdf/1710.03740) outlines, 
it offers three solutions.

<img src="/images/mp-scratch/histogram.png" width="800" style="margin: 0 auto; display: block; "/>


{% include disqus.html %}