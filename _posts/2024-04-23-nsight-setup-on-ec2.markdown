---
layout: post
comments: true
title:  'How to set up Nsight Compute Locally to profile Remote GPUs'
date:   2024-04-22 01:31:24 -0100
categories: jekyll update
---
<style>
{% include blogposts.css %}
</style>

Have a remote GPU instance? Want to see some rooflines with Nsight Compute? This tutorial is for you.

#### 0. Set up a remote GPU instance.

Step 0 because I assume you've already done this part (using any cloud provider - I'm going with AWS).
**You just need an SSH-able machine with CUDA installed.**

If you are using AWS EC2 like me:

I'm using an Ubuntu image like this on a g4dn.xlarge:

<img src="/images/nsight-ec2/ubuntu-image.png" width="800" style="margin: 0 auto; display: block; "/>

Personally recommend using Ubuntu over Amazon Linux. I couldn't get `nvcc` to work on Amazon Linux. 
Plus when you run into issues, you'll find more Ubuntu-related stack overflow posts.

#### 1. Download Nsight Compute on your local machine.

Download [link](https://developer.nvidia.com/tools-overview/nsight-compute/get-started). Nvidia developer account
required.

#### 2. Set up remote process.

* Click **Connect** and add a new remote connection. Use your remote machine's config.
<img src="/images/nsight-ec2/remote.png" width="800" style="margin: 0 auto; display: block; "/>

* Select the CUDA application you want to profile (the binary executable created with `nvcc` 
e.g. `bench` in `nvcc bench.cu -o bench`).
* Select the output file on your local machine you want to dump the profiler's result to
(I had to manually create a dummy file called `out`).
<img src="/images/nsight-ec2/bench-init.png" width="800" style="margin: 0 auto; display: block; "/>
* Click **Launch**

---
<br>
**If this works for you, then you're done. Enjoy your rooflines!**

If not, and you get a permissions error like this, go to Step 3.

```
Error: ERR_NVGPUCTRPERM - The user does not have permission to access 
NVIDIA GPU Performance Counters on the target device 0. 
For instructions on enabling permissions and 
to get more information see https://developer.nvidia.com/ERR_NVGPUCTRPERM
```

---
<br>
#### 3. Give profiling permission to all users on your remote instance.

* On your EC2 instance, run `sudo vim /etc/modprobe.d/nvidia.conf`. Add the following line:

```commandline
options nvidia NVreg_RestrictProfilingToAdminUsers=0
```
* Reboot your instance.
* Run `sudo update-initramfs -u -k all`

Check out the [official guide](https://developer.nvidia.com/ERR_NVGPUCTRPERM) for more context.

_NOTE for EC2 users: When you reboot your instance, AWS will assign a different public IP to your instance. Make sure to update it 
in your remote connection setting from Step 2._

#### 4. Launch again from Nsight Compute.

Cross your fingers.

#### 5. Enjoy your Rooflines.
<img src="/images/nsight-ec2/final.png" width="800" style="margin: 0 auto; display: block; "/>

{% include disqus.html %}