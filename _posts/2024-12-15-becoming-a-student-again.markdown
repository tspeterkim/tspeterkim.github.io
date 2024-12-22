---
layout: post
comments: true
title:  'I came back to school to study hardware after 5 years of doing ML'
date:   2024-12-15 00:00:00 -0100
categories: jekyll update
---
<style>
{% include blogposts.css %}
</style>

## Background

After 5 years of doing ML engineering, I decided to pivot into ML systems.

Building video recommendation systems was fun, and it was always interesting to see real-world metrics like playback time go up as a result of a better predictive model. Especially when I could see my own video recommendations improve.

However, after getting my hands dirty with some CUDA [side](https://github.com/tspeterkim/flash-attention-minimal) [projects](https://tspeterkim.github.io/posts/cuda-1brc), I was drawn to something more lower-level, and frankly, more concrete than ML:

<img src="/images/student-again/xkcd-ml.png" width="300" style="margin: 0 auto; display: block; "/>
<center><p style="color: #555;font-size: 14px;">Stirring the pile became boring after 5 years.</p></center>

I started to self-teach myself more CUDA, Triton, Pytorch Internals, ... basically anything related to ML systems and compilers.
One thing became crystal clear after doing so: all roads lead to hardware.

<b>To squeeze the most performance out of my AI accelerator, I needed to understand my hardware's microarchitecture.</b>

So I came back to school to study computer engineering and build up my intuition from the hardware level.

I just finished my first semester as a Master's student in the ECE department at UIUC. And boy did I get close to the hardware.

This blog is about the jarring experience of jumping from Python to SystemVerilog, and general things I've felt being a student again.

So what does school feel like the second time around?

<img src="/images/student-again/basically-me.png" width="400" style="margin: 0 auto; display: block; "/>
<center><p style="color: #555;font-size: 14px;">How I feel at times.</p></center>

## Lessons Learned 

### Pain is good

In line with my reason returning back to school, I enrolled in "Computer Organization and Design", numbered as ECE 411.
Little did I know, I had bit off more than I could chew.

Over the semester, we built a cache, a pipelined processor (w/ forwarding), and an out-of-order RISC-V processor, using a [HDL](https://en.wikipedia.org/wiki/Hardware_description_language) known as SystemVerilog.

"Pipelining, ey? That doesn't sound too bad. It's just this, right?"

<img src="/images/student-again/pipeline-is-ez.png" width="450" style="margin: 0 auto; display: block; "/>
<center><p style="color: #555;font-size: 14px;">The laundry analogy. If only it was this simple.</p></center>

It is conceptually. But when you're writing the code and debugging it, it looks more like this:

<img src="/images/student-again/verdi-is-hard.png" width="1000" style="margin: 0 auto; display: block; "/>
<center><p style="color: #555;font-size: 14px;">You haven't felt pain until you've stared at waveforms in Verdi.</p></center>

For someone who'd been writing Python for the past 5 years, this was damn near impossible. 
However, through sheer will and the help from excellent peers and TAs, I survived. It was excruciating though. I would spend literal weekends doing the assignments, only to make marginal progress. <b>But the truth is: I learnt as much as I suffered.</b> 
I can now confidently say that I know how out-of-order processors, caches, memory, (and more!) all actually work at the bare hardware level. I would not have been able to get to this point without going through the gauntlet of these brutal assignments.

<img src="/images/student-again/matrix.png" width="300" style="margin: 0 auto; display: block; "/>
<center><p style="color: #555;font-size: 14px;">My POV at the end of the semester after taking ECE 411. It really is all zeros and ones.</p></center>

This coincided with my experience in the industry too. The most stressful moments, whether it was a challenging project or a heated argument with a colleage, were the ones that made me into a better engineer and communicator. 
So yeah, pain is good. And I hope to feel more of it during my time here as a student.

### Group projects where you don't like your group - that's life

Even as an undergrad, I always knew that the point of these group projects were to prepare us for the industry, where we would be working as a team with others. After being in the industry, I can confirm this is true. However, I was wrong about what kind of a group I would be working in. 
Putting it bluntly, <b>the experiences of working in a group where you don't like your group is the most accurate reflection of working in the industry.</b>

You're not going to dislike everyone, of course. But there will always be a few people who are going to be tough to work with due to their working and communication style or just plain vibes (engineers are humans too). 

In the industry, it actually gets worse. Imagine if one of those few people you don't work well with is actually your manager. And at least right now, when the semester's done, the group disbands and you never have to see them again. But imagine if the group didn't. Imagine if you have to work with them for years. Finally, in the setting of a school, you can "report" them to your TA or Professor. In the real world, no one will be there help you (they will all be busy dealing with their own dysfunctional group).

This semester, whenever I faced difficult-to-work-with team members, I was surprised how unphased I was compared to how I would've felt as an undergrad. Primarily because I've faced worse in the industry for years. Now before you go off thinking that the industry is a place where everyone hates each other and never get any work done, it's not. 
<b>It's a place where everyone hates each other but still get amazing work done.</b> My time in the industry has taught me this skill and it's the point of these groups projects: not just to work well with people but to work well with people you don't like.

<img src="/images/student-again/coworker.jpg" width="300" style="margin: 0 auto; display: block; "/>
<center><p style="color: #555;font-size: 14px;">If you know, you know.</p></center>

### Education should take priority over grades

In class, your success depends on a predefined set of checkboxes. As long as you tick those boxes of doing well in assignments and exams, you are deterministically guaranteed to succeed. But what is success here? Is it getting a good grade? Or is it actually learning the material?

When I was being bombarded with assignments and exams from all three of my classes this semester, I realized how quickly I started to think success was about just completing these tasks. I had to constantly remind myself that doing the task meant nothing if I hadn't learned something out of it. 
<b>Ironically, education's fundamental point gets drowned out in the sea of deadlines that colleges set for their students.</b>

For one of my notoriously hard assignments this semester, I was struggling to pass all test cases. Through my own effort, I had just passed 70% of them. I should've have been more proud that I achieved that number on my own merit. 
But all I really felt was an overwhelming sense of doom that I would probably not have enough time to get the rest of the test cases, considering that the deadline was in less than 24 hours. 

Like the devil, a peer approached me that he was willing to share his test code that helped him get 100%. It was tempting. But eventually, I realized that using that code to find my bugs would deprive me of the learning opportunity for me to struggle and discover the solution on my own. I politely declined. I chose my education over getting a good grade. I did ask him for high-level hints though :)

<img src="/images/student-again/staying-fine.png" width="300" style="margin: 0 auto; display: block; "/>
<center><p style="color: #555;font-size: 14px;">As long as you're learning, it really is fine. Even if your grades don't say so.</p></center>

## Looking forward

Reflecting on this past semester, I do not regret coming back to school to study hardware. I've learned so much about computer architecture and now I'm excited to build up from here, developing a first-principles understanding of how to make GPUs go brrr.

Interestingly, I think I will get the most out of school the second time around because of my experience in the industry. I know I shouldn't be stressed about meeting deadlines and getting good grades as long as I'm continously learning. The only true stress comes from working with other people in group projects and that's natural. The industry has well prepared me for this.

### Special thanks to my wife

Thank you for always being there to support me through the many stressful nights I went through this semester. Because of you, I'm always reminded that I am really doing this for creating a better life with you, and making as many happy memories along the way.

{% include disqus.html %}