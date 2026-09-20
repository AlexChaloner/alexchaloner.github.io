---
layout: blog-post
title: "The State-Prediction Separation Hypothesis"
date: 2026-07-26
published: true
listed: false
original_tweet: https://x.com/alex_chaloner/status/2081446385547710499
---

# {{ page.title }}

Copied verbatim from [my Twitter thread](https://x.com/alex_chaloner/status/2081446385547710499) (26 July 2026).

<link rel="stylesheet" href="{{ '/assets/twitter-crosspost.css' | relative_url }}">

<div class="twitter-crosspost">
<section class="tweet" id="tweet-2081446385547710499">
<div class="tweet-text">"The State-Prediction Separation Hypothesis"

Authors here note that complex LLMs are not just doing next-token prediction, but also maintaining an internal state.

They find by explicitly separating these functions (simple masking) they can get same performance for cheaper!</div>
<div class="tweet-media">
<a href="{{ '/assets/twitter/2081446385547710499-1.jpg' | relative_url }}"><img src="{{ '/assets/twitter/2081446385547710499-1.jpg' | relative_url }}" alt="Image attached to the original tweet" width="892" height="1281" loading="lazy"></a>
</div>
</section>

<section class="tweet" id="tweet-2081446513050374203">
<div class="tweet-text">Full paper here:

<a href="https://arxiv.org/abs/2607.01218">https://arxiv.org/abs/2607.01218</a></div>
</section>

<section class="tweet" id="tweet-2081446674900361716">
<div class="tweet-text">Masking mechanism is simple:

Every token gets a parallel "predict token" inserted with it.

A sliding window includes the last K predict tokens, along with the usual full history of real tokens.

Next token prediction is only done on the predict tokens.</div>
<div class="tweet-media">
<a href="{{ '/assets/twitter/2081446674900361716-1.jpg' | relative_url }}"><img src="{{ '/assets/twitter/2081446674900361716-1.jpg' | relative_url }}" alt="Image attached to the original tweet" width="953" height="342" loading="lazy"></a>
</div>
</section>

<section class="tweet" id="tweet-2081446810095091854">
<div class="tweet-text">There's an inverse way of looking at this: at every token, the model gets a "thinking token".

This is the opposite end to having all the "chain of thought" at the end of the input but before the output.

An interesting followup idea is: can we get the same result with having thinking space every N tokens? (The algorithm here would need nontrivial adjustments). In some ways, this could look very similar to the Token Superposition paper.</div>
</section>

<section class="tweet" id="tweet-2081447106339127667">
<div class="tweet-text">(Token superposition paper:)</div>
<p class="tweet-quote"><a href="https://x.com/alex_chaloner/status/2069155536436670667">Quoted tweet by @alex_chaloner</a></p>
</section>

<section class="tweet" id="tweet-2081447506890662237">
<div class="tweet-text">N.B. the headline picture of "2.6x more token efficient" is taken from *before* learning rate decay phase (linear decay at the last 10% of steps).

There is also a small ~7% inference cost, at least in their implementation

Results are still a good and significant gain!</div>
<div class="tweet-media">
<a href="{{ '/assets/twitter/2081447506890662237-1.jpg' | relative_url }}"><img src="{{ '/assets/twitter/2081447506890662237-1.jpg' | relative_url }}" alt="Image attached to the original tweet" width="1349" height="804" loading="lazy"></a>
</div>
</section>

<section class="tweet" id="tweet-2081447733320126780">
<div class="tweet-text">Beyond the paper: in the age of Reinforcement Learning --
Doesn't this remind us of actor-critic architecture?

One part of the model tracks the state; the other handles the action.

I wonder if this will enable e.g. easy+cheap PPO implementation.</div>
<div class="tweet-media">
<a href="{{ '/assets/twitter/2081447733320126780-1.jpg' | relative_url }}"><img src="{{ '/assets/twitter/2081447733320126780-1.jpg' | relative_url }}" alt="Image attached to the original tweet" width="515" height="388" loading="lazy"></a>
</div>
</section>

<section class="tweet" id="tweet-2081447828623183876">
<div class="tweet-text">This is a really well-written, easy-to-read paper!&#32;
It's my favourite kind of paper: we had a simple intuitive insight, we found a way to implement it, and it gives us multiplicative gains!

Very well done to the authors 😊</div>
</section>
</div>
