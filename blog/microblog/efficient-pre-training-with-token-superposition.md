---
layout: blog-post
title: "Paper Summary: Efficient Pre-Training with Token Superposition"
date: 2026-06-22
published: true
listed: true
original_tweet: https://x.com/alex_chaloner/status/2069155536436670667
---

# {{ page.title }}

Copied verbatim from [my Twitter thread](https://x.com/alex_chaloner/status/2069155536436670667) (22 June 2026).

<link rel="stylesheet" href="{{ '/assets/twitter-crosspost.css' | relative_url }}">

<div class="twitter-crosspost">
<section class="tweet" id="tweet-2069155536436670667">
<div class="tweet-text">"Efficient Pre-Training with Token Superposition"

This paper finds that we can group up tokens to create a much more efficient training regime, achieving 2-4x speedups for the same loss.</div>
<div class="tweet-media">
<a href="{{ '/assets/twitter/2069155536436670667-1.jpg' | relative_url }}"><img src="{{ '/assets/twitter/2069155536436670667-1.jpg' | relative_url }}" alt="Image attached to the original tweet" width="938" height="1228" loading="lazy"></a>
</div>
</section>

<section class="tweet" id="tweet-2069155539351646550">
<div class="tweet-text">Paper can be found here: <a href="https://arxiv.org/abs/2605.06546">https://arxiv.org/abs/2605.06546</a>

Great blog post to pair with it, interactive diagrams: <a href="https://nousresearch.com/token-superposition">https://nousresearch.com/token-superposition</a></div>
<div class="tweet-media">
<a href="{{ '/assets/twitter/2069155539351646550-1.jpg' | relative_url }}"><img src="{{ '/assets/twitter/2069155539351646550-1.jpg' | relative_url }}" alt="Image attached to the original tweet" width="940" height="805" loading="lazy"></a>
</div>
</section>

<section class="tweet" id="tweet-2069155542212202738">
<div class="tweet-text">The paper groups up tokens into bags of size s, and predicts the next bag.

They call this Superposition (great naming).

Given its own intuitive CE loss.</div>
<div class="tweet-media">
<a href="{{ '/assets/twitter/2069155542212202738-1.png' | relative_url }}"><img src="{{ '/assets/twitter/2069155542212202738-1.png' | relative_url }}" alt="Image attached to the original tweet" width="615" height="111" loading="lazy"></a>
</div>
</section>

<section class="tweet" id="tweet-2069155544821121476">
<div class="tweet-text">For the first 30% of training we follow this method of next-token-bag prediction.

Then, they switch to normal token prediction! Brave! But look at that loss curve fall!

This is necessary because the bags of tokens are unordered.</div>
<div class="tweet-media">
<a href="{{ '/assets/twitter/2069155544821121476-1.png' | relative_url }}"><img src="{{ '/assets/twitter/2069155544821121476-1.png' | relative_url }}" alt="Image attached to the original tweet" width="566" height="620" loading="lazy"></a>
</div>
</section>

<section class="tweet" id="tweet-2069155548315234713">
<div class="tweet-text">I'm really a fan of how well-written this paper is. It's easy to follow, with great diagrams and very clear results.

Full details of GPUs, GPU-hours, hyperparameters sweeps, even code in the appendix (!) allows reproduction.&#32;

All too many papers miss this! But we need it for science!</div>
<div class="tweet-media">
<a href="{{ '/assets/twitter/2069155548315234713-1.jpg' | relative_url }}"><img src="{{ '/assets/twitter/2069155548315234713-1.jpg' | relative_url }}" alt="Image attached to the original tweet" width="1277" height="506" loading="lazy"></a>
</div>
</section>

<section class="tweet" id="tweet-2069155550873448830">
<div class="tweet-text">(This was especially refreshing after reading the GRAM paper which was terrible in these aspects)</div>
</section>

<section class="tweet" id="tweet-2069155553331318899">
<div class="tweet-text">And we finally have this really interesting breakdown, discussion and ablation study of all the moving parts in this paper.&#32;

Really fantastic stuff.</div>
<div class="tweet-media">
<a href="{{ '/assets/twitter/2069155553331318899-1.png' | relative_url }}"><img src="{{ '/assets/twitter/2069155553331318899-1.png' | relative_url }}" alt="Image attached to the original tweet" width="864" height="573" loading="lazy"></a>
</div>
</section>

<section class="tweet" id="tweet-2069155556594528574">
<div class="tweet-text">Riffing on my own here, you can start to see a general, hierarchical method reveal itself.

First - we had BPE on top of bytes to make tokens.
Now bags of tokens help performance.
Hierarchy seems to help.

I wonder how it pairs with token diffusion?

(Shitty diagram for visual.)</div>
<div class="tweet-media">
<a href="{{ '/assets/twitter/2069155556594528574-1.png' | relative_url }}"><img src="{{ '/assets/twitter/2069155556594528574-1.png' | relative_url }}" alt="Image attached to the original tweet" width="632" height="172" loading="lazy"></a>
</div>
</section>

<section class="tweet" id="tweet-2069155558574194908">
<div class="tweet-text">Overall big fan of this from @NousResearch.
Great communication and rigorous research. Will be reading more papers + following more closely.&#32;
Well done!</div>
</section>
</div>
