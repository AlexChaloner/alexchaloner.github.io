---
layout: blog-post
title: "Recirculation and Full Bandwidth Transformer"
date: 2026-08-20
published: true
listed: false
original_tweet: https://x.com/alex_chaloner/status/2090438759263416328
---

# {{ page.title }}

Copied verbatim from [my Twitter thread](https://x.com/alex_chaloner/status/2090438759263416328) (20 August 2026).

<link rel="stylesheet" href="{{ '/assets/twitter-crosspost.css' | relative_url }}">

<div class="twitter-crosspost">
<section class="tweet" id="tweet-2090438759263416328">
<div class="tweet-text">What's the difference between Recirculation and Full Bandwidth Transformer?

They are very similar!

The similarities: Both:
- feed back a deep latent from previous token to the next
- can be trained with expensive sequential prefill stage

Now the differences:</div>
<p class="tweet-quote"><a href="https://x.com/mc_mozer/status/2090211279462117796">Quoted tweet by @mc_mozer</a></p>
</section>

<section class="tweet" id="tweet-2090438771988893993">
<div class="tweet-text">1. Recirculation can be performed *without any training at all*

However it performs better with training,  or 'adaptation'.</div>
<div class="tweet-media">
<a href="{{ '/assets/twitter/2090438771988893993-1.jpg' | relative_url }}"><img src="{{ '/assets/twitter/2090438771988893993-1.jpg' | relative_url }}" alt="Image attached to the original tweet" width="1800" height="683" loading="lazy"></a>
</div>
</section>

<section class="tweet" id="tweet-2090438789147828251">
<div class="tweet-text">2. Recirculation studies the latent space passing from layer N on token T --&gt; layer M &lt; N on token T+1.

Full-Bandwidth focusses on final layer passed in to next token

(Left: Recirc, Right: Full-Bandwidth)</div>
<div class="tweet-media">
<a href="{{ '/assets/twitter/2090438789147828251-1.jpg' | relative_url }}"><img src="{{ '/assets/twitter/2090438789147828251-1.jpg' | relative_url }}" alt="Image attached to the original tweet" width="1080" height="679" loading="lazy"></a>
<a href="{{ '/assets/twitter/2090438789147828251-2.jpg' | relative_url }}"><img src="{{ '/assets/twitter/2090438789147828251-2.jpg' | relative_url }}" alt="Image attached to the original tweet" width="388" height="364" loading="lazy"></a>
</div>
</section>

<section class="tweet" id="tweet-2090438794407444761">
<div class="tweet-text">3. For training, Recirculation prefill training was thought expensive.

Full Bandwidth Transformers found the latent space is trained quickly and cheaply via a 2- and 3-stage partial prefill.

Obviously we need to try combining these!</div>
</section>

<section class="tweet" id="tweet-2090438813562839282">
<div class="tweet-text">4. Full Bandwidth Transformer is opinionated about the layer combinations, providing a single function with a strong performance.

Recirculation studies flexible family of combinations, introducing new hyperparameters.

(Left: Recirculation, Right: Full Bandwidth)</div>
<div class="tweet-media">
<a href="{{ '/assets/twitter/2090438813562839282-1.jpg' | relative_url }}"><img src="{{ '/assets/twitter/2090438813562839282-1.jpg' | relative_url }}" alt="Image attached to the original tweet" width="1004" height="560" loading="lazy"></a>
<a href="{{ '/assets/twitter/2090438813562839282-2.jpg' | relative_url }}"><img src="{{ '/assets/twitter/2090438813562839282-2.jpg' | relative_url }}" alt="Image attached to the original tweet" width="1080" height="553" loading="lazy"></a>
</div>
</section>

<section class="tweet" id="tweet-2090438819384619171">
<div class="tweet-text">It's always interesting when two separate authors land upon the same key idea at the same time!

Both add key insights that will multiply the other's effectiveness. This is exciting!!

Full Bandwidth Transformer:
<a href="https://arxiv.org/abs/2608.08888">https://arxiv.org/abs/2608.08888</a>

Recirculation:&#32;
<a href="https://arxiv.org/abs/2608.17981">https://arxiv.org/abs/2608.17981</a></div>
</section>

<section class="tweet" id="tweet-2090438825705422949">
<div class="tweet-text">Apologies as I may have missed some things - I am on holiday!

Please feel free to fill in gaps and correct me!</div>
</section>
</div>
