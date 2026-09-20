---
layout: blog-post
title: "Paper Summary: Non-transitive preferences"
date: 2026-08-05
published: true
listed: true
original_tweet: https://x.com/alex_chaloner/status/2084994163406192934
---

# {{ page.title }}

Copied verbatim from [my Twitter thread](https://x.com/alex_chaloner/status/2084994163406192934) (5 August 2026).

<link rel="stylesheet" href="{{ '/assets/twitter-crosspost.css' | relative_url }}">

<div class="twitter-crosspost">
<section class="tweet" id="tweet-2084994163406192934">
<div class="tweet-text">Yesterday I learned about non-transitive preferences, which I found very non-intuitive.

You might assume that, if A &gt; B and B &gt; C, then necessarily A &gt; C.
This is transitivity, and it's intuitive.

However, sometimes, C &gt; A. This is the case in non-transitive preference.

1/</div>
</section>

<section class="tweet" id="tweet-2084994165788598391">
<div class="tweet-text">To credit, I lifted a lot of this info from this great page of a series by Peter Fishburn: <a href="https://www.eecs.uottawa.ca/ordal/papers/fishburn/node9.html">https://www.eecs.uottawa.ca/ordal/papers/fishburn/node9.html</a></div>
</section>

<section class="tweet" id="tweet-2084994797844955330">
<div class="tweet-text">First, how can this happen?

A simple example we know is Rock-Paper-Scissors (illustrated)

Ok, you say: but this is a game, not real life. Surely all *my* preferences are transitive?</div>
<div class="tweet-media">
<a href="{{ '/assets/twitter/2084994797844955330-1.jpg' | relative_url }}"><img src="{{ '/assets/twitter/2084994797844955330-1.jpg' | relative_url }}" alt="Image attached to the original tweet" width="460" height="440" loading="lazy"></a>
</div>
</section>

<section class="tweet" id="tweet-2084995086782177290">
<div class="tweet-text">Now consider yourself in the morning and in the evening.

Maybe in the morning, breakfast &gt; beer. But in the evening, beer &gt; breakfast.

Ah, you say, there's a hidden variable there, the time of day!

there are hidden variables everywhere for those with the eyes to see</div>
</section>

<section class="tweet" id="tweet-2084995246274773390">
<div class="tweet-text">Ok, then let's see about people in a simple study check out this except from the aforementioned page!

~1/3 college students (presumably intelligent, reasonable) had nontransitive preferences for a simple choice!

Did they make a mistake? Maybe. But it still appears in the data.</div>
<div class="tweet-media">
<a href="{{ '/assets/twitter/2084995246274773390-1.jpg' | relative_url }}"><img src="{{ '/assets/twitter/2084995246274773390-1.jpg' | relative_url }}" alt="Image attached to the original tweet" width="2085" height="422" loading="lazy"></a>
</div>
</section>

<section class="tweet" id="tweet-2084995362884882710">
<div class="tweet-text">Moreover, even if you allow an individual to be perfectly rational, you will still see nontransitive relations in a population.

Voter 1 prefers A &gt; B &gt; C
Voter 2 prefers B &gt; C &gt; A
Voter 3 prefers C &gt; A &gt; B

If you take an average, you'll get A &gt; B &gt; C &gt; A.</div>
</section>

<section class="tweet" id="tweet-2084995567684329712">
<div class="tweet-text">Why is all of this important?

RLHF often has implicit assumption of preference transitivity. Some people have found improvements if you allow for non-transitivity.

See papers:
<a href="https://arxiv.org/abs/2410.02197">https://arxiv.org/abs/2410.02197</a>
<a href="https://arxiv.org/abs/2605.18721">https://arxiv.org/abs/2605.18721</a></div>
</section>

<section class="tweet" id="tweet-2084995716804448490">
<div class="tweet-text">I found this really interesting because I had no idea it existed before :)

Hidden assumptions always stab us in the knee. Bitter lesson, etc.

Entirely unrelated to RL: discussions about ELO ratings for games with nontransitive strategies!
<a href="https://arxiv.org/abs/2206.12301">https://arxiv.org/abs/2206.12301</a></div>
</section>

<section class="tweet" id="tweet-2084996502728974535">
<div class="tweet-text">✨️ Nontransitive preferences ✨️</div>
<div class="tweet-media">
<a href="{{ '/assets/twitter/2084996502728974535-1.jpg' | relative_url }}"><img src="{{ '/assets/twitter/2084996502728974535-1.jpg' | relative_url }}" alt="Image attached to the original tweet" width="960" height="540" loading="lazy"></a>
</div>
</section>
</div>
