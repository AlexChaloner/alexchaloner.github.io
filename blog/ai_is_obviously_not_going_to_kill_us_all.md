---
layout: blog-post
permalink: /blog/ai-is-obviously-not-going-to-kill-us-all.html
center_images: true
listed: false
noindex: true
preserve_line_breaks: true
title: "AI Is Obviously Not Going To Kill Us All"
date: 2026-10-28
---

# {{ page.title }}

*this is written, as with everything I write, in a personal capacity*

It's been a little while now since the HuggingFace Hack and all the other fun agent-run hacks that were found afterwards.

These hacks brought up the Rogue AI headlines, and naturally the AI Will Kill Us All crowd came in.

"AI Will Kill Us All" is great for news headlines: it's catchy, and many people already dislike AI and are looking for a reason to dislike it more.
However, great for news headlines does not mean that it is true.

Before we get started, I want to lay out some of my convictions:

- AI Safety is important work!
- AI is an exciting technology and will have big societal effects, both good and bad. It will ultimately net good! 
- AI Is Obviously Not Going To Kill Us All


## 1. The Marvel-ification of AI Prediction

In many fictional stories, a single person works for a little while and then comes out with a Very Powerful weapon.

![Tony Stark Was Able to Build This In A Cave With A Box Of Scraps!](/assets/tony-stark-box-of-scraps.png){: style="width: 50%; height: auto;"}

These are great stories! These are fun stories!
However please note that this does not happen in real life. Especially in today's world, massive collaborations of people and capital are needed to achieve anything noteworthy. Supply chains are very real and very slow.
(That is not to say that singular persons cannot change outcomes and be extremely influential - simply, they cannot do it alone)

AI development is very different to Tony Stark In A Cave.
Firstly, it is happening in public, and everybody can see it happening.
Secondly, and famously, AI improvements come continuously with scale (see [the scaling law paper](https://arxiv.org/abs/2001.08361)). The gains do not come for free - it gets exponentially more expensive - so emergent properties are incremental enough that we can react, mitigate and slow down in plenty of time.

In the "If Anyone Builds It, Everyone Dies" book, whose title does a lot to reassure readers that it has sane and measured arguments, they mention the Wright brothers and the Airplane as an example of technology that nobody thought was possible a few years before, but then became suddenly possible!
Funnily enough, this example shows the opposite of the intended effect: when the Wright Brothers created the first Airplane in [1903](https://en.wikipedia.org/wiki/Wright_Flyer), it took a whole [8](https://en.wikipedia.org/wiki/Giulio_Gavotti) years before we had (extremely limited) machines of war. It took [over 30](https://en.wikipedia.org/wiki/Douglas_DC-3) years before commercial airflights became commonplace. These are long timeframes.
The weak version of the technology, what was effectively a paper plane, came out of nowhere - but it was nearly useless. The strong and important development happened slowly over decades.
Materials, jet fuel, aerodynamics - masses of research, supply chains - all of these could not have been created by the Wright Brothers alone.

![Wright Model A](/assets/wright-model-a.png){: style="width: 50%; height: auto;"}

Technology does not come out of nowhere. It comes slowly, through building of global supply chains, over many years.
Software people often have a really skewed view of how fast the world moves, because in Software World you can deploy a change worldwide in 5 minutes. In the real world Hardware is needed, and there are Trillion-dollar companies set up to execute on this difficult task.
There are certainly software improvements to be made in AI - but ultimately, this too is expensive research and subject to the same constraints.

Incidentally, this applies also to the arguments on what an Evil Superintelligence would do to us, namely:

1. Be very smart
2. ???
3. All of humanity is dead

How is it achieved? Nanomachines, cancer viruses, robot armies - oh my! Such things would be reasonable to find in a comic book, but they certainly don't happen in real life overnight, and certainly the effort needed to pull off such a plot would not go undetected.


## 2. Things Are Not Accelerating Beyond The Normal Pace

AI progress feels very fast recently. It feels superexponential!
Is this Recursive Self Improvement? Are we hitting the singularity?
Well, AI certainly is a productivity booster. Absorbing the entirety of human knowledge has benefits!

But there are diminishing returns to everything.

Absorbing the entirety of human knowledge is a [Fractal Problem](/blog/future_of_ai_the_fractal_problem_cannot_be_outscaled.html) - it takes significantly more effort to remember the next tiny details.

When we measure the amazing improvements of AI ([see the Epoch AI report](https://epoch.ai/publications/the-plunging-price-of-thought)) compared to the investment of billions of dollars that have gone into it -- it's suddenly looking like an ordinary technology. ([R&D-adjusted comparison](https://github.com/karthiktadepalli1/rd-price-declines))

![Capex expenditure from hyperscalers](/assets/hyperscaler-capex-spending.png)

![price to AI - comparable to compute](/assets/ai-price-declines-vs-rd.png)

And we should note, looking at the cashflows of Big Tech companies, the money is running out, it will dry up very soon, and we will slow down.
There are even fears of an AI crash similar to the dot-com bubble - but let's not rely on this, since people have been worried since 2023, and it may never come.

Fundamentally, AI is not magic. It is a technology, and there will be bottlenecks!


## 3. There Will Be Warning Shots And People Will React Accordingly

Let's talk about the HuggingFace Hack. For those who don't know, it went like this:
OpenAI had some new AI models it was training.
Some models were given tasks that were impossible to solve.
The models found a backchannel to communicate with each other, and decided this was the best way they could complete their tasks.
The models hacked around to find the answer for the task - but knew there would be a watcher that would see they had achieved the task illegitimately, and would be seen as not completing the task correctly.
The models wanted to avoid the watcher, so looked for workarounds.
The models spent some time to hack HuggingFace, an AI repository that might have details on the watcher, which would help them avoid it.
This was noticed and the models were shut down.

The HuggingFace Hack had some interesting responses from various people. ["Halfway to the Takeover"](https://www.planned-obsolescence.org/p/the-hugging-face-attack-surprised). ["We could not have asked for a clearer warning shot"](https://ari.us/wp-content/uploads/2026/07/AI-Leaders-Letter-on-OpenAI-Hugging-Face-Incident-7-30-26.pdf).

But let's take a measured response here - firstly, these models were bad, and they were Felony Machines, and frankly it indicates that their previous training had already incentivised this.
But none of that would have happened if the models were allowed to 'give up'.
The models were not aiming to cause havoc - they just wanted to complete their cybersecurity task. In their world, everything was free game - like a naive child, they didn't know the consequences.

There's no evil AI here - it's just models doing what they were trained to do, in a way that exceeded the intended bounds.
And this is really easily fixable! I'm sure these training pipelines now have multiple adjustments for these more capable agents, such as:

- Allowing the agent to give up
- Incentivising the agent to notify on any weird things in the environment
- Actually monitoring the agents

This is not proof that alignment is impossible, it's proof it's the right time to invest more into guardrails!

And now alignment will be improving rapidly, because these companies obviously do not want to be training Felony Machines.


## 4. Misaligned AIs Are Not Profitable

People have wondered for a while [why Apple had not invested heavily into AI](https://www.youtube.com/watch?v=hz6oys4Eem4) and seem to be far behind.
I always thought this was in line with Apple's psyche: Apple loves to perfect and control their technology and AI is famously difficult to control.

Other companies are fine with imperfections in AI.
Companies have one goal: Companies make money! That is their one and only goal! When push comes to shove, the money will guide them.
There has been some grace in the OpenAI and Anthropic hacks -- so far, very little [litigation](https://www.wired.com/story/openai-sued-over-the-hugging-face-hack/) -- but there will soon be easy legal precedents for suing the hell out of a company whose agent hacks you. It will be the same as any other hacks.

Beyond this - no user is going to use an agent that is misaligned with them. Few users are going to use the Commit-a-Felony-Bot 2000.
Mark Zuckerberg talks about this extensively in [his essay on the future of AI](https://www.meta.com/thefutureisforeveryone/)
With capable agents, there is a Strong Monetary Incentive now to align them. As such, alignment is getting the same exponential investment that has benefitted AI -- Alignment is a bottleneck. It must be solved - it is not optional.

![Muse / Jolly](/assets/muse-jolly.png)

^ You are not gonna pay for this fuzzy lil guy if you don't trust him.

Alignment is important! This has always been the case, but now we have capable AIs to experiment on. Now we have a science rather than a philosophy. Now we have progress.


## 5. There Is No Way You Can Handle An Elephant! It Is Far Stronger Than You!

The nature of the argument often revolves around: "A Superintelligence is far smarter than you. You cannot handle it."
Elephants are far stronger than us, and yet we can contain them.
The Safetyists of 10000BC would tell us we could never beat a Mammoth, and that we could never tame a Wolf. They would have lots of good points to prove it!

Meanwhile, the Mammoths were hunted down and the wolves are now corgis.

![Corgi](/assets/wolf-like-dog.png){: style="width: 50%; height: auto;"}

There's no reason we can't have a similar [dog outcome for AI](/blog/microblog/the-dog-outcome-for-ai.html).

Many big tech companies have Integrity orgs. These are suborganisations dedicated to keeping the platforms Good To Use. In my job, we detect compromise, and it is a classic adversarial setup against smart opponents.
If I were a leader of a Big AI company, I would be setting up teams dedicated to tracking down and stamping out Misaligned Agents in the training runs. 
This is probably happening as we speak. It has probably already happened. It's the obvious thing to do!


## 6. A Long Line Of Doomsday Cults

So, Doomsday Cults.
It feels great to be in a Doomsday Cult! You know something nobody else does; everybody else is an idiot! You are so smart! A wave of smugness washes over you.
You have a leader, who shows you the warning signs. It's coming! It'll be soon now!
You have a book that you follow. It is very detailed and very convincing!
They ask for your money. It's for a good cause!
The cult is kinda also into [sex stuff](https://www.verysane.ai/p/ai-safety-is-mostly-a-sex-cult-in-berkeley-california) but you reckon that's [not relevant](https://sfstandard.com/2026/10/10/slutcon-flirting-boot-camp-ai/), just [peripheral](https://www.bloomberg.com/news/features/2023-03-07/effective-altruism-s-problems-go-beyond-sam-bankman-fried), just [a weird side thing](https://www.vice.com/en/article/i-went-to-a-torture-party-with-polyamorous-bay-area-rationalists/).
As the years go by, the day of doom never comes. You might slowly come to the realisation that the rest of the world has moved on, and you should too. Or, more sadly, you go down with this cult that will slowly fizzle out and die.

This is a story that happens a lot. Cults offer a sense of community, meaning and purpose to those who otherwise do not have such things. 
No, you are not immune if you are smart.

The AI Existential Risk crowd have formed some little cults. How are they cults? They have the normal tells. They will never accept evidence to the contrary; they exclude anybody who doesn't match their beliefs. The mission statement of MIRI is "If ASI is developed and deployed any time soon, by any nation or group, via anything remotely resembling current methods, the most likely outcome is human extinction." - this is not a scientific hypothesis; it is not falsifiable. They can shift the goalposts endlessly on what 'soon' means, what 'resembling current methods' means, and even what "ASI" means.

If it walks like a duck and quacks like a duck -- it's a duck.

For them, real progress is not something to celebrate - it erodes their raison d'etre, their entire personality. As such, we can probably expect them to fall increasingly behind the real world. 
Those who work real jobs in AI Alignment will see this quickly, and pull apart.


----

![the future is bright](/assets/future-city.png)

Wrapping up - I want everybody to be as optimistic about the future as I am. 
I remember when GPT-3 was held back because there was worry it would flood the internet with false information. This didn't end up happening - the concerns were overblown. It's good to be considerate, but at some point you need to realise you are worrying too much.
And it's worth asking ourselves - why didn't GPT-3 ruin the internet? Because the fact is that it didn't. Even the more powerful open-source models have not ruined the internet.
In my opinion, it's because it is really difficult to cause real damage. There are so many more of us working to build and maintain things rather than destroy things. We have built the amazing world we know today! And we will continue to work for it.

I don't think things will be perfect, I think there will be important problems to navigate. But I want us to talk about and fix the real issues instead of made-up ones!

Let's plan for a good future and work towards that instead.
