---
layout: default
title: Home
---

Howdy, welcome to my site. I'm a Software Engineer and a part-time tutor and I like to make stuff.
Here I will post some thoughts, explain some ideas, and share cool things.

## Projects

- [Kryptos Sandbox](https://alexchaloner.github.io/kryptos-sandbox/) — an interactive cryptanalytic workspace for experimenting with grids, classical ciphers, and transpositions.

## Dissertation

*Hierarchical Reinforcement Learning* — compressing action policies via graph abstraction.

## Blog posts

- 2026-05-03 [Simulating the Red Button / Blue Button Hypothetical](https://alexchaloner.substack.com/p/simulating-the-red-button-blue-button)
{% assign sorted_pages = site.pages | sort: 'date' | reverse %}
{% for p in sorted_pages %}
  {% if p.url contains '/musings/' and p.url != '/musings/' %}
- {{ p.date | date: "%Y-%m-%d" }} [{{ p.title }}]({{ p.url | relative_url }})
  {% endif %}
{% endfor %}
