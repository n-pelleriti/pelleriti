---
permalink: /
title: "About Me"
author_profile: true
redirect_from: 
  - /about/
  - /about.html
---

I am a PhD candidate in Mathematics at [TU Berlin](https://www.tu.berlin/), working in the [Interactive Optimization & Learning (IOL) lab](https://iol.zib.de/research/iol-learn.html) under the supervision of [Prof. Sebastian Pokutta](https://www.pokutta.com/). My research focuses on developing agentic AI systems for mathematical and scientific discovery, with particular attention to the interplay between autonomous reasoning and interactive, human-guided exploration. In parallel, I study foundational questions in optimization theory. Prior to this, I worked on learning-augmented algorithms that incorporate predictive models into classical methods from computational algebra.

Recent Publications
======

{% assign recent_pubs = site.publications | sort: 'date' | reverse | slice: 0, 3 %}
{% for post in recent_pubs %}
{% include archive-single-pub.html %}
{% endfor %}

