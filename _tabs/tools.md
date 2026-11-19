---
title: Tools
icon: fas fa-calculator
order: 3
description: Free personal finance calculators for tech employees, covering take-home pay, rent vs buy, 529 planning and RSU hold-vs-sell strategy.
---

Calculators I built in React and TypeScript for people at public tech companies. Use them to stress-test your income,
housing budget, family savings and RSU strategy. **Everything runs in your browser. I don't collect, store or use your data.**
They're tools for exploring options, not financial advice.

<div class="kb-tools">
{% for tool in site.data.tools %}
  <a class="kb-tool" href="{{ tool.url | relative_url }}">
    <span class="kb-card-kicker">{{ tool.step }}</span>
    <span class="kb-card-title">{{ tool.title }}</span>
    <span class="kb-card-text">{{ tool.goal }}</span>
    <ul class="kb-tool-features">
      {% for f in tool.features %}<li>{{ f }}</li>{% endfor %}
    </ul>
    <span class="kb-tool-cta">Open the calculator <i class="fas fa-arrow-right"></i></span>
  </a>
{% endfor %}
</div>

Steps 01 and 02 work as a pair: your take-home pay from the paycheck allocator becomes the budget limit in the rent vs. buy planner.
