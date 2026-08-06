
Karthik's Official website
==========================

I would like to thank [Jack](https://github.com/JiaKunUp) for his amazing work on [Jekyllrb](http://jekyllrb.com/). I have built my website on top of his [jekyll template](https://github.com/JiaKunUp/jalpc_jekyll_theme). You can get all the intricacies about this website's development there.

## Microsites Architecture (Jekyll + Vite SPAs)

This repository supports an automated, zero-config multi-microsite architecture:
### Personal Wealth Operating System (3-Step Roadmap)
1. **Paycheck & Tax Allocator** (`/paycheck-tax-investment-allocator/`): Calculates liquid net take-home cash after all taxes, 401(k), HSA, & ESPP deductions.
2. **Rent vs Buy & House Poor Stress Tester** (`/rent-vs-buy-house-poor-calculator/`): Evaluates target home purchase in target ZIP Code vs rent after lifestyle expenses.
3. **Child Financial Investment Planner** (`/child-financial-investment-planner/`): Projects 18-year compounding wealth for 529 College Plans & Custodial accounts.

### Adding a New Microsite
1. Create a subfolder with your Vite/React application (e.g., `my-new-tool`).
2. Ensure `base: '/my-new-tool/'` is set in `my-new-tool/vite.config.ts`.
3. Push to `main` / `master`. 

GitHub Actions will automatically discover your app, run `npm run build`, and deploy it to `https://<username>.github.io/my-new-tool/`.


