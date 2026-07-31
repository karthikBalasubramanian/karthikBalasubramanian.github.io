
Karthik's Official website
==========================

I would like to thank [Jack](https://github.com/JiaKunUp) for his amazing work on [Jekyllrb](http://jekyllrb.com/). I have built my website on top of his [jekyll template](https://github.com/JiaKunUp/jalpc_jekyll_theme). You can get all the intricacies about this website's development there.

## Microsites Architecture (Jekyll + Vite SPAs)

This repository supports an automated, zero-config multi-microsite architecture:
- **Root Site**: Jekyll powers the main portfolio landing page, blog, and navigation.
- **Microsites**: Any root-level subfolder containing a `package.json` file is recognized as a microsite (e.g. `child-financial-investment-planner`, `paycheck-tax-&-investment-allocator`).

### Adding a New Microsite
1. Create a subfolder with your Vite/React application (e.g., `my-new-tool`).
2. Ensure `base: './'` is set in `my-new-tool/vite.config.ts`.
3. Push to `main` / `master`. 

GitHub Actions will automatically discover your app, run `npm run build`, and deploy it to `https://<username>.github.io/my-new-tool/`.

