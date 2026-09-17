# Little site for myself

This is a little site for myself to scrawl my thoughts and let everyone else see them!

https://alexchaloner.github.io/

## link linter

I made a little static check to make sure all links on the site are working.

Requires Ruby and npm.

Open powershell in this folder, run:
```
./lint_links
```

## Blog

Posts, including the interactive flow-matching and IDBD articles, live in `blog/`.
Shared scripts, styles and precomputed illustration data live in `assets/`.
The home page and `/blog/` list dated posts. Legacy URLs are preserved by
`redirects/` and `_layouts/redirect.html`.
