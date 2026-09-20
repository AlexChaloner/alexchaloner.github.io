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
The home page and `/blog/` list dated posts unless `listed: false` is set.
Blog posts use the `blog-post` layout for reading margins; interactive posts
can set `wide: true` to retain room for diagrams. Legacy URLs are preserved by
`redirects/` and `_layouts/redirect.html`.

### Unlisted Twitter cross-posts

The four cross-posts in `blog/microblog/` have `published: true` and `listed: false`.
They are available at their `/blog/microblog/<name>.html` URLs, but are omitted
from the home-page and blog lists. To list a post, remove `listed: false`.
Each post links to the original thread at the top, retains the
original tweet date and wording, and stores the attached images in `assets/twitter/`.
Quoted tweets are linked separately. The scoped stylesheet preserves literal
asterisks and line breaks, and keeps image pairs in their original order.
