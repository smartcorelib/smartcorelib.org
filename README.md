# smartcorelib.org

This domain redirects to **https://smartcorelib.github.io/** — the canonical SmartCore website, now served via GitHub Pages from the [smartcorelib.github.io](https://github.com/smartcorelib/smartcorelib.github.io) repo.

This repo previously hosted the Jekyll source for the 0.2.0-era site. That source is preserved in the git history (see commit `2edb480` and earlier). Going forward, **all website edits go in [smartcorelib.github.io](https://github.com/smartcorelib/smartcorelib.github.io)**; this repo exists only to hold the redirect.

## Deployment

`index.html` is a redirect page. Deploy it to whatever hosting serves `smartcorelib.org` (currently AWS S3 + CloudFront). Once deployed, every request to `smartcorelib.org/*` is redirected to `smartcorelib.github.io/`.