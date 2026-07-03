# PhysAgent Project Page

This directory is a self-contained GitHub Pages package.

## Deploy

Create a GitHub repository, then run:

```bash
cd /home/huteng/lliqirui/RealWonder/project_page_github_pages_0703
git init
git add .
git commit -m "Add PhysAgent project page"
git branch -M main
git remote add origin git@github.com:<USER>/<REPO>.git
git push -u origin main
```

In GitHub:

1. Open repository Settings.
2. Go to Pages.
3. Set source to `Deploy from a branch`.
4. Select branch `main` and folder `/root`.

The page will be available at:

```text
https://<USER>.github.io/<REPO>/
```

## Notes

- `index.html` is the entry page.
- `.nojekyll` is included so GitHub Pages serves all static files directly.
- The overview demo is full length but re-encoded to a smaller GitHub-friendly file.
