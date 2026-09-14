# IA Engineering Practicals

Course website and materials for **MEEN41490 — AI in Engineering** (UCD). Each
practical session gets its own self-contained folder under `docs/practicals/`,
holding its walkthrough page, Colab notebook, and images together.

The site is published with GitHub Pages, set to deploy from the `main` branch,
`/docs` folder — GitHub builds the Jekyll site in `docs/` automatically on
every push.

See `claude.md` for module context, schedule, and the process for building a
new practical.

## Structure

```
docs/
├── index.md                # homepage, lists practicals from _data/practicals.yml
├── _data/practicals.yml    # ordered {week, title, path} entries shown on the homepage
├── guides/                 # cross-cutting how-tos (not tied to one week)
│   └── getting-started-colab/
└── practicals/              # one self-contained folder per practical
    └── inverted_pendulum_neuron_control/
        ├── index.md
        ├── notebook.ipynb
        └── assets/
```

## Running locally

The site is a Jekyll site built and served from a Docker container, with the
`docs/` folder bind-mounted so edits trigger auto-regeneration.

```bash
cd docs
docker build -t ia-engineering-practicals-site .
docker run -d --name ia-engineering-practicals-site-container \
  -p 4000:4000 -v "$PWD":/site ia-engineering-practicals-site
```

The site's `baseurl` is `/ai-engineering-practicals` (matching the GitHub
Pages project URL), so open http://localhost:4000/ai-engineering-practicals/
rather than the bare root.

- Logs: `docker logs -f ia-engineering-practicals-site-container`
- Stop: `docker stop ia-engineering-practicals-site-container`
