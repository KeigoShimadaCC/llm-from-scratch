# Wiki Source Policy

These files are the source of truth for the GitHub Wiki courseware. Publish them with:

```bash
uv run python scripts/publish_wiki.py --dry-run
uv run python scripts/publish_wiki.py --push --message "Add LLM from scratch course wiki"
```

`README.md` is intentionally not copied to the GitHub Wiki. The publish script copies only managed top-level Markdown
pages from this directory and preserves unmanaged pages that already exist in the wiki repository.
