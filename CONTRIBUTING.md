# Contributing to Castalia

Thank you for contributing to Castalia! This guide ensures consistency across all 176+ notebooks.

## Before You Start

1. Read [STYLE_GUIDE.md](STYLE_GUIDE.md) — every notebook must follow the template
2. Read [ROADMAP.md](ROADMAP.md) — understand where your notebook sits in the orbit structure
3. Check the domain README (e.g., `rag/README.md`) — your notebook must be listed there

## Notebook Checklist

Before submitting a notebook, verify:

### Structure
- [ ] Title cell with orbit, difficulty, reading time, Colab badge, prerequisites, learning objectives
- [ ] Colab setup cell as the second cell (pip installs + imports + model loading)
- [ ] Content cells alternate between markdown explanation and code
- [ ] Exercises section with at least 2 exercises
- [ ] Key takeaways table
- [ ] Cross-references (What's Next section with links)
- [ ] References & Further Reading section

### Content Quality
- [ ] First-principles explanations before API usage
- [ ] Comparison tables for tradeoffs
- [ ] Failure modes / limitations discussed
- [ ] Concrete numbers (VRAM, latency, token counts)

### Technical
- [ ] Runs cleanly on Google Colab Pro with T4 GPU
- [ ] Model weights cached to Google Drive
- [ ] No hardcoded API keys or secrets
- [ ] Print statements show intermediate results
- [ ] All pip installs use `-q` flag and minimum version pins

## File Naming

- Domain notebooks: `[NN]_[snake_case_title].ipynb` (e.g., `05_building_a_react_agent.ipynb`)
- Foundation notebooks: `[NN]_[snake_case_title].ipynb` in `foundations/`
- Use lowercase, underscores, no spaces

## Adding a New Notebook

1. Create the `.ipynb` following STYLE_GUIDE.md template
2. Add it to the domain's `README.md` table
3. Update `ROADMAP.md` to place it in the correct orbit
4. Add cross-references from/to adjacent notebooks
5. Test on Colab Pro with a fresh runtime

## Updating Existing Notebooks

- Preserve the existing cell structure (don't reorder sections)
- If adding content, maintain the markdown-to-code ratio (target 0.8:1 to 1.5:1)
- Update the reading time estimate if content changes significantly
- Run the full notebook on Colab Pro after changes

## Commit Messages

Use conventional commits:
```
feat(rag): add hybrid retrieval notebook
fix(evals): add missing Colab setup cell to 05
docs(roadmap): add foundations orbit
style(agents): standardize header template across all notebooks
```
