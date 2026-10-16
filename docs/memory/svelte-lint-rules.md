# eslint-plugin-svelte quirks

Область: [Сборка и инструменты](../getting-started.md)

- **`svelte/no-useless-mustaches` and `{' '}`.** A whitespace-only mustache is kept on purpose: Svelte trims plain whitespace at the edges of `{#if}`/`{#each}` blocks, so `{' '}` is the only way to keep the space (mention icon + title, comma-separated lists). The rule has no option for it and does not autofix it; mark it with `{' '}<!-- eslint-disable-line svelte/no-useless-mustaches -->`.
- **Svelte 4 rejects JS comments inside a mustache** (`{' ' /* ... */}` is a parse error), so eslint directives in markup go into HTML comments.
- **HTML-comment directives need both** `processor: svelte/svelte` and the `svelte/comment-directive` rule; the UI profile gets them from `plugin:svelte/prettier`. A hand-made `--no-eslintrc` config without them silently ignores the directives.
- Repo-wide autofix of a single rule: a scratch config with `svelte-eslint-parser`, `git ls-files -z '*.svelte' | xargs -0 packages/ui/node_modules/.bin/eslint --no-eslintrc -c <cfg> --resolve-plugins-relative-to packages/ui --no-ignore --fix`. No type info is needed for template rules, so this is minutes instead of a full `rush lint`.
