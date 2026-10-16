# eslint-plugin-svelte quirks

Область: [Сборка и инструменты](../getting-started.md)

- **A space next to block tags: write `&#32;`, not `{' '}`.** Svelte 4 itself keeps a plain space (`, {/if}`, `{/if} {:else}`), but prettier-plugin-svelte reflows it into a newline plus indentation, which breaks lines under `white-space: pre-wrap`. `&#32;` compiles to a single space in both SSR and DOM output, prettier leaves it alone, and `svelte/no-useless-mustaches` does not flag it (it flags `{' '}` and cannot autofix it).
- **Svelte 4 rejects JS comments inside a mustache** (`{' ' /* ... */}` is a parse error), so eslint directives in markup go into HTML comments.
- **HTML-comment directives need both** `processor: svelte/svelte` and the `svelte/comment-directive` rule; the UI profile gets them from `plugin:svelte/prettier`. A hand-made `--no-eslintrc` config without them silently ignores the directives.
- Repo-wide autofix of a single rule: a scratch config with `svelte-eslint-parser`, `git ls-files -z '*.svelte' | xargs -0 packages/ui/node_modules/.bin/eslint --no-eslintrc -c <cfg> --resolve-plugins-relative-to packages/ui --no-ignore --fix`. No type info is needed for template rules, so this is minutes instead of a full `rush lint`.
