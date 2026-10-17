# CI Docker namespace for forks

- Scripts and compose files read `DOCKER_NAMESPACE` (default `intabiafusion`). CI sets it from the repo variable `vars.DOCKER_NAMESPACE`; the same value is the Docker Hub login user, token in `secrets.DOCKER_ACCESS_TOKEN`.
- A fork (e.g. `haiodo/foundation`) sets `DOCKER_NAMESPACE=haiodo` and its own token; no workflow edits needed after merging upstream.
- The namespace applies to build, tests and push alike. Images that CI pulls instead of building must exist in the fork's namespace: `elastic-plugins:8.19.1-slim` (tests/ws-tests compose). Copy it once: `docker buildx imagetools create -t haiodo/elastic-plugins:8.19.1-slim intabiafusion/elastic-plugins:8.19.1-slim`.
- Base images in Dockerfiles stay `FROM intabiafusion/base:...` (public), independent of the namespace.
- Images are pushed only on tags `v*`/`s*`; a push to `develop` builds and tests without publishing.
