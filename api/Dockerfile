# syntax=docker/dockerfile:1.7
# The GPU runtime is built independently by Dockerfile.runtime. Rebuilding this
# image after a source-code change copies only app/ and tools/.
FROM manga-translator-runtime:gpu AS gpu

WORKDIR /app
COPY --link --chown=10001:10001 app ./app
COPY --link --chown=10001:10001 tools ./tools

USER translator
EXPOSE 8000
CMD ["python", "-m", "tools.run_container"]
