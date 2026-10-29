FROM ghcr.io/openclaw/openclaw:2026.3.24

USER root
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl ca-certificates uuid-runtime && \
    rm -rf /var/lib/apt/lists/*
RUN npm install -g @anthropic-ai/claude-code 2>/dev/null || true
RUN mkdir -p /tmp/sandbox /root/.openclaw/agents/main /root/.openclaw/workspace/skills
USER root
CMD ["sleep", "infinity"]
