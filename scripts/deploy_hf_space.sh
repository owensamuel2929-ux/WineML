#!/usr/bin/env bash
#
# Assemble and push the Hugging Face Space deployment.
#
# Hugging Face only auto-detects a Dockerfile named `Dockerfile` at the
# repository root, and Spaces require README.md frontmatter at that same root.
# Neither can be relocated via configuration, so the Space is a *separate*
# repository assembled from this project rather than a branch of it. That keeps
# the main repository's two-service compose setup untouched.
#
# The script stages a minimal, deployment-only working tree in a temporary
# directory and force-pushes it. Force-pushing is intentional: the Space is a
# build artifact, not a place where edits should accumulate.
#
# Usage:
#   scripts/deploy_hf_space.sh <hf-username>/<space-name>
#
# Example:
#   scripts/deploy_hf_space.sh owensamuel2929-ux/WineML
#
# Requires the Hugging Face CLI:
#   pip install -U "huggingface_hub[cli]"
#   hf auth login

set -euo pipefail

SPACE_ID="${1:-}"

if [[ -z "$SPACE_ID" ]]; then
    echo "error: missing Space ID" >&2
    echo "usage: $0 <hf-username>/<space-name>" >&2
    echo "example: $0 owensamuel2929-ux/WineML" >&2
    exit 2
fi

if [[ "$SPACE_ID" != */* ]]; then
    echo "error: Space ID must be '<username>/<space-name>', got '$SPACE_ID'" >&2
    exit 2
fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEPLOY_DIR="$PROJECT_ROOT/deploy/huggingface"

# ---------------------------------------------------------------------------
# Preflight: the image bakes in the model artifacts, so a Space built without
# them starts in the API's degraded state and every prediction returns 503.
# Catching that here is far cheaper than debugging it from Space logs.
# ---------------------------------------------------------------------------
REQUIRED_ARTIFACTS=(
    "$PROJECT_ROOT/models/classifier.joblib"
    "$PROJECT_ROOT/models/regressor.joblib"
    "$PROJECT_ROOT/models/model_metadata.json"
)

echo "==> Checking model artifacts"
missing=()
for artifact in "${REQUIRED_ARTIFACTS[@]}"; do
    if [[ -f "$artifact" ]]; then
        printf '    ok      %s (%s)\n' "$(basename "$artifact")" "$(du -h "$artifact" | cut -f1)"
    else
        printf '    MISSING %s\n' "$artifact"
        missing+=("$artifact")
    fi
done

if (( ${#missing[@]} > 0 )); then
    echo >&2
    echo "error: model artifacts are missing and must exist before deploying." >&2
    echo "       The Space bakes them into the image; without them the API" >&2
    echo "       starts degraded and every prediction returns HTTP 503." >&2
    echo >&2
    echo "       Run 'make train' first." >&2
    exit 1
fi

if [[ ! -f "$DEPLOY_DIR/Dockerfile" ]]; then
    echo "error: $DEPLOY_DIR/Dockerfile not found" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Verify the HF CLI is present and authenticated before doing any work.
# ---------------------------------------------------------------------------
if ! command -v hf >/dev/null 2>&1; then
    echo >&2
    echo "error: the Hugging Face CLI ('hf') is not installed." >&2
    echo "       pip install -U 'huggingface_hub[cli]'" >&2
    exit 1
fi

echo "==> Checking Hugging Face authentication"
if ! hf auth whoami >/dev/null 2>&1; then
    echo >&2
    echo "error: not logged in to Hugging Face. Run: hf auth login" >&2
    exit 1
fi
hf auth whoami | sed 's/^/    /'

# ---------------------------------------------------------------------------
# Stage a deployment-only tree.
# ---------------------------------------------------------------------------
STAGING="$(mktemp -d)"
trap 'rm -rf "$STAGING"' EXIT

echo "==> Staging deployment tree"

# Dockerfile and README must sit at the repository root for Spaces to detect
# them, so the deployment variants replace their project counterparts.
cp "$DEPLOY_DIR/Dockerfile" "$STAGING/Dockerfile"
cp "$DEPLOY_DIR/README.md" "$STAGING/README.md"

# Everything the image needs. The .dockerignore below excludes the rest, but
# only what is copied here can enter the build context at all.
cp -R "$PROJECT_ROOT/src" "$STAGING/src"
cp -R "$PROJECT_ROOT/app" "$STAGING/app"
cp -R "$PROJECT_ROOT/tests" "$STAGING/tests"
cp -R "$PROJECT_ROOT/raw_data" "$STAGING/raw_data"
cp -R "$DEPLOY_DIR" "$STAGING/deploy"
cp "$PROJECT_ROOT/pyproject.toml" "$STAGING/pyproject.toml"
cp "$PROJECT_ROOT/uv.lock" "$STAGING/uv.lock"

# Model artifacts are baked into the image. A Space has no compose volume, so
# there is nowhere else for them to come from. ~16 MB, and it makes the image
# fully self-contained.
mkdir -p "$STAGING/models"
cp "$PROJECT_ROOT/models/"*.joblib "$STAGING/models/"
cp "$PROJECT_ROOT/models/model_metadata.json" "$STAGING/models/"

# The project README is reused as the long-form documentation; the Space README
# is what HF reads and must stay at the root.
cp "$PROJECT_ROOT/README.md" "$STAGING/PROJECT_README.md" 2>/dev/null || true
cp "$PROJECT_ROOT/ARCHITECTURE.md" "$STAGING/ARCHITECTURE.md"

# Build context exclusions. Keep this tight: the Space has a size budget and
# every stray file slows the build.
cat > "$STAGING/.dockerignore" <<'DOCKERIGNORE'
.git/
.venv/
venv/
__pycache__/
*.py[cod]
*.egg-info/
.pytest_cache/
.ruff_cache/
.coverage
htmlcov/
coverage.xml
.ipynb_checkpoints/
.DS_Store
PROJECT_README.md
Dockerfile.hf
DOCKERIGNORE

cat > "$STAGING/.gitignore" <<'GITIGNORE'
__pycache__/
*.py[cod]
.venv/
.DS_Store
GITIGNORE

echo "==> Staged contents"
du -sh "$STAGING"
find "$STAGING" -maxdepth 1 -mindepth 1 | sort | sed 's|.*/|    |'

# ---------------------------------------------------------------------------
# Publish.
# ---------------------------------------------------------------------------
cd "$STAGING"
git init -q
git symbolic-ref HEAD refs/heads/main
git add -A

STAGED_SIZE="$(git diff --cached --name-only | wc -l | tr -d ' ')"
echo "==> Committing $STAGED_SIZE files"

git -c user.name="deploy-hf-space" \
    -c user.email="deploy-hf-space@localhost" \
    commit -q -m "deploy: assemble Space from WineML $(git -C "$PROJECT_ROOT" rev-parse --short HEAD 2>/dev/null || echo 'unknown')

Generated by scripts/deploy_hf_space.sh. Do not edit this Space directly —
changes belong in the main repository:
https://github.com/owensamuel2929-ux/WineML"

echo "==> Pushing to huggingface.co/spaces/$SPACE_ID"

# Plain git over HTTPS, authenticated by the credential helper `hf auth login`
# installs. Chosen over `hf upload` because git is the mechanism Spaces
# themselves use, so failures are diagnosable with familiar tooling.
git remote add space "https://huggingface.co/spaces/$SPACE_ID" 2>/dev/null || \
    git remote set-url space "https://huggingface.co/spaces/$SPACE_ID"

# Force-push: the Space mirrors the main repository, so its history is
# disposable and a normal push would reject on any divergence.
GIT_TERMINAL_PROMPT=0 git push --force space main

echo
echo "==> Done."
echo "    Space:  https://huggingface.co/spaces/$SPACE_ID"
echo "    Build:  https://huggingface.co/spaces/$SPACE_ID/logs"
echo
echo "    First build takes a few minutes. Watch the logs for the"
echo "    '[entrypoint]' lines to confirm all three services start."