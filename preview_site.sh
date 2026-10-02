#!/usr/bin/env bash
# Local preview of the course website (Jekyll inside Docker).
# The docs/ folder is mounted into the container, so every edit you save
# is rebuilt automatically; refresh the browser to see it.
#
# Usage (from the repository root, with Docker Desktop running):
#   ./preview_site.sh start     build the image if needed, start the site, open the browser
#   ./preview_site.sh stop      stop the site
#   ./preview_site.sh restart   stop and start again
#   ./preview_site.sh logs      follow the Jekyll output (Ctrl + C to leave)
#   ./preview_site.sh status    show whether the container is running
#   ./preview_site.sh rebuild   delete the container and the image, then start from scratch
#                               (use it after changing the Gemfile or the Dockerfile)

set -euo pipefail

IMAGE="ia-engineering-practicals-site"
CONTAINER="ia-engineering-practicals-site-container"
PORT="${PORT:-4000}"
DOCS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/docs"
URL="http://localhost:${PORT}/ai-engineering-practicals/"

require_docker() {
  if ! docker info >/dev/null 2>&1; then
    echo "Docker is not running. Open Docker Desktop, wait until it is ready, and try again."
    exit 1
  fi
}

container_exists() {
  docker ps -a --format '{{.Names}}' | grep -qx "$CONTAINER"
}

build() {
  echo "Building the image (only needed the first time)..."
  docker build -t "$IMAGE" "$DOCS_DIR"
}

start() {
  require_docker
  docker image inspect "$IMAGE" >/dev/null 2>&1 || build
  if container_exists; then
    docker start "$CONTAINER" >/dev/null
  else
    docker run -d --name "$CONTAINER" -p "${PORT}:4000" -v "$DOCS_DIR":/site "$IMAGE" \
      bundle exec jekyll serve --host 0.0.0.0 --port 4000 --force_polling >/dev/null
  fi
  echo "Starting the site. It will be ready in a few seconds at:"
  echo "  $URL"
  echo "Follow the build with: ./preview_site.sh logs"
  if command -v open >/dev/null 2>&1; then
    (sleep 5 && open "$URL") >/dev/null 2>&1 &
  fi
}

stop() {
  require_docker
  if container_exists; then
    docker stop "$CONTAINER" >/dev/null
    echo "Site stopped."
  else
    echo "The site is not running."
  fi
}

case "${1:-}" in
  start)   start ;;
  stop)    stop ;;
  restart) stop; start ;;
  logs)    require_docker; docker logs -f "$CONTAINER" ;;
  status)  require_docker; docker ps -a --filter "name=^${CONTAINER}$" --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}' ;;
  rebuild)
    require_docker
    docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
    docker rmi "$IMAGE" >/dev/null 2>&1 || true
    build
    start ;;
  *)
    sed -n '2,14p' "$0" | sed 's/^# \{0,1\}//'
    exit 1 ;;
esac
