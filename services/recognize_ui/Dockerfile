# syntax=docker/dockerfile:1.6
#
# Phase 6.1 frontend image: nginx serves the static SPA and proxies
# /api/* and /health to the recognize-api container.
#
# No build step needed -- everything is plain HTML/CSS/JS.

FROM nginx:alpine

COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY static/ /usr/share/nginx/html/

# nginx:alpine exposes 80 by default; Compose maps it to host 8080.
EXPOSE 80
