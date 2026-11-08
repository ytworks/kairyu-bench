# Source/base ABI and dependency versions must match the pinned staging receipt.
# No RUN, network download or package-install hook.
ARG BASE_IMAGE
FROM ${BASE_IMAGE}
COPY --chown=1003:1003 site-packages/ /usr/local/lib/python3.12/dist-packages/
USER 1003:1003

