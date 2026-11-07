ARG BASE_IMAGE
FROM ${BASE_IMAGE}
# The owner pins BASE_IMAGE to the audited immutable image ID before building.
COPY --chown=1003:1003 python/ /opt/kairyu-sparse-overlay/
COPY --chown=1003:1003 smoke/ /opt/kairyu-sparse-overlay-smoke/
ENV PYTHONPATH=/opt/kairyu-sparse-overlay:${PYTHONPATH}
USER 1003:1003
