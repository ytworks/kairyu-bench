#!/bin/sh

# Run indexed input lines through long-lived workers. Each worker atomically
# claims the next pending item, so a free slot is refilled immediately instead
# of waiting for the other workers in a fixed batch.

worker_pool_cleanup_queue() {
    [ -n "${WORKER_POOL_QUEUE:-}" ] || return 0
    for queue_item in \
        "$WORKER_POOL_QUEUE"/*.pending \
        "$WORKER_POOL_QUEUE"/*.claimed \
        "$WORKER_POOL_QUEUE"/failed; do
        [ -e "$queue_item" ] || continue
        rm -f "$queue_item"
    done
    rmdir "$WORKER_POOL_QUEUE" >/dev/null 2>&1 || true
    WORKER_POOL_QUEUE=
}

worker_pool_stop() {
    for worker_pool_pid in ${WORKER_POOL_PIDS:-}; do
        kill "$worker_pool_pid" >/dev/null 2>&1 || true
    done
    for worker_pool_pid in ${WORKER_POOL_PIDS:-}; do
        wait "$worker_pool_pid" >/dev/null 2>&1 || true
    done
    WORKER_POOL_PIDS=
    worker_pool_cleanup_queue
}

worker_pool_worker() (
    worker_pool_callback=$1
    worker_pool_item_pid=

    worker_pool_worker_stop() {
        trap - HUP INT TERM
        if [ -n "$worker_pool_item_pid" ]; then
            kill "$worker_pool_item_pid" >/dev/null 2>&1 || true
            wait "$worker_pool_item_pid" >/dev/null 2>&1 || true
        fi
        exit 130
    }

    worker_pool_worker_finished() {
        worker_pool_status=$?
        trap - EXIT HUP INT TERM
        if [ "$worker_pool_status" -ne 0 ]; then
            : >"$WORKER_POOL_QUEUE/failed"
        fi
        exit "$worker_pool_status"
    }

    trap 'worker_pool_worker_finished' EXIT
    trap 'worker_pool_worker_stop' HUP INT TERM

    while [ ! -e "$WORKER_POOL_QUEUE/failed" ]; do
        worker_pool_claimed=
        for worker_pool_pending in "$WORKER_POOL_QUEUE"/*.pending; do
            [ -f "$worker_pool_pending" ] || continue
            worker_pool_candidate=${worker_pool_pending%.pending}.claimed
            if mv "$worker_pool_pending" "$worker_pool_candidate" 2>/dev/null; then
                worker_pool_claimed=$worker_pool_candidate
                break
            fi
        done
        [ -n "$worker_pool_claimed" ] || break

        worker_pool_tab=$(printf '\t')
        IFS="$worker_pool_tab" read -r worker_pool_index worker_pool_value \
            <"$worker_pool_claimed"
        "$worker_pool_callback" "$worker_pool_index" "$worker_pool_value" &
        worker_pool_item_pid=$!
        if wait "$worker_pool_item_pid"; then
            worker_pool_item_pid=
        else
            worker_pool_status=$?
            worker_pool_item_pid=
            exit "$worker_pool_status"
        fi
        rm -f "$worker_pool_claimed"
    done
)

worker_pool_run() {
    worker_pool_workers=$1
    WORKER_POOL_QUEUE=$2
    worker_pool_input=$3
    worker_pool_callback=$4
    WORKER_POOL_PIDS=

    mkdir -p "$WORKER_POOL_QUEUE"
    worker_pool_index=0
    while IFS= read -r worker_pool_value; do
        [ -n "$worker_pool_value" ] || continue
        worker_pool_index=$((worker_pool_index + 1))
        worker_pool_key=$(printf '%08d' "$worker_pool_index")
        printf '%s\t%s\n' "$worker_pool_index" "$worker_pool_value" \
            >"$WORKER_POOL_QUEUE/$worker_pool_key.pending"
    done <"$worker_pool_input"

    worker_pool_worker_number=0
    while [ "$worker_pool_worker_number" -lt "$worker_pool_workers" ]; do
        worker_pool_worker_number=$((worker_pool_worker_number + 1))
        worker_pool_worker "$worker_pool_callback" &
        WORKER_POOL_PIDS="$WORKER_POOL_PIDS $!"
    done

    worker_pool_failed=0
    for worker_pool_pid in $WORKER_POOL_PIDS; do
        wait "$worker_pool_pid" || worker_pool_failed=1
    done
    WORKER_POOL_PIDS=
    if [ "$worker_pool_failed" -ne 0 ] || [ -e "$WORKER_POOL_QUEUE/failed" ]; then
        worker_pool_cleanup_queue
        echo "kairyu-bench: one or more workers failed" >&2
        return 2
    fi
    worker_pool_cleanup_queue
}
