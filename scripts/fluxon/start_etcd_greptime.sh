#!/usr/bin/env bash
set -euo pipefail

SESSION_NAME="fluxon_kv_stack"
ROOT_DIR=$(pwd)

ETCD_CONFIG_PATH="/tmp/etcd.config.sh"
GREPTIME_CONFIG_PATH="/tmp/greptime.config.sh"
PD_CONFIG_PATH="/tmp/pd.config.sh"
TIKV_CONFIG_PATH="/tmp/tikv.config.sh"


main() {
  require_command tmux
  require_command python3

  require_file "$ROOT_DIR/ext_images/etcd/start.sh"
  require_file "$ROOT_DIR/ext_images/greptime/start.sh"
  require_file "$ROOT_DIR/ext_images/tikv/start_pd.sh"
  require_file "$ROOT_DIR/ext_images/tikv/start_tikv.sh"
  # Deployment flow A starts kv master and owner in start_kv_and_fs_svc_master.py.
  # require_file "$ROOT_DIR/start_master_owner.py"

  if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "tmux session already exists: $SESSION_NAME"
    exit 1
  fi

  write_etcd_config
  write_greptime_config
  write_pd_config
  write_tikv_config

  tmux new-session -d -s "$SESSION_NAME" -n etcd
  send_window_command "etcd" "cd '$ROOT_DIR' && bash ./ext_images/etcd/start.sh --config '$ETCD_CONFIG_PATH' --workdir /tmp/fluxon_service_plane_demo/etcd"

  tmux new-window -t "$SESSION_NAME" -n greptime
  send_window_command "greptime" "cd '$ROOT_DIR' && bash ./ext_images/greptime/start.sh --config '$GREPTIME_CONFIG_PATH' --workdir /tmp/fluxon_service_plane_demo/greptime"

  tmux new-window -t "$SESSION_NAME" -n pd
  send_window_command "pd" "cd '$ROOT_DIR' && bash ./ext_images/tikv/start_pd.sh --config '$PD_CONFIG_PATH' --workdir /tmp/fluxon_service_plane_demo/tikv_pd"

  tmux new-window -t "$SESSION_NAME" -n tikv
  send_window_command "tikv" "cd '$ROOT_DIR' && bash ./ext_images/tikv/start_tikv.sh --config '$TIKV_CONFIG_PATH' --workdir /tmp/fluxon_service_plane_demo/tikv"

  # Deployment flow A keeps this tmux session focused on external dependencies only.
  # tmux new-window -t "$SESSION_NAME" -n master_owner
  # send_window_command "master_owner" "cd '$ROOT_DIR' && python3 start_master_owner.py"

  print_summary
}


require_command() {
  local name="$1"
  if ! command -v "$name" >/dev/null 2>&1; then
    echo "required command not found: $name"
    exit 1
  fi
}


require_file() {
  local path="$1"
  if [ ! -f "$path" ]; then
    echo "required file not found: $path"
    echo "run this script from the repo root or release bundle root"
    exit 1
  fi
}


write_etcd_config() {
  cat >"$ETCD_CONFIG_PATH" <<'EOF'
ETCD_ARGS=(
  --data-dir "$WORKDIR/etcd-data"
  --name etcd0
  --advertise-client-urls "http://127.0.0.1:2379"
  --listen-client-urls "http://0.0.0.0:2379"
  --listen-peer-urls "http://0.0.0.0:2380"
  --initial-advertise-peer-urls "http://127.0.0.1:2380"
  --initial-cluster "etcd0=http://127.0.0.1:2380"
  --initial-cluster-token "etcd-cluster"
  --initial-cluster-state "new"
  --auto-compaction-retention=1
)
EOF
}


write_greptime_config() {
  cat >"$GREPTIME_CONFIG_PATH" <<'EOF'
GREPTIME_ARGS=(
  standalone start
  --data-home "$WORKDIR/greptimedb"
  --http-addr 0.0.0.0:34030
)
EOF
}


write_pd_config() {
  cat >"$PD_CONFIG_PATH" <<'EOF'
PD_ARGS=(
  --name pd0
  --data-dir "$WORKDIR/pd-data"
  --client-urls "http://127.0.0.1:12379"
  --advertise-client-urls "http://127.0.0.1:12379"
  --peer-urls "http://127.0.0.1:12380"
  --advertise-peer-urls "http://127.0.0.1:12380"
  --initial-cluster "pd0=http://127.0.0.1:12380"
  --log-file "$WORKDIR/pd.log"
)
EOF
}


write_tikv_config() {
  cat >"$TIKV_CONFIG_PATH" <<'EOF'
TIKV_ARGS=(
  --pd-endpoints "127.0.0.1:12379"
  --addr "127.0.0.1:20160"
  --advertise-addr "127.0.0.1:20160"
  --status-addr "127.0.0.1:20180"
  --data-dir "$WORKDIR/tikv-data"
  --log-file "$WORKDIR/tikv.log"
)
EOF
}


send_window_command() {
  local window_name="$1"
  local command_text="$2"
  tmux send-keys -t "$SESSION_NAME:$window_name" "$command_text" C-m
}


print_summary() {
  cat <<EOF
started tmux session: $SESSION_NAME
windows:
  - etcd
  - greptime
  - pd
  - tikv
attach:
  tmux attach -t $SESSION_NAME
kill:
  bash kill_kv_stack_tmux.sh
EOF
}


main "$@"

