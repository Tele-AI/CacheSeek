# Fluxon KV stack — helper scripts

Operational scripts for bringing up the **Fluxon** KV backend that CacheSeek uses
when `kv_store_type=fluxon` (the TeleFuser Wan2.2 deployment).

| Script | Role |
| --- | --- |
| `start_etcd_greptime.sh` | data plane — starts tmux session `fluxon_kv_stack` with etcd (2379) / greptime (34030) / pd (12379) / tikv (20160) |
| `start_master_owner.py` | control + storage — starts the kv **master** (port 31000, web UI 18080) and **owner** (64 GB DRAM pool); foreground, Ctrl-C to stop |
| `kill_etcd_greptime.sh` | tears down the `fluxon_kv_stack` tmux session |
| `external_config.yaml` | the **client** config the service points at via `fluxon_config_path`; its `cluster_name` / `shared_memory_path` must match `start_master_owner.py` |

> **Run-in-place.** These are canonical copies. `start_etcd_greptime.sh` uses
> `$(pwd)` + relative `./ext_images/...` and `start_master_owner.py` imports
> `fluxon_py` from the TeleFuser venv — so they must be **run from the Fluxon
> release-bundle dir on the host** (the dir holding `ext_images/`), not from here.

Startup / restart / troubleshooting: see the inline comments in
`start_master_owner.py` and `start_etcd_greptime.sh`.

---

## 启动（standalone runbook，2026-06-10 实战版）

```bash
# 0) 先查是不是已经在跑 —— 已在跑就别重复起
pgrep -af "fluxon_py.runtime" && ls /dev/shm/ | grep -i flux
#    两者都有 → 栈活着,跳到「客户端连接」

# 1) 数据面(etcd/greptime,tmux 会话 fluxon_kv_stack;通常常驻,挂了才重启)
cd <fluxon-release-bundle-dir>          # 含 ext_images/ 的目录
bash start_etcd_greptime.sh

# 2) master + owner(64GB DRAM 池;前台脚本,用 nohup/tmux 托管)
cd <fluxon-repo-or-bundle-dir>          # start_master_owner.py 所在目录,run-in-place
nohup <venv>/bin/python start_master_owner.py > /tmp/fluxon_master_owner.log 2>&1 &

# 3) 等 owner 就绪(shm bundle 出现即可)
until [ -d /dev/shm/<cluster_shm_name> ]; do sleep 3; done
```

## 客户端连接 —— 唯一高频坑

client 的 `external_config.yaml` 里 **`cluster_name` + `shared_memory_path` 必须与
运行中 master/owner 的完全一致**。不一致的症状非常有辨识度：

| 症状 | 含义 | 处置 |
|---|---|---|
| client 刷 `Waiting owner shared bundle to be ready... (Ns)` 不止 | 连了不存在/不匹配的集群,或 owner 已死 | 核对 cluster_name（解码运行中 master 的 `--config-b64` 可见真值:`tr '\0' ' ' </proc/<master_pid>/cmdline \| grep -o 'config-b64 [A-Za-z0-9+/=]*' \| cut -d' ' -f2 \| base64 -d`） |
| client 海量 `Lease keepalive response ttl invalid` + `transport_tcp` 报错、同一 msg_id 无限重试 | 栈运行态腐化（长期闲置后常见） | 重启 master/owner（见下） |
| `get_tensor() missing ... 'shape' and 'dtype'` | API 用法：Fluxon 存原始 buffer,取回必须给 view 规格 | 调用侧补 shape/dtype（cacheseek 的 `TensorStoreTierStore` 已内置 spec 簿记） |

## 重启 master/owner（清洁流程）

```bash
# 1) 按 PID 杀,不要 pkill -f —— 模式会匹配到你自己的 ssh/脚本 cmdline 把会话杀掉
pgrep -af "fluxon_py.runtime"           # 记下 master/owner 以及父进程 PID
kill <pids>; sleep 3
pgrep -af "fluxon_py.runtime" || echo dead   # 必须确认真死(SIGKILL 对 D 态进程有延迟)

# 2) 验证端口释放(机器可能没有 ss,用 python)
python3 - <<'PY'
import socket
for p in (31000, 18080):
    s = socket.socket(); print(p, "free" if s.connect_ex(("127.0.0.1", p)) else "OCCUPIED"); s.close()
PY

# 3) 清残留 shm bundle 再启(防新 owner 撞旧状态)
rm -rf /dev/shm/<cluster_shm_name>
# 然后回到「启动」第 2 步
```

> 实测参考(H100 同机,dlpack):GET ≈ 12.2 GB/s(shm 零拷贝 view),PUT ≈ 1.06 GB/s。
