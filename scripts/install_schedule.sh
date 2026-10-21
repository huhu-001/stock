#!/bin/bash
# 在 macOS 上安装定时任务：每个工作日 14:30 自动运行尾盘选股
#   安装:  bash scripts/install_schedule.sh
#   卸载:  bash scripts/install_schedule.sh uninstall
#   时间:  bash scripts/install_schedule.sh 14 35   （自定义 时 分）
set -e
LABEL="com.stock.tail"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
DIR="$(cd "$(dirname "$0")/.." && pwd)"

if [ "$1" = "uninstall" ]; then
  launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
  rm -f "$PLIST"
  echo "已卸载定时任务。"
  exit 0
fi

HOUR="${1:-14}"
MINUTE="${2:-30}"
if [ -x "$DIR/.venv/bin/python" ]; then PY="$DIR/.venv/bin/python"; else PY="$(command -v python3)"; fi
if ! "$PY" -c "import akshare, plotly, yaml" 2>/dev/null; then
  echo "❌ $PY 中缺少依赖，请先执行: pip install -r requirements.txt"; exit 1
fi
mkdir -p "$DIR/output" "$HOME/Library/LaunchAgents"

INTERVALS=""
for d in 1 2 3 4 5; do
  INTERVALS="$INTERVALS<dict><key>Weekday</key><integer>$d</integer><key>Hour</key><integer>$HOUR</integer><key>Minute</key><integer>$MINUTE</integer></dict>"
done

cat > "$PLIST" <<PL
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>$LABEL</string>
  <key>ProgramArguments</key><array>
    <string>$PY</string><string>$DIR/main.py</string><string>tail</string>
  </array>
  <key>WorkingDirectory</key><string>$DIR</string>
  <key>StartCalendarInterval</key><array>$INTERVALS</array>
  <key>EnvironmentVariables</key><dict>
    <key>PATH</key><string>/usr/local/bin:/opt/homebrew/bin:/usr/bin:/bin</string>
    <key>PYTHONIOENCODING</key><string>utf-8</string>
  </dict>
  <key>StandardOutPath</key><string>$DIR/output/schedule.log</string>
  <key>StandardErrorPath</key><string>$DIR/output/schedule.log</string>
</dict></plist>
PL

launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
launchctl bootstrap "gui/$(id -u)" "$PLIST"
printf '✅ 已安装：每周一至周五 %02d:%02d 自动运行尾盘选股\n' "$HOUR" "$MINUTE"
echo "   Python: $PY"
echo "   日志:   $DIR/output/schedule.log"
echo "   立即试运行: launchctl kickstart gui/$(id -u)/$LABEL"
