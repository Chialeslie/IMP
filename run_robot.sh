#!/bin/bash

# Define your absolute project directory
PROJECT_DIR="/home/chiagekliang/chiagekliangIMP"

# Navigate to the directory
cd "$PROJECT_DIR" || exit 1

echo "Activating virtual environment..."
source "$PROJECT_DIR/venv/bin/activate"

echo "Starting Flask Application (app.py)..."
python -u "$PROJECT_DIR/app.py" &

# Give Flask and the USB ports 5 seconds to fully initialize
sleep 5

echo "Starting Robot Main(IMP_robot_yunet_voice.py)..."
python -u "$PROJECT_DIR/IMP_robot_yunet_voice.py" > "$PROJECT_DIR/robot_error.log" 2>&1

# Function to handle clean shutdown when you press Ctrl+C
cleanup() {
    echo ""
    echo "Stopping all robot services..."
    kill $APP_PID 2>/dev/null
    kill $VOICE_PID 2>/dev/null
    wait $APP_PID 2>/dev/null
    wait $VOICE_PID 2>/dev/null
    echo "All services stopped successfully."
    exit 0
}

# Trap Ctrl+C (SIGINT) so it triggers our cleanup function
trap cleanup SIGINT

# Keep the script running and waiting for background processes
wait
