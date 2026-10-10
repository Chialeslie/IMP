import time
import wave
import serial
from pathlib import Path

# --- CONFIGURATION ---
DURATION = 7          
TOTAL_SAMPLES = 5
BASE_OUTPUT_DIR = Path("authorized_voices")
BASE_OUTPUT_DIR.mkdir(exist_ok=True)

# Prompt for the speaker name
speaker_name = input("Enter the name of the speaker (e.g., chia): ").strip().lower()
if not speaker_name:
    print("[ERROR] Speaker name cannot be empty.")
    exit(1)

# Create the dedicated folder for this speaker under authorized_voices/
speaker_dir = BASE_OUTPUT_DIR / speaker_name
speaker_dir.mkdir(exist_ok=True)

# ESP32-C3 Serial Port Setup for INMP441 audio stream
PORT = "/dev/serial/by-id/usb-Espressif_USB_JTAG_serial_debug_unit_A4:CB:8F:20:E0:3C-if00"
BAUD = 921600
RATE = 16000  # 16kHz sample rate from ESP32-C3

try:
    ser = serial.Serial(PORT, BAUD, timeout=0.01)
    print(f"[INFO] Connected to ESP32-C3 Audio Node on {PORT}")
except Exception as e:
    print(f"[ERROR] Failed to open ESP32 audio stream port: {e}")
    exit(1)

print(f"\n[INFO] Starting batch recording of {TOTAL_SAMPLES} samples for '{speaker_name}' via INMP441...")

for sample_num in range(1, TOTAL_SAMPLES + 1):
    filename = speaker_dir / f"sample_{sample_num}.wav"
    
    print(f"\n--- Recording Sample {sample_num} of {TOTAL_SAMPLES} ---")
    for i in range(3, 0, -1):
        print(f"[INFO] Recording starts in {i}...")
        time.sleep(1)

    print("🔴 RECORDING LIVE... Speak your commands into the INMP441 now!")

    frames = []
    target_bytes = RATE * 2 * DURATION  # 16000 samples/sec * 2 bytes/sample * 7 seconds
    bytes_collected = 0
    
    # Clear any stale buffer data before starting
    ser.reset_input_buffer()

    start_time = time.time()
    while bytes_collected < target_bytes and (time.time() - start_time) < (DURATION + 2):
        chunk = ser.read(512)
        if chunk:
            frames.append(chunk)
            bytes_collected += len(chunk)
            # Print a dot progress indicator roughly every second
            if len(frames) % 62 == 0:
                print(".", end="", flush=True)

    print("\n[SUCCESS] Sample captured from ESP32-C3!")

    # Save the file into the speaker's subfolder
    wf = wave.open(str(filename), 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(2) # 16-bit PCM
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

    print(f"[SUCCESS] Saved as '{filename}'")
    
    if sample_num < TOTAL_SAMPLES:
        input("\nPress [Enter] when ready for the next sample...")

ser.close()
print(f"\n✨ All {TOTAL_SAMPLES} voice samples successfully recorded and saved for '{speaker_name}' via the INMP441!")