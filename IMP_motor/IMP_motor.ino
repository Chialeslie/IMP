#include <Servo.h>
#include <NewPing.h>

#include <Arduino_GigaDisplay_GFX.h>
#include <Arduino_GigaDisplayTouch.h>

// -------------------------------------------------------------
// DISPLAY INITIALIZATION & COLOR MACROS
// -------------------------------------------------------------
#define GC9A01A_CYAN    0x07FF
#define GC9A01A_RED     0xF800
#define GC9A01A_BLUE    0x001F
#define GC9A01A_GREEN   0x07E0
#define GC9A01A_MAGENTA 0xF81F
#define GC9A01A_WHITE   0xFFFF
#define GC9A01A_BLACK   0x0000
#define GC9A01A_YELLOW  0xFFE0

GigaDisplay_GFX display;

// -------------------------------------------------------------
// PROTOCOL DEFINITIONS (FRAMING & HANDSHAKE)
// -------------------------------------------------------------
#define START_MARKER 0xAA
#define STOP_MARKER  0x55
#define ACK_BYTE     0x06

typedef struct __attribute__((packed)) {
  int32_t x;
  int32_t y;
  int32_t mode;
  uint8_t checksum;
} struct_control;

struct_control incomingControl = {512, 512, 10, 0}; // x-axis, y-axis, mode, checksum

// Raspberry Pi 5 Autonomous Command String Buffer
String piCommand = "IDLE";

// -------------------------------------------------------------
// MOTOR & SENSOR PIN CONFIGURATION
// -------------------------------------------------------------
const int ENA = 4;   
const int ENB = 5;   
const int in1 = 23;
const int in2 = 25;   
const int in3 = 27;   
const int in4 = 29;   
const int LED_INDICATOR = 31;

const int TRIG_F = 22, ECHO_F = 24;
const int TRIG_L = 26, ECHO_L = 28;
const int TRIG_R = 30, ECHO_R = 32;
const int TRIG_B = 34, ECHO_B = 36;

int ObstacleFront = 0;
int ObstacleLeft  = 0;
int ObstacleRight = 0;
int ObstacleBack  = 0;
uint8_t sensorIndex = 0; // Round-robin index

const int JOYSTICK_DEADZONE = 125;
const int JOYSTICK_CENTER   = 512;

unsigned long lastSensorScan = 0;
unsigned long lastPacketReceived = 0;
const unsigned long LINK_TIMEOUT = 2000;
bool isConnected = false;

// Display UI State Tracking
bool ackTriggered = false;
bool lastAckState = false;
unsigned long ackDisplayTimer = 0;
unsigned long lastUIUpdate = 0;

// Serial Parsing State Machine
enum RxState { WAIT_START, READ_PAYLOAD, WAIT_STOP };
RxState parserState = WAIT_START;
uint8_t rxBuffer[sizeof(struct_control)];
size_t rxIndex = 0;

// -------------------------------------------------------------
// HELPER FUNCTIONS
// -------------------------------------------------------------
uint8_t calculateChecksum(const uint8_t *data, size_t length) {
  uint8_t crc = 0;
  for (size_t i = 0; i < length; i++) {
    crc ^= data[i];
  }
  return crc;
}

// Optimized Non-Blocking Sonar Distance Function (~100cm max range)
int getSensorDistance(int trigPin, int echoPin) {
  digitalWrite(trigPin, LOW);
  delayMicroseconds(4);
  digitalWrite(trigPin, HIGH);
  delayMicroseconds(10);
  digitalWrite(trigPin, LOW);
  
  // 5800us timeout restricts blocking delay to ~100cm max range
  long duration = pulseIn(echoPin, HIGH, 5800); 
  if (duration == 0) return 0;
  
  int distance = duration * 0.034 / 2;
  if (distance > 2 && distance <= 40) return 1;
  return 0;
}

// -------------------------------------------------------------
// MOTOR MOVEMENT FUNCTIONS
// -------------------------------------------------------------
void move(int i1, int i2, int i3, int i4, int speedA = 130, int speedB = 130) {
  analogWrite(ENA, speedA); 
  analogWrite(ENB, speedB);  
  digitalWrite(in1, i1);
  digitalWrite(in2, i2);
  digitalWrite(in3, i3);
  digitalWrite(in4, i4);
}

void move_forward()  { move(HIGH, LOW, HIGH, LOW); }
void move_backward() { move(LOW, HIGH, LOW, HIGH); }
void move_left()     { move(HIGH, LOW, HIGH, LOW, 130, 50); }    
void move_right()    { move(HIGH, LOW, HIGH, LOW, 50, 130); }
void move_stop()     { move(LOW, LOW, LOW, LOW, 0, 0); }

// -------------------------------------------------------------
// STATIC UI INITIALIZATION
// -------------------------------------------------------------
void setupUI() {
  display.fillScreen(GC9A01A_BLACK);

  display.fillRect(0, 0, 800, 60, GC9A01A_BLUE);
  display.setCursor(180, 18);
  display.setTextSize(3);
  display.setTextColor(GC9A01A_WHITE);
  display.print("GIGA ROBOT CONTROL SYSTEM");

  display.drawFastVLine(400, 70, 290, GC9A01A_BLUE);

  display.setTextSize(2);
  display.setTextColor(GC9A01A_WHITE);
  display.setCursor(30, 90);   display.print("LINK STATUS : ");
  display.setCursor(30, 140);  display.print("DRIVE MODE  : ");
  display.setCursor(30, 200);  display.print("JOYSTICK X  : ");
  display.setCursor(30, 250);  display.print("JOYSTICK Y  : ");

  display.setCursor(420, 90);  display.print("PI 5 CMD    : ");
  display.setCursor(420, 140); display.print("OBSTACLE SENSORS:");
  display.setCursor(420, 180); display.print("FRONT: ");
  display.setCursor(620, 180); display.print("BACK : ");
  display.setCursor(420, 230); display.print("LEFT : ");
  display.setCursor(620, 230); display.print("RIGHT: ");

  display.drawRect(30, 380, 740, 65, GC9A01A_WHITE);
  display.setCursor(50, 402);
  display.setTextSize(2);
  display.setTextColor(GC9A01A_WHITE);
  display.print("UART STATUS : LISTENING FOR TELEMETRY...");
}

// -------------------------------------------------------------
// DYNAMIC UI DRAWING ROUTINE
// -------------------------------------------------------------
void updateUI() {
  display.setTextSize(2);

  // Link Status
  display.setCursor(200, 90);
  if (isConnected) {
    display.setTextColor(GC9A01A_GREEN, GC9A01A_BLACK);
    display.print("CONNECTED   ");
  } else {
    display.setTextColor(GC9A01A_RED, GC9A01A_BLACK);
    display.print("DISCONNECTED");
  }

  // Drive Mode Display
  display.setCursor(200, 140);
  if (incomingControl.mode == 10) {
    display.setTextColor(GC9A01A_GREEN, GC9A01A_BLACK);
    display.print("MANUAL    ");
  } else if (incomingControl.mode == 11) {
    display.setTextColor(GC9A01A_YELLOW, GC9A01A_BLACK);
    display.print("AUTONOMOUS");
  } else {
    display.setTextColor(GC9A01A_RED, GC9A01A_BLACK);
    display.print("UNKNOWN   ");
  }

  // Joystick Telemetry
  display.setTextColor(GC9A01A_WHITE, GC9A01A_BLACK);
  display.setCursor(200, 200);
  display.print(incomingControl.x);
  display.print("    ");

  display.setCursor(200, 250);
  display.print(incomingControl.y);
  display.print("    ");

  // Pi 5 Autonomous Command Display
  display.setCursor(570, 90);
  display.setTextColor(GC9A01A_CYAN, GC9A01A_BLACK);
  display.print(piCommand);
  for (int i = piCommand.length(); i < 12; i++) {
    display.print(" ");
  }

  // Sensor Indicators
  display.setCursor(500, 180);
  display.setTextColor(ObstacleFront ? GC9A01A_RED : GC9A01A_GREEN, GC9A01A_BLACK);
  display.print(ObstacleFront ? "DETECTED " : "CLEAR    ");

  display.setCursor(700, 180);
  display.setTextColor(ObstacleBack ? GC9A01A_RED : GC9A01A_GREEN, GC9A01A_BLACK);
  display.print(ObstacleBack ? "DETECTED " : "CLEAR    ");

  display.setCursor(500, 230);
  display.setTextColor(ObstacleLeft ? GC9A01A_RED : GC9A01A_GREEN, GC9A01A_BLACK);
  display.print(ObstacleLeft ? "DETECTED " : "CLEAR    ");

  display.setCursor(700, 230);
  display.setTextColor(ObstacleRight ? GC9A01A_RED : GC9A01A_GREEN, GC9A01A_BLACK);
  display.print(ObstacleRight ? "DETECTED " : "CLEAR    ");

  // Bottom UART ACK Status Box (Flicker-free State Lock)
  bool currentAckActive = ackTriggered && (millis() - ackDisplayTimer < 300);
  if (currentAckActive != lastAckState) {
    lastAckState = currentAckActive;
    
    if (currentAckActive) {
      display.fillRect(30, 380, 740, 65, GC9A01A_GREEN);
      display.setCursor(50, 402);
      display.setTextColor(GC9A01A_BLACK, GC9A01A_GREEN);
      display.print("[ACK SENT TO HELTEC SLAVE : 0x06 - PACKET VALID]");
    } else {
      display.fillRect(30, 380, 740, 65, GC9A01A_BLACK);
      display.drawRect(30, 380, 740, 65, GC9A01A_WHITE);
      display.setCursor(50, 402);
      display.setTextColor(GC9A01A_WHITE, GC9A01A_BLACK);
      display.print("UART STATUS : LISTENING FOR TELEMETRY...  ");
    }
  }
}

// -------------------------------------------------------------
// NON-BLOCKING SERIAL PARSERS
// -------------------------------------------------------------
void parsePiSerial() {
  static String piBuffer = "";
  while (Serial.available() > 0) {
    char c = Serial.read();
    if (c == '\n') {
      piBuffer.trim();
      if (piBuffer.length() > 0) {
        piCommand = piBuffer;
      }
      piBuffer = "";
    } else {
      piBuffer += c;
    }
  }
}

void parseHeltecSerial() {
  while (Serial2.available() > 0) {
    uint8_t byteIn = Serial2.read();

    switch (parserState) {
      case WAIT_START:
        if (byteIn == START_MARKER) {
          rxIndex = 0;
          parserState = READ_PAYLOAD;
        }
        break;

      case READ_PAYLOAD:
        rxBuffer[rxIndex++] = byteIn;
        if (rxIndex >= sizeof(struct_control)) {
          parserState = WAIT_STOP;
        }
        break;

      case WAIT_STOP:
        if (byteIn == STOP_MARKER) {
          struct_control tempBuffer;
          memcpy(&tempBuffer, rxBuffer, sizeof(struct_control));

          // Calculate checksum up to the byte offset of the checksum field safely
          size_t payloadLength = offsetof(struct_control, checksum);
          uint8_t expectedChecksum = calculateChecksum((uint8_t*)&tempBuffer, payloadLength);

          if (tempBuffer.checksum == expectedChecksum) {
            incomingControl = tempBuffer;
            lastPacketReceived = millis();
            isConnected = true;

            digitalWrite(LED_INDICATOR, !digitalRead(LED_INDICATOR));

            Serial2.write(ACK_BYTE);
            ackTriggered = true;
            ackDisplayTimer = millis();
          }
        }
        parserState = WAIT_START;
        break;
    }
  }
}

// -------------------------------------------------------------
// SETUP
// -------------------------------------------------------------
void setup() {
  Serial.begin(115200);   
  Serial2.begin(115200);  

  pinMode(ENA, OUTPUT);
  pinMode(ENB, OUTPUT);
  pinMode(in1, OUTPUT);
  pinMode(in2, OUTPUT);
  pinMode(in3, OUTPUT);
  pinMode(in4, OUTPUT);
  pinMode(LED_INDICATOR, OUTPUT);

  pinMode(TRIG_F, OUTPUT); pinMode(ECHO_F, INPUT);
  pinMode(TRIG_L, OUTPUT); pinMode(ECHO_L, INPUT);
  pinMode(TRIG_R, OUTPUT); pinMode(ECHO_R, INPUT);
  pinMode(TRIG_B, OUTPUT); pinMode(ECHO_B, INPUT);

  move_stop();

  display.begin();
  display.setRotation(1); 

  setupUI();
  updateUI();

  Serial.println("Arduino GIGA Controller & Pi 5 Listener Ready.");
}

// -------------------------------------------------------------
// MAIN LOOP
// -------------------------------------------------------------
void loop() {
  // 1. Process Heltec & Pi Communications
  parseHeltecSerial();
  parsePiSerial();

  // 2. Link Timeout Failsafe
  if (isConnected && (millis() - lastPacketReceived > LINK_TIMEOUT)) {
    isConnected = false;
    digitalWrite(LED_INDICATOR, LOW);
    move_stop();
  }

  // 3. Non-Blocking Round-Robin Sensor Sweep (30ms per sensor)
  if (millis() - lastSensorScan >= 30) {
    lastSensorScan = millis();
    switch (sensorIndex) {
      case 0: ObstacleFront = getSensorDistance(TRIG_F, ECHO_F); sensorIndex = 1; break;
      case 1: ObstacleBack  = getSensorDistance(TRIG_B, ECHO_B); sensorIndex = 2; break;
      case 2: ObstacleLeft  = getSensorDistance(TRIG_L, ECHO_L); sensorIndex = 3; break;
      case 3: ObstacleRight = getSensorDistance(TRIG_R, ECHO_R); sensorIndex = 0; break;
    }
  }

  // 4. Direction-Selective Navigation Logic
  
  if (isConnected) {
    if (incomingControl.mode == 10) { // MANUAL MODE
      bool pushForward  = (incomingControl.y < (JOYSTICK_CENTER - JOYSTICK_DEADZONE));
      bool pushBackward = (incomingControl.y > (JOYSTICK_CENTER + JOYSTICK_DEADZONE));
      bool pushLeft     = (incomingControl.x < (JOYSTICK_CENTER - JOYSTICK_DEADZONE));
      bool pushRight    = (incomingControl.x > (JOYSTICK_CENTER + JOYSTICK_DEADZONE));

      // 1. Check obstacles first for the primary direction being requested
      if (pushForward && ObstacleFront == 1) {
        move_stop();
      } else if (pushBackward && ObstacleBack == 1) {
        move_stop();
      } else if (pushForward) {
        // Forward + Left/Right Steering Curves
        if (pushLeft && ObstacleLeft == 0)       move_left();
        else if (pushRight && ObstacleRight == 0) move_right();
        else move_forward();
      } else if (pushBackward) {
        move_backward();
      } else if (pushLeft) {
        // Standalone Spot Turn Left (Reverses left wheel, drives right wheel forward)
        if (ObstacleLeft == 0) {
          move(LOW, HIGH, HIGH, LOW, 130, 130); 
        } else {
          move_stop();
        }
      } else if (pushRight) {
        // Standalone Spot Turn Right (Drives left wheel forward, reverses right wheel)
        if (ObstacleRight == 0) {
          move(HIGH, LOW, LOW, HIGH, 130, 130); 
        } else {
          move_stop();
        }
      } else {
        move_stop();
      }
    } 
    else if (incomingControl.mode == 11) { // AUTONOMOUS MODE
      // Require explicit drive commands from Pi 5 (defaults to stop when IDLE)
      if (piCommand == "MOVE_FORWARD" && ObstacleFront == 0) {
        move_forward();
      } else if (piCommand == "MOVE_LEFT" && ObstacleLeft == 0) {
        move_left();
      } else if (piCommand == "MOVE_RIGHT" && ObstacleRight == 0) {
        move_right();
      } else {
        move_stop(); // Safely stops if piCommand is "IDLE", "STOP", or unrecognized
      }
    }
  } else {
    move_stop();
  }

  // 5. Periodic Display Refresh (150ms)
  if (millis() - lastUIUpdate >= 150) {
    lastUIUpdate = millis();
    updateUI();
  }

  // Clear ACK Status Timer
  if (ackTriggered && (millis() - ackDisplayTimer >= 300)) {
    ackTriggered = false;
  }
}