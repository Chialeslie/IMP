#include <Servo.h>
#include <NewPing.h>
#include <TinyGPS++.h>
#include <Wire.h>
#include <Adafruit_QMC5883P.h>
#include <RPC.h>
#include <Adafruit_Sensor.h>
#include <mbed.h>
#include <Arduino_GigaDisplay_GFX.h>
#include <Arduino_GigaDisplayTouch.h>
#include <Arduino.h>

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
// SENSOR MODULES INITIALIZATION
// -------------------------------------------------------------
TinyGPSPlus gps;
Adafruit_QMC5883P compass = Adafruit_QMC5883P();

struct GPSData {
  double latitude;
  double longitude;
  uint8_t satellites;
  bool fix;
} currentGPS = {0.0, 0.0, 0, false};

int currentHeading = 0; 
float imuRoll = 0;
float imuPitch = 0;
float imuGyroZ = 0;

// -------------------------------------------------------------
// PROTOCOL DEFINITIONS (UNIFIED FRAMING)
// -------------------------------------------------------------
#define START_MARKER 0x3C
#define STOP_MARKER  0x3E

typedef struct __attribute__((packed)) struct_control {
  int32_t x;
  int32_t y;
  int32_t mode;
  int32_t counter;
} struct_control;

struct_control incomingControl = {512, 512, 11, 0}; 

unsigned long lastTelemetrySend = 0;
unsigned long lastPiPacketReceived = 0;
// -------------------------------------------------------------
// MOTOR AND SENSOR PIN CONFIGURATION
// -------------------------------------------------------------
const int R_EN = 4;   
const int R_RPWM = 5;
const int R_LPWM = 6;   
const int L_EN = 7;
const int L_RPWM = 8;   
const int L_LPWM = 9;   
const int LED_INDICATOR = 31;

const int TRIG_F = 22, ECHO_F = 24;
const int TRIG_L = 26, ECHO_L = 28;
const int TRIG_R = 30, ECHO_R = 32;
const int TRIG_B = 34, ECHO_B = 36;

int ObstacleFront = 0;
int ObstacleLeft  = 0;
int ObstacleRight = 0;
int ObstacleBack  = 0;
uint8_t sensorIndex = 0; 

const int JOYSTICK_DEADZONE = 125;
const int JOYSTICK_CENTER   = 512;

int currentPanAngle = 135;
int currentTiltAngle = 90;

unsigned long lastSensorScan = 0;
unsigned long lastCompassScan = 0;
unsigned long lastImuScan = 0;
unsigned long lastPacketReceived = 0;
const unsigned long LINK_TIMEOUT = 2000;
bool isConnected = false;

// Framed Serial Parsing State Machine
enum RxState { WAIT_START, READ_PAYLOAD, WAIT_STOP };
RxState parserState = WAIT_START;
uint8_t rxBuffer[sizeof(struct_control)];
size_t rxIndex = 0;

// Autonomous State Variables
String piCommand = "IDLE";
bool piEmergencyStop = false;
int targetSteeringError = 0; 
int targetBaseSpeed = 0;
unsigned long lastUIUpdate = 0;
String piStatus = "NONE";

// Servo Declaration
Servo panServo;
Servo tiltServo;
const int Pan = 2;
const int Tilt = 3;

// HELPER FUNCTIONS FOR ULTRASONIC SENSORS

int getSensorDistance(int trigPin, int echoPin) {
  digitalWrite(trigPin, LOW);
  delayMicroseconds(4);
  digitalWrite(trigPin, HIGH);
  delayMicroseconds(10);
  digitalWrite(trigPin, LOW);
  
  long duration = pulseIn(echoPin, HIGH, 5800); 
  if (duration == 0) return 0;
  
  int distance = duration * 0.034 / 2;
  if (distance > 2 && distance <= 40) return 1;
  return 0;
}
// HW-039 motor driver function
void move(int RMotorP, int RMotorN, int LMotorP, int LMotorN) {
  digitalWrite(R_EN, HIGH); 
  digitalWrite(L_EN, HIGH);  
  analogWrite(R_RPWM, RMotorP);
  analogWrite(R_LPWM, RMotorN);
  analogWrite(L_RPWM, LMotorP);
  analogWrite(L_LPWM, LMotorN);
}
void move_forward(int Speed = 200) { move(Speed, 0, Speed, 0); }
void move_backward(int Speed = 200) { move(0, Speed, 0, Speed); }
void move_left(int SpeedR = 200, int SpeedL = 100) { move(SpeedR, 0, SpeedL, 0); }    
void move_right(int SpeedR = 100, int SpeedL = 200) { move(SpeedR, 0, SpeedL, 0); }
void move_stop() { move(0, 0, 0, 0); }
void turn_spot_left(int Speed = 200)  { move(0, Speed, Speed, 0); }
void turn_spot_right(int Speed = 200) { move(Speed, 0, 0, Speed); }

// HW-039 motor driver for tracking 
void TrackingSpeed (int leftSpeed, int rightSpeed) {
  int l_rpwm = 0, l_lpwm = 0;
  int r_rpwm = 0, r_lpwm = 0;

  // Left Motor Direction & Speed
  if (leftSpeed > 0) {
    l_rpwm = constrain(leftSpeed, 0, 255);
    l_lpwm = 0;
  } else if (leftSpeed < 0) {
    l_rpwm = 0;
    l_lpwm = constrain(abs(leftSpeed), 0, 255);
  }

  // Right Motor Direction & Speed
  if (rightSpeed > 0) {
    r_rpwm = constrain(rightSpeed, 0, 255);
    r_lpwm = 0;
  } else if (rightSpeed < 0) {
    r_rpwm = 0;
    r_lpwm = constrain(abs(rightSpeed), 0, 255);
  }

  move(r_rpwm, r_lpwm, l_rpwm, l_lpwm);
}

// servo function
void moveServo(Servo &s, int degrees, int maxDegrees) {
  int pulse = map(degrees, 0, maxDegrees, 500, 2500);
  s.writeMicroseconds(pulse);
}

// STATIC UI INITIALIZATION
void setupUI() {
  display.fillScreen(GC9A01A_BLACK);
  display.fillRect(0, 0, 800, 60, GC9A01A_BLUE);
  display.setCursor(110, 18);
  display.setTextSize(3);
  display.setTextColor(GC9A01A_WHITE);
  display.print("ROBOT CONTROL SYSTEM INFORMATION");

  display.drawFastVLine(400, 70, 290, GC9A01A_BLUE);

  display.setTextSize(2);
  display.setTextColor(GC9A01A_WHITE);
  display.setCursor(30, 75);  display.print("LORA STATUS : ");
  display.setCursor(30, 105); display.print("DRIVE MODE  : ");
  display.setCursor(30, 135); display.print("JOYSTICK X  : ");
  display.setCursor(30, 165); display.print("JOYSTICK Y  : ");
  
  display.drawFastHLine(30, 195, 350, GC9A01A_BLUE);
  display.setCursor(30, 210); display.print("GPS LAT : ");
  display.setCursor(30, 240); display.print("GPS LON : ");
  display.setCursor(30, 270); display.print("SATS/LCK: ");
  display.setCursor(30, 300); display.print("COMPASS : ");
  display.setCursor(30, 330); display.print("IMU ROLL: ");
  display.setCursor(30, 355); display.print("IMU PCH : ");

  display.setCursor(420, 75); display.print("PI 5 CMD   : ");
  display.setCursor(420, 110); display.print("TRACKING   : ");
  display.setCursor(420, 145); display.print("OBSTACLE SENSORS:");
  display.setCursor(420, 180); display.print("FRONT: ");
  display.setCursor(620, 180); display.print("BACK : ");
  display.setCursor(420, 215); display.print("LEFT : ");
  display.setCursor(620, 215); display.print("RIGHT: ");

  display.drawRect(30, 380, 740, 65, GC9A01A_WHITE);
  display.setCursor(50, 402);
  display.setTextSize(2);
  display.setTextColor(GC9A01A_WHITE);
  display.print("UART STATUS : LISTENING FOR TELEMETRY...");
}

// DYNAMIC UI DRAWING ROUTINE
void updateUI() {
  display.setTextSize(2);
  display.setCursor(200, 75);
  if (isConnected) {
    display.setTextColor(GC9A01A_GREEN, GC9A01A_BLACK);
    display.print("CONNECTED   ");
  } else {
    display.setTextColor(GC9A01A_RED, GC9A01A_BLACK);
    display.print("DISCONNECTED");
  }

  display.setCursor(200, 105);
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

  display.setTextColor(GC9A01A_WHITE, GC9A01A_BLACK);
  display.setCursor(200, 135);
  display.print(incomingControl.x); display.print("    ");

  display.setCursor(200, 165);
  display.print(incomingControl.y); display.print("    ");

  if (currentGPS.fix) {
    display.setTextColor(GC9A01A_CYAN, GC9A01A_BLACK);
    display.setCursor(150, 210); display.print(currentGPS.latitude, 6);  display.print("  ");
    display.setCursor(150, 240); display.print(currentGPS.longitude, 6); display.print("  ");
    display.setTextColor(GC9A01A_GREEN, GC9A01A_BLACK);
    display.setCursor(150, 270); display.print(currentGPS.satellites); display.print(" (3D FIX) ");
  } else {
    display.setTextColor(GC9A01A_YELLOW, GC9A01A_BLACK);
    display.setCursor(150, 210); display.print("SEARCHING...  ");
    display.setCursor(150, 240); display.print("SEARCHING...  ");
    display.setCursor(150, 270); display.print(gps.satellites.value()); display.print(" (NO LOCK)  ");
  }

  display.setTextColor(GC9A01A_MAGENTA, GC9A01A_BLACK);
  display.setCursor(150, 300);
  display.print(currentHeading); display.print(" deg   ");

  display.setCursor(150, 330);
  display.print(imuRoll, 1); display.print(" deg   ");

  display.setCursor(150, 355);
  display.print(imuPitch, 1); display.print(" deg   ");

  display.setCursor(570, 75);
  display.setTextColor(GC9A01A_CYAN, GC9A01A_BLACK);
  display.print(piCommand);
  for (size_t i = piCommand.length(); i < 12; i++) {
    display.print(" ");
  }
  display.setCursor(570, 110);
  display.setTextColor(GC9A01A_MAGENTA, GC9A01A_BLACK);
  display.print(piStatus);
  for (size_t i = piStatus.length(); i < 12; i++) {
    display.print(" ");
  }

  display.setCursor(500, 180);
  display.setTextColor(ObstacleFront ? GC9A01A_RED : GC9A01A_GREEN, GC9A01A_BLACK);
  display.print(ObstacleFront ? "DETECTED " : "CLEAR    ");

  display.setCursor(700, 180);
  display.setTextColor(ObstacleBack ? GC9A01A_RED : GC9A01A_GREEN, GC9A01A_BLACK);
  display.print(ObstacleBack ? "DETECTED " : "CLEAR    ");

  display.setCursor(500, 215);
  display.setTextColor(ObstacleLeft ? GC9A01A_RED : GC9A01A_GREEN, GC9A01A_BLACK);
  display.print(ObstacleLeft ? "DETECTED " : "CLEAR    ");

  display.setCursor(700, 215);
  display.setTextColor(ObstacleRight ? GC9A01A_RED : GC9A01A_GREEN, GC9A01A_BLACK);
  display.print(ObstacleRight ? "DETECTED " : "CLEAR    ");
}

// SENSORS & SERIAL PARSERS
void parseGPS() {
  while (Serial1.available() > 0) {
    gps.encode(Serial1.read());
  }
  currentGPS.satellites = gps.satellites.value();
  if (gps.location.isValid()) {
    currentGPS.latitude = gps.location.lat();
    currentGPS.longitude = gps.location.lng();
    currentGPS.fix = true;
  } else {
    currentGPS.fix = false;
  }
}

void parseCompass() {
  if (millis() - lastCompassScan >= 100) { 
    lastCompassScan = millis();
    Wire.beginTransmission(0x2C);
    Wire.write(0x01); 
    Wire.endTransmission(false);
    
    byte bytesRequested = Wire.requestFrom(0x2C, 6, true);
    if (bytesRequested >= 6) {
      int16_t x = (int16_t)(Wire.read() | (Wire.read() << 8));
      int16_t y = (int16_t)(Wire.read() | (Wire.read() << 8));
      int16_t z = (int16_t)(Wire.read() | (Wire.read() << 8));

      if (x != 0 || y != 0) {
        float calcHeading = atan2((float)y, (float)x) * 180.0 / PI;
        if (calcHeading < 0) calcHeading += 360.0;
        currentHeading = (int)calcHeading;
      }
    }
  }
}

void parseIMU() {
  if (millis() - lastImuScan >= 50) { 
    lastImuScan = millis();
    Wire.beginTransmission(0x68);
    Wire.write(0x3B);
    byte writeErr = Wire.endTransmission(true);

    if (writeErr == 0) {
      byte bytesReceived = Wire.requestFrom(0x68, 6, true);
      if (bytesReceived == 6) {
        int16_t ax = (int16_t)(Wire.read() << 8 | Wire.read());
        int16_t ay = (int16_t)(Wire.read() << 8 | Wire.read());
        int16_t az = (int16_t)(Wire.read() << 8 | Wire.read());

        imuRoll  = atan2(-ay, sqrt((long)ax * ax + (long)az * az)) * 180.0 / PI;
        imuPitch = atan2(ax, sqrt((long)ay * ay + (long)az * az)) * 180.0 / PI;
      }
    }
  }
}

void parsePiSerial() {
  while (SerialUSB.available() > 0) {
    lastPiPacketReceived = millis();
    uint8_t header = SerialUSB.peek();
    // --- USE HEADER BYTE TO DIFFERENTIATE THE MESSAGE RECEIVED FROM PI ---
    if (header == 0xFF) {
      SerialUSB.read();
      lastPiPacketReceived = millis();
      piEmergencyStop = true;
      piCommand = "EMERGENCY";
      move_stop();
    } 
    else if (header == 0x01) {
      SerialUSB.read();
      lastPiPacketReceived = millis();
      piEmergencyStop = false;
      piStatus = "TRACKING";
      piCommand = "TRACKING";
      unsigned long startTime = millis();
      while (SerialUSB.available() < 2 && millis() - startTime < 50) {}
      
      if (SerialUSB.available() >= 2) {
        targetSteeringError = SerialUSB.read() - 100;
        targetBaseSpeed = SerialUSB.read();
      }
    }
    else if (header == 0x02) {
      SerialUSB.read();
      lastPiPacketReceived = millis();
      unsigned long startTime = millis();
      while (SerialUSB.available() < 4 && millis() - startTime < 50) {}
      
      if (SerialUSB.available() >= 4) {
        uint8_t rawPanByte = SerialUSB.read();
        uint8_t rawTiltByte = SerialUSB.read();
        uint8_t rawErrorByte = SerialUSB.read();
        uint8_t rawSpeedByte = SerialUSB.read();

        currentPanAngle = constrain(rawPanByte, 30, 240);
        currentTiltAngle = constrain(rawTiltByte, 0, 180);
        moveServo(panServo, currentPanAngle, 270);
        moveServo(tiltServo, currentTiltAngle, 180);

        piEmergencyStop = false;
        piStatus = "TRACKING";
        piCommand = "TRACKING";
        targetSteeringError = rawErrorByte - 100;
        targetBaseSpeed = rawSpeedByte;
      }
    }

    else if (header == 0x03) {
      SerialUSB.read();
      lastPiPacketReceived = millis();
      String textCmd = SerialUSB.readStringUntil('\n');
      textCmd.trim();
      if (textCmd.length() > 0) {
        lastPiPacketReceived = millis();
        int colonIndex = textCmd.indexOf(':');
        if (colonIndex != -1) {
          String source = textCmd.substring(0, colonIndex);
          String cmd = textCmd.substring(colonIndex + 1);
          source.toUpperCase();
          cmd.toUpperCase();
          piCommand = cmd;
          piStatus = source;
          if (cmd == "FORWARD") move_forward();
          else if (cmd == "BACKWARD") move_backward();
          else if (cmd == "STOP") move_stop();
          else if (cmd == "LEFT") move_left();
          else if (cmd == "RIGHT") move_right();
        } else {
          textCmd.toUpperCase();
          piCommand = textCmd;
          piStatus = "DIRECT";
          if (textCmd == "FORWARD") move_forward();
          else if (textCmd == "BACKWARD") move_backward();
          else if (textCmd == "STOP") move_stop();
          else if (textCmd == "LEFT") move_left();
          else if (textCmd == "RIGHT") move_right();
        }
      }
    }
    else {
      SerialUSB.read();
    }
  }
}
void parseSlaveSerial() {
  while (Serial2.available() > 0) {
    uint8_t incomingByte = Serial2.read();

    switch (parserState) {
      case WAIT_START:
        if (incomingByte == START_MARKER) {
          rxIndex = 0; 
          parserState = READ_PAYLOAD;
        }
        break;

      case READ_PAYLOAD:
        rxBuffer[rxIndex++] = incomingByte;
        if (rxIndex >= sizeof(struct_control)) {
          parserState = WAIT_STOP;
        }
        break;

      case WAIT_STOP:
        if (incomingByte == STOP_MARKER) {
          struct_control *tempControl = (struct_control*)rxBuffer;
          if (tempControl->mode == 10 || tempControl->mode == 11) {
            incomingControl = *tempControl;
            lastPacketReceived = millis();
            isConnected = true;
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
  Serial1.begin(9600);    
  Serial2.begin(115200);
  SerialUSB.begin(115200);  

  panServo.attach(Pan, 500, 2500);
  tiltServo.attach(Tilt, 500, 2500);
  moveServo(panServo, 135, 270);
  moveServo(tiltServo, 90, 180);

  Wire.begin();
  Wire.setClock(100000); 
  delay(200); 
  
  // Initialize MPU6050
  Wire.beginTransmission(0x68);
  Wire.write(0x6B); Wire.write(0x00); Wire.endTransmission(true);
  delay(50); 

  // Initialize QMC5883P Compass
  Wire.beginTransmission(0x2C); Wire.write(0x29); Wire.write(0x06); Wire.endTransmission(); delay(10);
  Wire.beginTransmission(0x2C); Wire.write(0x0B); Wire.write(0x08); Wire.endTransmission(); delay(10);
  Wire.beginTransmission(0x2C); Wire.write(0x0A); Wire.write(0xCD); Wire.endTransmission(); delay(50);

  pinMode(R_EN, OUTPUT); pinMode(L_EN, OUTPUT);
  pinMode(R_RPWM, OUTPUT); pinMode(R_LPWM, OUTPUT);
  pinMode(L_RPWM, OUTPUT); pinMode(L_LPWM, OUTPUT);
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
}

// -------------------------------------------------------------
// MAIN LOOP
// -------------------------------------------------------------
void loop() {
  parseSlaveSerial();
  parsePiSerial();
  parseGPS();
  parseCompass();
  parseIMU();

  // Send Framed Telemetry to Slave every 200ms
  if (millis() - lastTelemetrySend >= 200) {
    lastTelemetrySend = millis();
    
    // Define exact 12-byte struct matching the ESP32 slave side
    typedef struct __attribute__((packed)) struct_gps {
      float latitude;      // 4 bytes
      float longitude;     // 4 bytes
      uint8_t satellites;  // 1 byte
      bool fix;            // 1 byte
      int16_t heading;     // 2 bytes
      uint8_t panAngle;
      uint8_t tiltAngle;
    } struct_gps;

    struct_gps telemetryData;
    telemetryData.latitude = (float)currentGPS.latitude;
    telemetryData.longitude = (float)currentGPS.longitude;
    telemetryData.satellites = (uint8_t)currentGPS.satellites;
    telemetryData.fix = (bool)currentGPS.fix;
    telemetryData.heading = (int16_t)currentHeading;
    telemetryData.panAngle = (uint8_t)currentPanAngle; 
    telemetryData.tiltAngle = (uint8_t)currentTiltAngle;
    // send GPS coordinate to slave
    Serial2.write(START_MARKER);
    Serial2.write((uint8_t*)&telemetryData, sizeof(struct_gps)); // Sends precisely 12 bytes
    Serial2.write(STOP_MARKER);
    
  }
  
  if (isConnected && (millis() - lastPacketReceived > LINK_TIMEOUT)) {
    isConnected = false;
    digitalWrite(LED_INDICATOR, LOW);
    incomingControl.mode = 11; 
  }

  if (millis() - lastSensorScan >= 30) {
    lastSensorScan = millis();
    switch (sensorIndex) {
      case 0: ObstacleFront = getSensorDistance(TRIG_F, ECHO_F); sensorIndex = 1; break;
      case 1: ObstacleBack  = getSensorDistance(TRIG_B, ECHO_B); sensorIndex = 2; break;
      case 2: ObstacleLeft  = getSensorDistance(TRIG_L, ECHO_L); sensorIndex = 3; break;
      case 3: ObstacleRight = getSensorDistance(TRIG_R, ECHO_R); sensorIndex = 0; break;
    }
  }

  if (incomingControl.mode == 10) { 
    if (isConnected) {
      bool pushForward  = (incomingControl.y < (JOYSTICK_CENTER - JOYSTICK_DEADZONE));
      bool pushBackward = (incomingControl.y > (JOYSTICK_CENTER + JOYSTICK_DEADZONE));
      bool pushLeft     = (incomingControl.x < (JOYSTICK_CENTER - JOYSTICK_DEADZONE));
      bool pushRight    = (incomingControl.x > (JOYSTICK_CENTER + JOYSTICK_DEADZONE));

      if (pushLeft && !pushForward && !pushBackward) {
        turn_spot_left();
      } else if (pushRight && !pushForward && !pushBackward) {
        turn_spot_right();
      } else if (pushForward && ObstacleFront == 1) {
        move_stop();
      } else if (pushBackward && ObstacleBack == 1) {
        move_stop();
      } else if (pushForward) {
        move_forward();
      } else if (pushBackward) {
        move_backward();
      } else {
        move_stop();
      }
    } else {
      move_stop();
    }
  } 
  else if (incomingControl.mode == 11) { 
    if (piEmergencyStop || ObstacleFront == 1) {
      move_stop();
    } 
    else if (piCommand == "TRACKING") {
      int leftSpeed = constrain(targetBaseSpeed - targetSteeringError, -255, 255);
      int rightSpeed = constrain(targetBaseSpeed + targetSteeringError, -255, 255);
      TrackingSpeed(leftSpeed, rightSpeed);
    } 
    else if (piCommand == "FORWARD") {
      move_forward();
    }
    else if (piCommand == "BACKWARD") {
      move_backward();
    }
    else if (piCommand == "LEFT") {
      move_left();
    }
    else if (piCommand == "RIGHT") {
      move_right();
    }
    else if (piCommand == "STOP" || piCommand == "IDLE") {
      move_stop();
    }
    else {
      move_stop();
    }
  }
  if (millis() - lastPiPacketReceived > LINK_TIMEOUT) {
    if (piCommand != "IDLE" && piCommand != "DISCONNECTED") {
      piCommand = "IDLE";
      piStatus = "NONE";
      move_stop();
    }
  }

  if (millis() - lastUIUpdate >= 200) {
    lastUIUpdate = millis();
    updateUI();
  }
}