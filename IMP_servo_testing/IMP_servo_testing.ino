#include <ESP32Servo.h>

const int servoPin1 = 13;
const int servoPin2 = 12;

Servo servo1;
Servo servo2;

struct SmoothServo {
  Servo& instance;
  int currentAngle;
  int targetAngle;
  unsigned long LastMoveTime;
  unsigned long stepInterval;
};

SmoothServo s1 = {servo1, 0, 0, 0, 15};
SmoothServo s2 = {servo2, 0, 0, 0, 15};

const unsigned long totalMovementTime = 2000; // total duration you want the movementb ton take

void setup() {
  ESP32PWM::allocateTimer(0);
  ESP32PWM::allocateTimer(1);
  ESP32PWM::allocateTimer(2);
  ESP32PWM::allocateTimer(3);

  s1.instance.setPeriodHertz(50);
  s2.instance.setPeriodHertz(50);

  s1.instance.write(s1.currentAngle);
  s2.instance.write(s2.currentAngle);
  delay(1000);
}

void loop() {

  updateServo(s1);
  updateServo(s2);

  static unsigned long lastCommandTime = 0;
  static int pattern = 0;

  if (millis() - lastCommandTime >= 5000) {
    lastCommandTime = millis();
    if (pattern == 0) {
      setCoordinatedTarget(s1, 180);
      setCoordinatedTarget(s2, 45);
      pattern = 1;
    } else {
      setCoordinatedTarget(s1, 0);
      setCoordinatedTarget(s2, 0);
      pattern = 0;
    }
  }
}

// Function to calculate customised step speeds based on travel distance
void setCoordinatedTarget(SmoothServo& s, int newTarget) {
  s.targetAngle = constrain(newTarget, 0, 180); //set to 270 if using 270 deg model
  int totalDegreesToTravel = abs(s.targetAngle - s.currentAngle);
  if (totalDegreesToTravel > 0) {
    s.stepInterval = totalMovementTime / totalDegreesToTravel;
  }
}

// Independent non-blocking updater for an individual servo
void updateServo(SmoothServo& s) {
  if (millis() - s.LastMoveTime >= s.stepInterval) {
    if (s.currentAngle < s.targetAngle) {
      s.currentAngle++;
      s.instance.write(s.currentAngle);
      s.LastMoveTime = millis();
    }
    else if (s.currentAngle > s.targetAngle) {
      s.currentAngle--;
      s.instance.write(s.currentAngle);
      s.LastMoveTime = millis();
    }
  }
}