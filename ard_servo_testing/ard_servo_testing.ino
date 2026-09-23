#include <mbed.h>

// Define hardware PWM pins for the GIGA R1 (Pins 9 and 10)
mbed::PwmOut panServo(digitalPinToPinName(9));   
mbed::PwmOut tiltServo(digitalPinToPinName(10)); 

// Helper function mirroring your ESP32 mapping logic
void moveServo(mbed::PwmOut &servo, int degrees, int maxDegrees) {
  int pulse = map(degrees, 0, maxDegrees, 500, 2500);
  servo.pulsewidth_us(pulse); // Set pulse width directly in microseconds
}

void setup() {
  Serial.begin(115200);
  delay(1000);

  // Set standard 50Hz servo frequency (20ms period)
  panServo.period_ms(20);
  tiltServo.period_ms(20);

  Serial.println("GIGA Hardware PWM Servo Test Initialized.");
  
  // Move to initial positions
  moveServo(panServo, 0, 270);
  moveServo(tiltServo, 0, 180);
  delay(1000);
}

void loop() {
  Serial.println("Sweeping Pan (0 -> 135 -> 270 -> 0)");
  moveServo(panServo, 0, 270);
  delay(2000);
  moveServo(panServo, 135, 270);
  delay(2000);
  moveServo(panServo, 270, 270);
  delay(2000);
  moveServo(panServo, 0, 270);
  delay(2000);

  Serial.println("Sweeping Tilt (0 -> 90 -> 180 -> 0)");
  moveServo(tiltServo, 0, 180);
  delay(2000);
  moveServo(tiltServo, 90, 180);
  delay(2000);
  moveServo(tiltServo, 180, 180);
  delay(2000);
  moveServo(tiltServo, 0, 180);
  delay(2000);
}