#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>

Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver(0x40, Wire1);

// --- PCA9685 CHANNEL MAPPING (3 Servos per Limb: 12 Servos Total) ---
const int FR_HIP      = 0;
const int FR_SHOULDER = 1;
const int FR_KNEE     = 2;

const int FL_HIP      = 4;
const int FL_SHOULDER = 5;
const int FL_KNEE     = 6;

const int BR_HIP      = 8;
const int BR_SHOULDER = 9;
const int BR_KNEE     = 10;

const int BL_HIP      = 12;
const int BL_SHOULDER = 13;
const int BL_KNEE     = 14;

// --- NEUTRAL STANDING PWM VALUES (Calibrated for 270° RDS3225 Ticks) ---
const int NEUTRAL_HIP      = 460; // 225 degrees neutral center
const int NEUTRAL_SHOULDER = 307; 
const int NEUTRAL_KNEE     = 380; // Standing extension height

// --- STAIR CLIMBING & HIP PARAMETERS ---
const int HIP_SWING_AMPLITUDE   = 25;  // Active hip sway to shift weight during high lifts
const int SWING_AMPLITUDE       = 25;  // Forward/backward reach for stepping
const int STAIR_LIFT_HEIGHT     = 120; // High vertical knee clearance for tall steps

void setAllLegs(int hipPWM, int shoulderPWM, int kneePWM) {
  int channels[] = {
    FR_HIP, FR_SHOULDER, FR_KNEE,
    FL_HIP, FL_SHOULDER, FL_KNEE,
    BR_HIP, BR_SHOULDER, BR_KNEE,
    BL_HIP, BL_SHOULDER, BL_KNEE
  };
  
  for (int i = 0; i < 12; i += 3) {
    pwm.setPWM(channels[i],   0, hipPWM);
    pwm.setPWM(channels[i+1], 0, shoulderPWM);
    pwm.setPWM(channels[i+2], 0, kneePWM);
  }
}

void setup() {
  Serial.begin(115200);
  Wire1.begin();
  pwm.begin();
  pwm.setPWMFreq(50);  
  delay(1000);

  // Initialize robot into a neutral stance
  setAllLegs(NEUTRAL_HIP, NEUTRAL_SHOULDER, NEUTRAL_KNEE);
  delay(1000);
  Serial.println("3-DOF RDS3225 High-Clearance Stair Climbing Initialized.");
}

void loop() {
  // Diagonal pairs execute the climbing phase with coordinated hip shifts
  executeStairPhase(FR_HIP, FR_SHOULDER, FR_KNEE,  BL_HIP, BL_SHOULDER, BL_KNEE,  // Moving Pair
                    FL_HIP, FL_SHOULDER, FL_KNEE,  BR_HIP, BR_SHOULDER, BR_KNEE); // Supporting Pair

  executeStairPhase(FL_HIP, FL_SHOULDER, FL_KNEE,  BR_HIP, BR_SHOULDER, BR_KNEE,  // Moving Pair
                    FR_HIP, FR_SHOULDER, FR_KNEE,  BL_HIP, BL_SHOULDER, BL_KNEE); // Supporting Pair
}

// --- CORE FUNCTIONS ---

void executeStairPhase(int m1H, int m1S, int m1K, int m2H, int m2S, int m2K,
                       int s1H, int s1S, int s1K, int s2H, int s2S, int s2K) {
  int steps = 25; // Higher resolution for smooth, stable heavy-load movement

  for (int i = 0; i <= steps; i++) {
    float p = (float)i / steps;
    
    // High vertical arc for clearing tall step obstacles
    int stairLift = sin(p * 3.14159) * STAIR_LIFT_HEIGHT;

    // Moving pair trajectories (reaching up and forward)
    int moveShoulder = (NEUTRAL_SHOULDER - SWING_AMPLITUDE) + (SWING_AMPLITUDE * 2.0 * p);
    int moveKnee     = NEUTRAL_KNEE - stairLift; 
    
    // Active Hip Motion: Hips swing/tilt outward to accommodate the lift
    int moveHip      = NEUTRAL_HIP + (sin(p * 3.14159) * HIP_SWING_AMPLITUDE);

    // Supporting pair trajectories (holding weight and pushing up)
    int supportShoulder = (NEUTRAL_SHOULDER + SWING_AMPLITUDE) - (SWING_AMPLITUDE * 2.0 * p);
    int supportKnee     = NEUTRAL_KNEE; 
    
    // Counter Hip Motion: Hips shift in the opposite direction to balance body weight
    int supportHip      = NEUTRAL_HIP - (sin(p * 3.14159) * HIP_SWING_AMPLITUDE);

    // Apply PWM to Moving Pair 1
    pwm.setPWM(m1H, 0, moveHip);
    pwm.setPWM(m1S, 0, moveShoulder);
    pwm.setPWM(m1K, 0, moveKnee);

    // Apply PWM to Moving Pair 2
    pwm.setPWM(m2H, 0, moveHip);
    pwm.setPWM(m2S, 0, moveShoulder);
    pwm.setPWM(m2K, 0, moveKnee);

    // Apply PWM to Supporting Pair 1
    pwm.setPWM(s1H, 0, supportHip);
    pwm.setPWM(s1S, 0, supportShoulder);
    pwm.setPWM(s1K, 0, supportKnee);

    // Apply PWM to Supporting Pair 2
    pwm.setPWM(s2H, 0, supportHip);
    pwm.setPWM(s2S, 0, supportShoulder);
    pwm.setPWM(s2K, 0, supportKnee);

    delay(20); // Controlled delay for high-torque servo stability
  }
}
