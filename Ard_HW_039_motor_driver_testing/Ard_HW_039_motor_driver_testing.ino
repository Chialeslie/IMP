#include <Arduino.h>

const int R_EN = 23;   
const int R_RPWM = 4;
const int R_LPWM = 5;   
const int L_EN = 27;
const int L_RPWM = 6;   
const int L_LPWM = 7;   

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


void setup() {

  pinMode(R_EN, OUTPUT); 
  pinMode(L_EN, OUTPUT);
  pinMode(R_RPWM, OUTPUT); 
  pinMode(R_LPWM, OUTPUT);
  pinMode(L_RPWM, OUTPUT); 
  pinMode(L_LPWM, OUTPUT);

}

void loop() {
  move_stop();
  delay(2000);
  move_forward();
  delay(2000);
  move_stop();
  delay(1000);
  move_backward();
  delay(2000);
  move_stop();
  delay(1000);
  move_left();
  delay(2000);
  move_stop();
  delay(1000);
  move_right();
  delay(2000);
}
