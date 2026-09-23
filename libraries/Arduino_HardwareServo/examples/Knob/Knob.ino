/*
    This file is part of the Arduino_HardwareServo library.

    Copyright (C) Arduino s.r.l. and/or its affiliated companies

    This Source Code Form is subject to the terms of the Mozilla Public
    License, v. 2.0. If a copy of the MPL was not distributed with this
    file, You can obtain one at http://mozilla.org/MPL/2.0/.

*/

#include <Arduino_HardwareServo.h>

HardwareServo myservo;  // create Servo object to control a servo

int val;    // variable to read the value from the analog pin

void setup() {
    myservo.attach(9);  // attaches the servo on pin 9 to the Servo object
}

void loop() {
    val = analogRead(A0);                // reads the value of the potentiometer connected to A0 (value between 0 and 1023)
    val = map(val, 0, 1023, 0, 180);     // scale it for use with the servo (value between 0 and 180)
    myservo.write(val);                  // sets the servo position according to the scaled value
    delay(15);                           // waits for the servo to get there
}
