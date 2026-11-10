// Rocky sonar, NO-SERVO version: the sensor stays still and the shape is turned by hand.
// Wiring (4 female-to-male wires, nothing else): sensor VCC->5V, Trig->D9, Echo->D10, GND->GND
// Protocol: send the character 'R' from the PC; for REC_MS milliseconds the Arduino prints "time_ms,distance_cm" lines, then prints END.
const uint8_t TRIG = 9, ECHO = 10;
const unsigned long REC_MS = 10000;            // length of one recording (keep REC_S in rocky_rotate.py equal to this / 1000)
const float TEMP_C = 22.0;                     // room temperature, sets the speed of sound

float pingOnce() {                             // distance in cm, or -1 when no echo comes back
  digitalWrite(TRIG, LOW);  delayMicroseconds(3);
  digitalWrite(TRIG, HIGH); delayMicroseconds(10);
  digitalWrite(TRIG, LOW);
  unsigned long us = pulseIn(ECHO, HIGH, 25000UL);          // 25 ms timeout = about 4 m
  if (us == 0) return -1;
  float cm = us * (331.3 + 0.606 * TEMP_C) / 10000.0 / 2.0; // round trip, so divide by 2
  return (cm < 2 || cm > 400) ? -1 : cm;
}

void setup() {
  Serial.begin(115200);
  pinMode(TRIG, OUTPUT); pinMode(ECHO, INPUT);
  digitalWrite(TRIG, LOW);
  Serial.println("READY");
}

void loop() {
  if (Serial.available() && Serial.read() == 'R') {
    unsigned long t0 = millis();
    while (millis() - t0 < REC_MS) {
      Serial.print(millis() - t0); Serial.print(','); Serial.println(pingOnce(), 1);
      delay(35);                               // about 15-25 readings per second
    }
    Serial.println("END");
  }
}
