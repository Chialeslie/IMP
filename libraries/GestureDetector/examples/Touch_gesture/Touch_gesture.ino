#include "Arduino_GigaDisplay_GFX.h"
#include "GestureDetector.h"

// Example program for Giga gesture detector library and Giga GFX.

// Uses startBuffering and endBuffering calls to batch up
// screen updates to prevent flicker and draw more smoothly.
// These can be found in the gilesp1729 fork of
// Arduino_GigaDisplay_GFX. THe sketch works fine with the
// original GFX if these calls are commented out; it just
// flickers a bit more.

// Construct the graphics and gesture libs
GestureDetector detector;
GigaDisplay_GFX tft;

// Coordinates of initial rect
int box_x, box_y, box_w, box_h;

// 4 corners of rect (which may be rotated)
Point cp[4];

// Show a message on the screen.
void Log(char *str, int x = 0, int y = 0)
{
  tft.fillRect(x, y, 50*strlen(str), 50, 0);
  tft.setCursor(x, y);
  tft.print(str);
  Serial.println(str);
}

// Draw a rectangle made up from 4 lines. (done this way so rect can be rotated)
void draw_rect(Point p0, Point p1, Point p2, Point p3)
{
  tft.drawLine(p0.x, p0.y, p1.x, p1.y, 0xFFFF);
  tft.drawLine(p1.x, p1.y, p2.x, p2.y, 0xFFFF);
  tft.drawLine(p2.x, p2.y, p3.x, p3.y, 0xFFFF);
  tft.drawLine(p3.x, p3.y, p0.x, p0.y, 0xFFFF);
}

// Reset the rect to the middle of the screen.
void reset(void)
{
  tft.fillScreen(0);
  box_x = tft.width() / 4;
  box_y = tft.height() / 4;
  box_w = tft.width() / 2;
  box_h = tft.height() / 2;
  cp[0] = Point(box_x, box_y);
  cp[1] = Point(box_x + box_w, box_y);
  cp[2] = Point(box_x + box_w, box_y + box_h);
  cp[3] = Point(box_x, box_y + box_h);

  // Redraw the box
  draw_rect(cp[0], cp[1], cp[2], cp[3]);
}

void tap_cb(EventType ev, int indx, void *param, int x, int y)
{
  if ((ev & EV_RELEASED) == 0)
    return;   // we only act on the releases
    
  if (ev & EV_LONG_PRESS)
  {
    tft.fillCircle(x, y, 20, 0xFFFF);   // long press
    Log("Long press");
  }
  else    
  {
    tft.drawCircle(x, y, 20, 0xFFFF);
    Log("Tap");
  }
}

void drag_cb_box(EventType ev, int indx, void *param, int x, int y, int dx, int dy)
{
  Point p[4];

  // A drag has been released. Save the cumulative dx/dy into box_x/y.
  if (ev & EV_RELEASED)
  {
    cp[0] = cp[0].transform(dx, dy);
    cp[1] = cp[1].transform(dx, dy);
    cp[2] = cp[2].transform(dx, dy);
    cp[3] = cp[3].transform(dx, dy);
    return;
  }

  // Move the box. dx/dy are cumulative (they refer to the original box origin)
  // so we don't change the cp[] until the end of the drag.
  p[0] = cp[0].transform(dx, dy);
  p[1] = cp[1].transform(dx, dy);
  p[2] = cp[2].transform(dx, dy);
  p[3] = cp[3].transform(dx, dy);
  tft.startBuffering();
  tft.fillScreen(0);
  draw_rect(p[0], p[1], p[2], p[3]);
  Log("Drag box");
  tft.endBuffering();

  // Re-origin the regions for taps and drags on box
  detector.onTap(p, 4 , tap_cb, 2);
  detector.onDrag(p, 4, drag_cb_box, 5);
}

void drag_cb_line(EventType ev, int indx, void *param, int x, int y, int dx, int dy)
{
  // Draw trail of drag outside the box.
  tft.drawCircle(x + dx, y + dy, 10, 0xFFFF);
  Log("Drag line");
}

void swipe_cb(EventType ev, int indx, void *param, int x, int y, int dx, int dy)
{
  // Clear screen and redraw box in centre.
  reset();
  Log("Clear");

  // Re-origin regions using the xywh calls.
  detector.onTap(box_x, box_y, box_w, box_h, tap_cb, 2);
  detector.onDrag(box_x, box_y, box_w, box_h, drag_cb_box, 5);
}

void pinch_cb(EventType ev, int indx, void *param, int dx, int dy, float sx, float sy)
{
  Point p[4];

  if (ev & EV_RELEASED)
  {
    cp[0] = cp[0].transform(dx, dy, sx, sy);
    cp[1] = cp[1].transform(dx, dy, sx, sy);
    cp[2] = cp[2].transform(dx, dy, sx, sy);
    cp[3] = cp[3].transform(dx, dy, sx, sy);
    return;
  }

  p[0] = cp[0].transform(dx, dy, sx, sy);
  p[1] = cp[1].transform(dx, dy, sx, sy);
  p[2] = cp[2].transform(dx, dy, sx, sy);
  p[3] = cp[3].transform(dx, dy, sx, sy);
  tft.startBuffering();
  tft.fillScreen(0);
  draw_rect(p[0], p[1], p[2], p[3]);
  Log("Pinch");
  tft.endBuffering();

  // Re-origin the regions for taps and drags on box
  detector.onTap(p, 4 , tap_cb, 2);
  detector.onDrag(p, 4, drag_cb_box, 5);
}

void rotatable_pinch_cb(EventType ev, int indx, void *param, int dx, int dy, float scosa, float ssina)
{
  Point p[4];

  if (ev & EV_RELEASED)
  {
    cp[0] = cp[0].transform(dx, dy, scosa, -ssina, ssina, scosa);
    cp[1] = cp[1].transform(dx, dy, scosa, -ssina, ssina, scosa);
    cp[2] = cp[2].transform(dx, dy, scosa, -ssina, ssina, scosa);
    cp[3] = cp[3].transform(dx, dy, scosa, -ssina, ssina, scosa);
    return;
  }

  p[0] = cp[0].transform(dx, dy, scosa, -ssina, ssina, scosa);
  p[1] = cp[1].transform(dx, dy, scosa, -ssina, ssina, scosa);
  p[2] = cp[2].transform(dx, dy, scosa, -ssina, ssina, scosa);
  p[3] = cp[3].transform(dx, dy, scosa, -ssina, ssina, scosa);
  tft.startBuffering();
  tft.fillScreen(0);
  draw_rect(p[0], p[1], p[2], p[3]);
  Log("Pinch");
  tft.endBuffering();

  // Re-origin the regions for taps and drags on box
  detector.onTap(p, 4 , tap_cb, 2);
  detector.onDrag(p, 4, drag_cb_box, 5);
}

void setup() 
{
  Serial.begin(9600);
  while(!Serial) {}

  tft.begin();
  if (detector.begin()) {
    Serial.println("Touch controller init - OK");
  } else {
    Serial.println("Touch controller init - FAILED");
    while(1) ;
  }

  // Set the rotation. These must occur together.
  tft.setRotation(1);
  detector.setRotation(1);

  // Draw a box at 1/4 - 3/4 screen. It will become the initial
  // region for gestures.
  reset();

  // Various forms of onXxx calls:
  // Taps and long presses inside the box
  detector.onTap(box_x, box_y, box_w, box_h, tap_cb, 2);

  // Swiping anywhere clears the screen. 
  detector.onSwipe(0, 0, 0, 0, swipe_cb, 3);

  // Drags outside box just draw lines/spots
  detector.onDrag(0, 0, 0, 0, drag_cb_line, 4, NULL, CO_NONE, 5);

  // Drag the box around (note the higher priority of this drag)
  detector.onDrag(box_x, box_y, box_w, box_h, drag_cb_box, 5, NULL, CO_NONE, 5);

  // Pinch zoom the box (from anywhere on screen). Versions for both
  // rotatable and non-rotatable pinches.
  detector.onPinch(0, 0, 0, 0, pinch_cb, 6, NULL, false, CO_NONE, 5);
  //detector.onPinch(0, 0, 0, 0, rotatable_pinch_cb, 6, NULL, true, CO_NONE, 5);

  // Mark out the top left on the screen when testing rotations. 
  tft.setTextSize(4);
  Log("Top");
}

void loop() {

  detector.poll();

  delay(10);
}