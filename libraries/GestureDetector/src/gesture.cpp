#include "Arduino.h"
#include "GestureDetector.h"

// Simple ASSERT needs to flash some lights or do something at the end
#define ASSERT(expr)  if (!(expr)) {Serial.print(__FILE__);Serial.print("(");Serial.print(__LINE__);Serial.print("): ");Serial.println(#expr);while(1);}

bool GestureDetector::begin()
{
  // Set up the tracked events and other initialisations.
  track.type = EV_NONE;
  for (int i = 0; i < MAX_EVENTS; i++)
    events[i].type = EV_NONE;

  // Call the underlying begin()
  return Arduino_GigaDisplayTouch::begin();
}

// A helper to determine if a point (x, y) is in a rectangular region
// (upper-left inclusive, lower-right exclusive). Points in regions with
// zero area (zero w or h) always are considered inside. The test is only
// caried out once, as drag callbacks might update the region.
bool GestureDetector::in_region(RegEvent *event, int x, int y)
{
  Point p(x, y);
  if (event->nPts == 0)
    return true;

  return p.in_polygon(event->reg, event->nPts) != 0;
}

// Reject any drags/wipes that don't meet the constraints.
// TODO a version for pinches.
bool GestureDetector::check_constraints(RegEvent *event, int dx, int dy)
{
  if (abs(dx) != 0 && abs(dy/dx) > event->angle_tol)
  {
    // Vertical (or nearly so). Reject if we only want horizontal.
    if (event->constraint == CO_HORIZ)
      return false;
  }
  else if (abs(dy) != 0 && abs(dx/dy) > event->angle_tol)
  {
    // Horizontal (or nearly so). Reject if we only want vertical.
    if (event->constraint == CO_VERT)
      return false;
  }
  else
  {
    // It's in between vertical and horizontal. Only accept if CO_NONE.
    if (event->constraint != CO_NONE)
      return false;
  }
  return true;
}

// Adjust the dx/dy of a drag or pinch to constraints and angle_tol.
// This version cannot reject an event but only modify it.
// Return the constraint that was either enforced or impled by the angle_tol.
Constraint GestureDetector::enforce_constraints(RegEvent *event, int dx, int dy, int *new_dx, int *new_dy)
{
  *new_dx = dx;
  *new_dy = dy;
  if (event->constraint == CO_VERT)         // TODO snap sx/sy here for pinches likewise
  {
    *new_dx = 0;
    return CO_VERT;
  }
  else if (event->constraint == CO_HORIZ)
  {
    *new_dy = 0;
    return CO_HORIZ;
  }
  else if (abs(dx) == 0 || abs(dy/dx) > event->angle_tol)
  {
    *new_dx = 0;
    return CO_VERT;
  }
  else if (abs(dy) == 0 || abs(dx/dy) > event->angle_tol)
  {
    *new_dy = 0;
    return CO_HORIZ;
  }

  return CO_NONE;
}


// Call any valid callback function for the tracked event. Apply any constraints
// and check if initial point(s) are inside any given regions for the registered
// callback.
void GestureDetector::call_cb(void)
{
  int i, dx, dy, dx0, dy0, dx1, dy1;
  float sx, sy;
  EventType released = track.type & EV_RELEASED;  // this is why they are const ints, not an enum
  EventType ev = track.type & ~EV_RELEASED;

  // Switch on the underlying event type
  switch (ev)
  {
  case EV_TAP:
    // Look for a matching event if we haven't already got one
    if (track.active_event < 0)
    {
      for (i = MAX_EVENTS - 1; i >= 0; i--)
      {
        if (!in_region(&events[i], track.cont[0].init_x, track.cont[0].init_y))
          continue;
        if (events[i].type == EV_TAP)
        {
          track.active_event = i;
          break;
        }
      }
    }

    if (track.active_event < 0)
      return;   // nothing to do here

#if 0
    Serial.print("Tap ");
    if (released)
      Serial.print("(Rel) ");
    Serial.print(track.active_event);
    Serial.print(" ");
    Serial.print(track.cont[0].init_x + track.cont[0].dx);
    Serial.print(" ");
    Serial.println(track.cont[0].init_y + track.cont[0].dy);
#endif

    i = track.active_event;

    if (track.hold_time >= LONG_PRESS_TIME)
    {
      events[i].tapCallback(EV_TAP | EV_LONG_PRESS | released, i, events[i].param, track.cont[0].init_x, track.cont[0].init_y);
      return;
    }
    else
    {
      events[i].tapCallback(EV_TAP | released, i, events[i].param, track.cont[0].init_x, track.cont[0].init_y);
      return;
    }
    break;

  case EV_DRAG:
  case EV_SWIPE:
    if (track.active_event < 0)
    {
      for (i = MAX_EVENTS - 1; i >= 0; i--)
      {
        if (!in_region(&events[i], track.cont[0].init_x, track.cont[0].init_y))
          continue;
        if (!check_constraints(&events[i], track.cont[0].dx, track.cont[0].dy))
          continue;
        if (events[i].type == ev)
        {
          track.active_event = i;
          break;
        }
      }
    }

    if (track.active_event < 0)
      return;   // nothing to do here

#if 0
    if (ev == EV_DRAG)
      Serial.print("Drag ");
    else
      Serial.print("Swipe ");
    if (released)
      Serial.print("(Rel) ");
    Serial.print(track.active_event);
    Serial.print(" ");
    Serial.print(track.cont[0].init_x + track.cont[0].dx);
    Serial.print(" ");
    Serial.println(track.cont[0].init_y + track.cont[0].dy);
#endif

    i = track.active_event;
    ASSERT(events[i].type == EV_DRAG || events[i].type == EV_SWIPE);
    enforce_constraints(&events[i], track.cont[0].dx, track.cont[0].dy, &dx, &dy);
    events[i].dragCallback(ev | released, i, events[i].param, track.cont[0].init_x, track.cont[0].init_y, dx, dy);
    break;

  case EV_PINCH:
    if (track.active_event < 0)
    {
      for (i = MAX_EVENTS - 1; i >= 0; i--)
      {
        // This event has two contacts, which both must be in-region
        // and pass any H/V constraints based on their initial positions.
        if (!in_region(&events[i], track.cont[0].init_x, track.cont[0].init_y))
          continue;
        if (!in_region(&events[i], track.cont[1].init_x, track.cont[1].init_y))
          continue;

        if
        (
          !check_constraints
          (
            &events[i],
            track.cont[0].init_x - track.cont[1].init_x,
            track.cont[0].init_y - track.cont[1].init_y
          )
        )
          continue;

        if (events[i].type == EV_PINCH)
        {
          int dummy_dx, dummy_dy;

          track.active_event = i;

          // We can set the working constraint up once here, as it only
          // depends on the initial contact points.
          track.working_co =
            enforce_constraints
            (
              &events[i],
              track.cont[0].init_x - track.cont[1].init_x,
              track.cont[0].init_y - track.cont[1].init_y,
              &dummy_dx,
              &dummy_dy
            );
          break;
        }
      }
    }

    if (track.active_event < 0)
      return;   // nothing to do here

#if 0
    Serial.print("Pinch ");
    if (released)
      Serial.print("(Rel) ");
    Serial.print(track.active_event);
    Serial.print(" ");
    Serial.print(track.cont[0].init_x + track.cont[0].dx);
    Serial.print(" ");
    Serial.print(track.cont[0].init_y + track.cont[0].dy);
    Serial.print(" ");
    Serial.print(track.cont[1].init_x + track.cont[1].dx);
    Serial.print(" ");
    Serial.println(track.cont[1].init_y + track.cont[1].dy);
#endif

    i = track.active_event;
    ASSERT(events[i].type == EV_PINCH);

    // Handle this differently depending on whether this pinch is rotatable or not.
    // Some local variables/line shorteners
    int init_x0 = track.cont[0].init_x;
    int init_y0 = track.cont[0].init_y;
    int init_x1 = track.cont[1].init_x;
    int init_y1 = track.cont[1].init_y;
    int x0 = track.cont[0].init_x + track.cont[0].dx;
    int y0 = track.cont[0].init_y + track.cont[0].dy;
    int x1 = track.cont[1].init_x + track.cont[1].dx;
    int y1 = track.cont[1].init_y + track.cont[1].dy;

    if (events[i].rotatable)
    {
      // Measure the (single) scale factor.
      float len0 = length(init_x0, init_x1, init_y0, init_y1);
      float len1 = length(x0, x1, y0, y1);
      float scale = len1 / len0;

      // Measure the rotation by comparing the lines between the endpoints,
      // giving the cos and sin of the rotation angle.
      float cosa = dot(init_x1 - init_x0, x1 - x0, init_y1 - init_y0, y1 - y0);
      cosa = (cosa / len1) / len0;

      float sina = perp(init_x1 - init_x0, x1 - x0, init_y1 - init_y0, y1 - y0);
      sina = (sina / len1) / len0;

#if 0
        Serial.print("Scale ");
        Serial.print(scale);
        Serial.print(" Cosa/sina ");
        Serial.print(cosa);
        Serial.print(" ");
        Serial.println(sina);
#endif

      // Calculate the translation components
      if (scale < MIN_SCALE)
        scale = MIN_SCALE;
      sx = cosa * scale;
      sy = sina * scale;
      dx = x0 - (sx * init_x0 - sy * init_y0);
      dy = y0 - (sy * init_x0 + sx * init_y0);
#if 0
      Serial.print("dx/dy ");
      Serial.print(" ");
      Serial.print(dx);
      Serial.print(" ");
      Serial.println(dy);
#endif
    }
    else
    {
      // Simpler case with two scales and no rotation. Solve 4 simultaneous
      // equations for 4 coefficients.
      if (track.working_co == CO_VERT)
        sx = 1.0f;
      else
        sx = (float)(x0 - x1) / ((init_x0 - init_x1));

      if (track.working_co == CO_HORIZ)
        sy = 1.0f;
      else
        sy = (float)(y0 - y1) / ((init_y0 - init_y1));

#if 0
      Serial.print("Scales ");
      Serial.print(sx);
      Serial.print(" ");
      Serial.println(sy);
#endif

      // Calculate the translation components
      if (sx < MIN_SCALE)
        sx = MIN_SCALE;
      if (sy < MIN_SCALE)
        sy = MIN_SCALE;
      dx = x0 - sx * init_x0;
      dy = y0 - sy * init_y0;
    }
    events[i].pinchCallback(EV_PINCH | released, i, events[i].param, dx, dy, sx, sy);
    break;
  }
}

// Start a new tracked event. zero out timer counter and active
// event index, so the next call to call_cb() finds the right event.
void GestureDetector::start_new_tracked(unsigned long current_time, EventType ev)
{
  track.start_time = current_time;
  track.hold_time = 0;
  track.active_event = -1;
  track.type = ev;
}


// Poll for some activity.
void GestureDetector::poll()
{
  uint8_t contacts;
  GDTpoint_t points[5];
  unsigned long current_time;
  int x[5], y[5], t;

  // If it hasn't been 30ms since last time, just return.
  current_time = millis();
  if (current_time < last_polled + SCAN_TIME)
    return;
  last_polled = current_time;

  contacts = getTouchPoints(points);

#if 0
    // Debugging code to print out the active contacts.
  if (contacts > 0)
  {
    Serial.print("Contacts: ");
    Serial.println(contacts);

    for (uint8_t i = 0; i < contacts; i++)
    {
      Serial.print(i);
      Serial.print(" ");
      Serial.print(points[i].trackId);
      Serial.print(" ");
      Serial.print(points[i].x);
      Serial.print(" ");
      Serial.print(points[i].y);
      Serial.print(" ");
      Serial.println(points[i].area);
    }
  }
#endif

  // Handle rotation on the points coming back from the touch screen.
  for (uint8_t i = 0; i < contacts; i++)
  {
    switch (rotation)
    {
    case 0:
      x[i] = points[i].x;
      y[i] = points[i].y;
      break;
    case 1:
      x[i] = points[i].y;
      y[i] = WIDTH - 1 - points[i].x;
      break;
    case 2:
      x[i] = WIDTH - 1 - points[i].x;
      y[i] = HEIGHT - 1 - points[i].y;
      break;
    case 3:
      x[i] = HEIGHT - 1 - points[i].y;
      y[i] = points[i].x;
      break;
    }
  }

  // Deal with the combinations of current contacts and previous contacts.
  switch (contacts)
  {
  case 0:
    // No contacts. Call the relevant callback with the released flag set.
    if (track.type == EV_TAP && (track.cont[0].dx != 0 || track.cont[0].dy != 0))
      start_new_tracked(current_time, EV_SWIPE);
    track.type |= EV_RELEASED;
    call_cb();
    start_new_tracked(current_time, EV_NONE);
    break;

  case 1:
    // One contact. See what we have been doing previously.
    switch (track.type)
    {
    case EV_NONE:
      // A new contact. Send to tap. It could become a long press or a drag later.
      start_new_tracked(current_time, EV_TAP);
      track.cont[0].init_x = x[0];
      track.cont[0].init_y = y[0];
      track.cont[0].dx = track.cont[0].dy = 0;
      track.cont[0].indx = 0;
      call_cb();
      break;

    case EV_TAP:
    case EV_DRAG:
      // Still holding a tap, or perheps dragging. Cope with the case where
      // the finger moves.
      track.cont[0].dx = x[0] - track.cont[0].init_x;
      track.cont[0].dy = y[0] - track.cont[0].init_y;
      track.hold_time = current_time - track.start_time;
      if (track.cont[0].dx != 0 || track.cont[0].dy != 0)
      {
        // There is movement. This contact is promoted to a drag.
        // If the movement is fast (it's released in <= SWIPE_TIME ms)
        // then it becomes a swipe. Call its callback. Do not release
        // the tap, as it isn't really a tap anyway. Update the start time
        // for a new speed calculation now that drag has begun.
        {
          if (track.type == EV_TAP && track.hold_time > SWIPE_TIME)
            start_new_tracked(current_time, EV_DRAG);
          call_cb();
        }
      }
      break;

    case EV_PINCH:
      // We have been pinching, but one finger has been lifted off.
      // Since we don't know which one, we can't continue a drag from its
      // last position; release the pinch and start a new drag from here.
      track.type |= EV_RELEASED;
      call_cb();

      start_new_tracked(current_time, EV_DRAG);
      track.cont[0].init_x = x[0];
      track.cont[0].init_y = y[0];
      track.cont[0].dx = track.cont[0].dy = 0;
      track.cont[0].indx = 0;
      break;
    }
    break;

  case 2:
    // There are two contacts. We are either starting or continuing a pinch.
    if (track.type != EV_PINCH)
    {
      // Start a new pinch.
      start_new_tracked(current_time, EV_PINCH);
      track.cont[0].init_x = x[0];
      track.cont[0].init_y = y[0];
      track.cont[0].dx = track.cont[0].dy = 0;
      track.cont[0].indx = 0;
      track.cont[1].init_x = x[1];
      track.cont[1].init_y = y[1];
      track.cont[1].dx = track.cont[1].dy = 0;
      track.cont[1].indx = 1;
    }
    else
    {
      track.cont[0].dx = x[0] - track.cont[0].init_x;
      track.cont[0].dy = y[0] - track.cont[0].init_y;
      track.cont[1].dx = x[1] - track.cont[1].init_x;
      track.cont[1].dy = y[1] - track.cont[1].init_y;
      track.hold_time = current_time - track.start_time;
      call_cb();
    }
    break;
  }
}

// Register callbacks for the various kinds of gestures. The various flavours of this call,
// along with their default arguments, are set up in the header file.
void GestureDetector::fill_event
(
  EventType ev,
  Point *rc,
  int nPts,
  TapCB tapCB,
  DragCB dragCB,
  PinchCB pinchCB,
  int indx,
  void *param,
  bool rotatable,
  Constraint constraint,
  int angle_tol
)
{
  if (indx >= MAX_EVENTS)
    return;   // TODO figure out how to return an error code here
  if (nPts > MAX_POINTS)
    nPts = MAX_POINTS;

  events[indx].type = ev;
  events[indx].param = param;
  if (nPts != 0)
    memcpy(events[indx].reg, rc, nPts * sizeof(Point));
  events[indx].reg[nPts] = events[indx].reg[0];  // Close polygon for in_polygon test
  events[indx].nPts = nPts;
  events[indx].tapCallback = tapCB;
  events[indx].dragCallback = dragCB;
  events[indx].pinchCallback = pinchCB;
  events[indx].constraint = constraint;
  events[indx].angle_tol = angle_tol;
  events[indx].rotatable = rotatable;
}

