#ifndef GESTURE_DETECTOR_H
#define GESTURE_DETECTOR_H

#include "Arduino_GigaDisplayTouch.h"

// Gesture detection.

// Width and height of screen in its natural rotation.
#define WIDTH             480
#define HEIGHT            800

// The time between scans of the touch screen, in ms.
#define SCAN_TIME         30

// The long press duration (in ms)
#define LONG_PRESS_TIME   500

// The swipe speed, defined as a pixels/ms value (total dx or dy / time in ms)
// as measured after at least SWIPE_TIME has elapsed.
// The swipe time is about 5 scans worth at 30ms / scan.
#define SWIPE_TIME        150

// Maximum number of Points in a polygon region.
#define MAX_POINTS        16

// The minimum scale for a pinch, so we don't pinch the scale factors
// down to zero or go negative.
#define MIN_SCALE         0.1

// Event types. We would like to use and enum here, but C++ will not let
// you AND or OR them.
typedef int EventType;
int const   EV_NONE = 0;
int const   EV_TAP = 1;
int const   EV_DRAG = 2;
int const   EV_SWIPE = 3;
int const   EV_PINCH = 4;
int const   EV_RELEASED = 0x100;    // OR'd in when the event is released
int const   EV_LONG_PRESS = 0x200;  // OR'd in when a tap is held for more than LONG_PRESS_TIME ms

// The maximum number of events that can be registered
#define MAX_EVENTS  20

// Allowable constraints on drag, swipe and pinch directions.
// They can be restricted to only horizontal or vertical. Any that
// do not lie within the angle_tol of the allowed direction will
// not be acted upon.
typedef enum Constraint
{
  CO_NONE = 0,
  CO_HORIZ,
  CO_VERT
};

// A 2D point class used by the regions on onTap, etc. calls.
class Point
{
public:
  Point() { }
  // Point(Point &p) { x = p.x; y = p.y; }  // compiler should do this on its own
  Point(int u, int v) { x = u; y = v; }     // initialised from supplied x,y
  ~Point(void) { }

  int x;
  int y;

  Point transform(int dx, int dy)
  {
    return Point(x + dx, y + dy);
  }

  // Other transforms as needed
  // Transform by a translation only
  Point transform(int dx, int dy, float sx, float sy)
  {
    return Point(sx * x + dx, sy * y + dy);
  }

  // Transform by a general 6-component matrix (4x4 with translation components)
  Point transform(int dx, int dy, float a11, float a12, float a21, float a22)
  {
    return Point(a11 * x + a12 * y + dx, a21 * x + a22 * y + dy);
  }

  // Tests for point inside polygon.
  int in_polygon(Point* V, int n);

private:
  int is_left(Point P0, Point P1);
};

// Callback functions for various events. They are called with:
// type       The event being called back on. When released, the call is made
//            with its type OR'd with EV_RELEASED.
// indx       A unique priority passed to the onXxx functions below. Reusing an index
//            will overwrite an existing callback registration. Higher indices have
//            higher priorities.
// param      Any user parameter passed in from the onXxx function
// x, y       The initial tap position. The last seen position is
//            always (x + dx, y + dy)
// dx, dy     The total change in x/y, for the finger position (drag)
//            or the translation component (for a pinch)
// sx, sy     The total scale factors in x/y (for a non-rotatable pinch)
//            [x'] = [sx 0 dx][x]
//            [y'] = [0 sy dy][y]
//            [1 ] = [0  0  1][1]

//            The single scale factor multiplied by the sin and cos of the rotation
//            (for a rotatable pinch)
//            [x'] = [sx -sy dx][x]     where sx = S cos(a), sy = S sin(a)
//            [y'] = [sy  sx dy][y]
//            [1 ] = [0   0   1][1]

typedef void (*TapCB)(EventType type, int indx, void *param, int x, int y);     // for both taps and long presses
typedef void (*DragCB)(EventType type, int indx, void *param, int x, int y, int dx, int dy);  // for drags and swipes
typedef void (*PinchCB)(EventType type, int indx, void *param, int dx, int dy, float sx, float sy);

class GestureDetector : public Arduino_GigaDisplayTouch
{
  public:
    GestureDetector() {}
    ~GestureDetector(void) {}
    bool begin();

    // Poll for events. Call as frequently as possible at the top of loop().
    void poll();

    // Register a callback for taps or long presses. There are two versions,
    // for an upright rect and a n-Point region.
    // xywh         Rectangular region to pick up taps in
    // rc/npts      Array of Points and its size
    // tapCB        Callback function
    // indx         Priority index to pass to callback. Used as index to events array
    //              and setting the priority for overlapping events.
    //              Allows overwrite/update of registration, e.g. if region changes.
    //              Higher indices have higher priority.
    // param        Any parameter to be passed to the callback functions
    void onTap(Point *rc, int nPts, TapCB tapCB, int indx, void *param = NULL)
    {
      fill_event(EV_TAP, rc, nPts, tapCB, NULL, NULL, indx, param);
    }
    void onTap(int x, int y, int w, int h, TapCB tapCB, int indx, void *param = NULL)
    {
      int nPts;
      Point rc[4];
      nPts = fill_rect_region(x, y, w, h, rc);
      fill_event(EV_TAP, rc, nPts, tapCB, NULL, NULL, indx, param);
    }

    // Register a callback for drags, swipes or pinches.
    // As above, plus:
    // rotatable    For a pinch, if true then rotation is allowed but only a single
    //              scale factor. The sx/sy are returned as S cos(a) and S sin(a).
    // constraint   Restricts to H/V movement.
    // angle_tol    Tolerance below which a drag or pinch
    //              is snapped to an h/v axis, expressed
    //              as a multiple, e.g. setting to 10 means:
    //              that dx/dy > 10 --> dy = 0 (horizontal)
    //              and dy/dx > 10 --> dx = 0 (vertical)
    void onDrag(Point *rc, int nPts, DragCB dragCB, int indx, void *param = NULL, Constraint constraint = CO_NONE, int angle_tol = 3)
    {
      fill_event(EV_DRAG, rc, nPts, NULL, dragCB, NULL, indx, param, false, constraint, angle_tol);
    }
    void onDrag(int x, int y, int w, int h, DragCB dragCB, int indx, void *param = NULL, Constraint constraint = CO_NONE, int angle_tol = 3)
    {
      int nPts;
      Point rc[4];
      nPts = fill_rect_region(x, y, w, h, rc);
      fill_event(EV_DRAG, rc, nPts, NULL, dragCB, NULL, indx, param, false, constraint, angle_tol);
    }

    void onSwipe(Point *rc, int nPts, DragCB dragCB, int indx, void *param = NULL, Constraint constraint = CO_NONE, int angle_tol = 1)
    {
      fill_event(EV_SWIPE, rc, nPts, NULL, dragCB, NULL, indx, param, false, constraint, angle_tol);
    }
    void onSwipe(int x, int y, int w, int h, DragCB dragCB, int indx, void *param = NULL, Constraint constraint = CO_NONE, int angle_tol = 1)
    {
      int nPts;
      Point rc[4];
      nPts = fill_rect_region(x, y, w, h, rc);
      fill_event(EV_SWIPE, rc, nPts, NULL, dragCB, NULL, indx, param, false, constraint, angle_tol);
    }

    void onPinch(Point *rc, int nPts, PinchCB pinchCB, int indx, void *param = NULL, bool rotatable = false, Constraint constraint = CO_NONE, int angle_tol = 3)
    {
      fill_event(EV_PINCH, rc, nPts, NULL, NULL, pinchCB, indx, param, rotatable, constraint, angle_tol);
    }
    void onPinch(int x, int y, int w, int h, PinchCB pinchCB, int indx, void *param = NULL, bool rotatable = false, Constraint constraint = CO_NONE, int angle_tol = 3)
    {
      int nPts;
      Point rc[4];
      nPts = fill_rect_region(x, y, w, h, rc);
      fill_event(EV_PINCH, rc, nPts, NULL, NULL, pinchCB, indx, param, rotatable, constraint, angle_tol);
    }

    // Cancel an event at the given index.
    void cancelEvent(int indx) { events[indx].type = EV_NONE; }

    // Ask if there is an event at the given index
    bool isEventRegistered(int indx) { return events[indx].type != EV_NONE; }

    // Set the screen rotation. Use with GFX::setRotation to keep coordinates in step.
    void setRotation(int rot) { rotation = rot; }

  private:
    int rotation = 0;
    unsigned long last_polled = 0;

    // Struct to keep track of a contact on the touch screen.
    typedef struct TrackedContact
    {
      int     indx;           // Which contact this is, returned from underlying library
      int     init_x, init_y; // Initial point
      int     dx, dy;         // Total movement since (init_x, init_y).
    };

    // Struct to keep track of a currently progressing gesture.
    typedef struct TrackedEvent
    {
      EventType       type;     // Is this a tap, long press, drag or pinch
      unsigned long   start_time; // Time in ms of initial press (used to time long presses)
      unsigned long   hold_time; // Time in ms that a tap has been held
      int             active_event; // Index of event currently being tracked or -1 if none.
      TrackedContact  cont[2];  // Up to two tracked contacts (to allow pinches)
      Constraint      working_co; // Constraint used for pinch
                                // (combines event constraint and initial contact point angle)
    };

    // The event structure for all registered events.
    typedef struct RegEvent
    {
      EventType   type;           // What this is (tap, drag or pinch)
      void        *param;         // User parameter passed to callbacks
      Point       reg[MAX_POINTS]; // The region it is sensitive to
      int         nPts;          // Number of Points in region
      TapCB       tapCallback;    // Callback function for taps and long presses.
      DragCB      dragCallback;   // For drags and swipes
      PinchCB     pinchCallback;  // For pinches.
      Constraint  constraint;     // Whether restricted to h/v drag/pinch
      int         angle_tol;      // Tolerance below which a drag or pinch
                                  // is snapped to an h/v axis, expressed
                                  // as a multiple, e.g. setting to 10 means:
                                  // that dx/dy > 10 --> dy = 0 (horizontal)
                                  // and dy/dx > 10 --> dx = 0 (vertical)
      bool        rotatable;      // Whether pinch is rotatable
    };

    TrackedEvent  track;
    RegEvent      events[MAX_EVENTS];

    void start_new_tracked(unsigned long current_time, EventType ev);
    void call_cb(void);
    bool in_region(RegEvent *event, int x, int y);

    // Checker on dx/dy speed, rejecting any that don't fit
    bool check_constraints(RegEvent *event, int dx, int dy);
    // Enforcer of HV constraints on dx/dy
    Constraint enforce_constraints(RegEvent *event, int dx, int dy, int *new_dx, int *new_dy);

    // Fill in an event of any type
    void fill_event
    (
      EventType ev,
      Point *rc,
      int nPts,
      TapCB tapCB,
      DragCB dragCB,
      PinchCB pinchCB,
      int indx,
      void *param,
      bool rotatable = false,
      Constraint constraint = CO_NONE,
      int angle_tol = 5
    );

    int fill_rect_region(int x, int y, int w, int h, Point *rc)
    {
      if (w == 0 || h == 0)
        return 0;     // region is empty
      rc[0] = Point(x, y);
      rc[1] = Point(x + w, y);
      rc[2] = Point(x + w, y + h);
      rc[3] = Point(x, y + h);
      return 4;
    }


};

// Geometry routines not belonging to a class

// Length of vector betwen two points
float length(int x1, int x2, int y1, int y2);

// Dot product of two vectors
float dot(int u1, int u2, int v1, int v2);

// Perp product of two vectors
float perp(int u1, int u2, int v1, int v2);

#endif // def GESTURE_DETECTOR_H