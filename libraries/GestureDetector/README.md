# Touch_gesture
Gesture detection library for the Arduino Giga Display.

Users can register callbacks for various types of gesture within a defined region on the screen:
- tap
- long press
- drag
- swipe (a fast drag or flick)
- pinch without rotation (two scale factors) or with rotation (one scale and angle)

Moving gestures (drags, swipes and pinches) can be constrained to fixed directions or free.
Pinches can allow rotation (e.g. a map) or not (e.g. a document)

The example program draws a rectangle on the screen and responds to:
- taps in the box (draws circle where tapped)
- long presses in the box (draws filled circle)
- dragging the box around
- swiping to clear the screen and reset the box
- pinching of two kinds on the box

Dependencies:
- Arduino Giga touch library (Arduino_GigaDisplayTouch)

The gilesp1729 fork of the Giga GFX library has two new calls that stop the display flickering
and improve appearance of smooth drags and pinches. See the example code for how to comment these
out if you don't want to use the fork.
