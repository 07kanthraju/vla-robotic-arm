import threading
import cv2
import time
from camera_module import HandTracker
from gesture_module import count_fingers
from servo_controller import ServoController
from actions import gesture_actions, motion_lock

shutdown_flag = False  # NEW: global flag to stop new threads
active_threads = []    # Track running gesture threads

def main():
    global shutdown_flag, active_threads

    servo = ServoController("/dev/ttyUSB0", 9600)
    tracker = HandTracker()
    last_finger_count = -1

    print("✅ Gesture-Control System Started")

    while True:
        frame, result = tracker.get_frame()
        if frame is None:
            continue

        finger_count = 0
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                tracker.draw_landmarks(frame, hand_landmarks)
                finger_count = count_fingers(hand_landmarks)

        cv2.putText(frame, f"Fingers: {finger_count}", (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

        # Only trigger if system is active
        if not shutdown_flag and finger_count != last_finger_count and not motion_lock.locked():
            print(f"🖐 {finger_count} fingers detected")
            thread = threading.Thread(target=gesture_actions[finger_count], args=(servo,))
            thread.start()
            active_threads.append(thread)
            last_finger_count = finger_count

        tracker.show(frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC pressed
            print("\n🛑 Shutting down gracefully...")
            shutdown_flag = True
            break

    # Wait for all threads to finish safely
    for t in active_threads:
        t.join()

    tracker.release()
    servo.close()
    print("✅ Shutdown complete — port closed safely.")

if __name__ == "__main__":
    main()
