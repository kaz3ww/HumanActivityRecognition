import cv2
import mediapipe as mp
import csv

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Video Capture
cap = cv2.VideoCapture(0)

# Output CSV file
csv_file = 'pose_data.csv'
actions = ['walking', 'sitting', 'jumping']  # Define activities
data_columns = ['label'] + [f'kp{i}' for i in range(132)]  # 33 landmarks x 4 values

# Open CSV file to store keypoints
with open(csv_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(data_columns)  # Write header row

    print("Press 's' to start recording data, 'q' to quit.")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)

        # Draw pose on the frame
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Extract keypoints if pose is detected
        if results.pose_landmarks:
            keypoints = []
            for landmark in results.pose_landmarks.landmark:
                keypoints.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])

            # Ask user to label data
            cv2.putText(frame, "Enter label (walking/sitting/jumping): ", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.imshow("Pose Estimation", frame)

            # Wait for user input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):  # 's' key to start recording
                label = input("Enter label: ").strip().lower()
                if label in actions:
                    writer.writerow([label] + keypoints)
                    print(f"Recorded {label} data")
            elif key == ord('q'):  # 'q' key to quit
                break

        # Display frame
        cv2.imshow('Pose Estimation', frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
