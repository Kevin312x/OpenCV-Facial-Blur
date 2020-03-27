#include "opencv2/opencv.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace cv;

String face_cascade_file = "lbpcascade_frontalface_improved.xml";
CascadeClassifier face_cascade;

void faceDetection(Mat frame);
Mat blur_roi(Mat roi);

int main(int argc, char* argv[]) {
	VideoCapture cap;

	// Loads the lbp cascade file
	face_cascade.load(face_cascade_file);

	// Opens webcam
	if (!cap.open(0)) {
		std::cout << "Unable to open webcam\n";
		return -1;
	}

	// Loops
	while (true) {
		Mat frame;
		cap >> frame;

		if (frame.empty()) {
			break;
		} else {
			// For each frame, looks for a face
			faceDetection(frame);
		}

		// Press ESC to exit
		if (waitKey(10) == 27) {
			break;
		}
	}
	cap.release();
	return 0;
}

void faceDetection(Mat frame) {
	Mat gray_frame;
	std::vector<Rect> face;

	// Converts image to gray scale (needed for cascade)
	cvtColor(frame, gray_frame, COLOR_BGR2GRAY);

	// Stores faces, if any, in face vector
	face_cascade.detectMultiScale(gray_frame, face, 1.1, 4);

	// If a face exists in image, creates a sub-matrix and blurs it.
	// Then replaces the area on the source image with the blurred image
	for (int i = 0; i < face.size(); ++i) {
		Mat blurred_image;
		Rect roi = Rect(face[0].x, face[0].y, face[0].width*1.25, face[0].height*1.25);
		blurred_image = blur_roi(Mat(frame, roi));
		blurred_image.copyTo(frame(roi));
	}
	
	// Displays image
	imshow("Webcam", frame);
}

// Blurs image
Mat blur_roi(Mat roi) {
	Mat dst;
	blur(roi, dst, Size(55, 55), Point(-1, -1));
	return dst;
}