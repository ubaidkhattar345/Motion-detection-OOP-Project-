#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <memory>

using namespace cv;
using namespace std;

// Base class for a generic sensor
class Sensor {
protected:
    string id;
public:
    Sensor(const string& sensor_id) : id(sensor_id) {}
    virtual bool detectMotion(const Mat& frame) = 0;
    virtual void displayStatus(bool motionDetected) = 0;
    virtual ~Sensor() {}
};

// Derived class for motion detection using frame difference
class MotionSensor : public Sensor {
private:
    Mat previousFrameGray;
public:
    MotionSensor(const string& sensor_id) : Sensor(sensor_id) {}

    bool detectMotion(const Mat& frame) override {
        Mat gray, diff;
        cvtColor(frame, gray, COLOR_BGR2GRAY);
        GaussianBlur(gray, gray, Size(21, 21), 0);

        if (previousFrameGray.empty()) {
            previousFrameGray = gray;
            return false; // No motion on first frame
        }

        absdiff(previousFrameGray, gray, diff);
        threshold(diff, diff, 25, 255, THRESH_BINARY);
        dilate(diff, diff, Mat(), Point(-1, -1), 2);

        vector<vector<Point>> contours;
        findContours(diff, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

        previousFrameGray = gray;

        for (auto& contour : contours) {
            if (contourArea(contour) > 500) { // Ignore small movements
                return true;
            }
        }
        return false;
    }

    void displayStatus(bool motionDetected) override {
        if (motionDetected)
            cout << "Motion detected by sensor " << id << "!" << endl;
        else
            cout << "No motion detected by sensor " << id << "." << endl;
    }
};

// Surveillance system class
class SurveillanceSystem {
protected:
    vector<shared_ptr<Sensor>> sensors;
public:
    void addSensor(shared_ptr<Sensor> sensor) {
        sensors.push_back(sensor);
    }

    void monitor() {
        VideoCapture cap(0); // Open default camera
        if (!cap.isOpened()) {
            cerr << "Error opening camera" << endl;
            return;
        }

        Mat frame;
        cout << "Starting surveillance..." << endl;

        while (true) {
            cap >> frame;
            if (frame.empty()) break;

            for (auto& sensor : sensors) {
                bool motion = sensor->detectMotion(frame);
                sensor->displayStatus(motion);
            }

            imshow("Camera Feed", frame);
            if (waitKey(30) == 27) break; // Exit on 'ESC'
        }
        cap.release();
        destroyAllWindows();
    }
};

int main() {
    auto sensor1 = make_shared<MotionSensor>("S1");
    auto sensor2 = make_shared<MotionSensor>("S2");

    SurveillanceSystem system;
    system.addSensor(sensor1);
    system.addSensor(sensor2);

    system.monitor();
    return 0;
}

