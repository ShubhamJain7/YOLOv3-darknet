#include <iostream>
#include<fstream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/dnn/dnn.hpp>

using namespace cv;
using namespace std;

// declare file paths
string classesFile = "C:/Users/dell/source/repos/YOLOv3/coco.names";
string configFile = "C:/Users/dell/source/repos/YOLOv3/models/yolov3.cfg";
string weightsFile = "C:/Users/dell/source/repos/YOLOv3/models/yolov3.weights";

// declare constants
float conf_threshold = 0.5f;
float nms = 0.4f;
int width = 416;
int height = 416;

// struct to store results
struct Detection {
    int classId;
    float probability;
    int x;
    int y;
    int width;
    int height;
};

int main()
{   
    // read class names and store into vector
    vector<string> classes;
    ifstream ifs(classesFile.c_str());
    string line;
    while(getline(ifs, line)) 
        classes.push_back(line);

    // load model from config and weights file
    dnn::Net net = dnn::readNetFromDarknet(configFile, weightsFile);

    // get output node names
    vector<String> outputLayerNames;
    vector<String> layersNames = net.getLayerNames();
    vector<int> outLayers = net.getUnconnectedOutLayers();
    outputLayerNames.resize(outLayers.size());
    for (size_t i = 0; i < outLayers.size(); ++i)
        outputLayerNames[i] = layersNames[outLayers[i] - 1];

    // load image and normalize it
    Mat image, blob;
    image = imread("C:/Users/dell/source/repos/YOLOv3/test.jpg", IMREAD_COLOR);
    dnn::blobFromImage(image, blob, 1.0 / 255, Size(width, height), Scalar(0,0,0), true, false);
    net.setInput(blob);

    // feed input blob into network and get outputs
    vector<Mat> outs;
    net.forward(outs, outputLayerNames);
    
    // filter and process the results
    vector<int> classIds;
    vector<float> confidences;
    vector<Rect> boxes;
    for (size_t i = 0; i < outs.size(); i++)
    {
        float* data = (float*)outs[i].data;
        for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols)
        {
            Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
            Point classidPoint;
            double confidence;
            minMaxLoc(scores, 0, &confidence, 0, &classidPoint);
            // include only those results that exceed the confidence threshold
            if (confidence > conf_threshold)
            {
                // calculate bounding-box co-ordinates
                int centerX = (int)(data[0] * image.cols);
                int centerY = (int)(data[1] * image.rows);
                int w = (int)(data[2] * image.cols);
                int h = (int)(data[3] * image.rows);
                int x = centerX - w / 2;
                int y = centerY - h / 2;

                // store filtered results
                classIds.push_back(classidPoint.x);
                confidences.push_back((float)confidence);
                boxes.push_back(Rect(y, x, w, h));
            }
        }
    }

    // perform non-maximim-suppression to remove overlapping boxes for the same object 
    vector<int> indexes;
    dnn::NMSBoxes(boxes, confidences, conf_threshold, nms, indexes);

    // store results in struct and display
    vector<Detection> objects;
    for (size_t i = 0; i < indexes.size(); ++i)
    {
        int idx = indexes[i];

        Detection d;
        d.classId = classIds[idx];
        d.probability = confidences[idx];
        Rect box = boxes[idx];
        d.x = box.x;
        d.y = box.y;
        d.width = box.width;
        d.height = box.height;

        objects.push_back(d);

        cout << d.classId << "(" << d.probability << "):";
        cout << "[" << d.x << "," << d.y << "," << d.width << "," << d.height << "]" << endl;
    }
}