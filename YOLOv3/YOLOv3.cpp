#include <iostream>
#include<fstream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/dnn/dnn.hpp>

using namespace cv;
using namespace std;

// Declare file paths
string classesFile = "C:/Users/dell/source/repos/YOLOv3/coco.names";
string configFile = "C:/Users/dell/source/repos/YOLOv3/models/yolov3.cfg";
string weightsFile = "C:/Users/dell/source/repos/YOLOv3/models/yolov3.weights";

// Declare constants
float conf_threshold = 0.5f;
float nms = 0.4f;
int width = 416;
int height = 416;

int main()
{   
    // Read class names and store into vector
    vector<string> classes;
    ifstream ifs(classesFile.c_str());
    string line;
    while(getline(ifs, line)) 
        classes.push_back(line);

    // Load model from config and weights file
    dnn::Net net = dnn::readNetFromDarknet(configFile, weightsFile);

    // Get output node names
    vector<String> outputLayerNames;
    vector<String> layersNames = net.getLayerNames();
    vector<int> outLayers = net.getUnconnectedOutLayers();
    outputLayerNames.resize(outLayers.size());
    for (size_t i = 0; i < outLayers.size(); ++i)
        outputLayerNames[i] = layersNames[outLayers[i] - 1];

    // Load image and normalize it
    Mat image, blob;
    image = imread("C:/Users/dell/source/repos/YOLOv3/test.jpg", IMREAD_COLOR);
    dnn::blobFromImage(image, blob, 1.0 / 255, Size(width, height), Scalar(0,0,0), true, false);
    net.setInput(blob);

    // Feed input blob into network and get outputs
    vector<Mat> outs;
    net.forward(outs, outputLayerNames);
    
    // Filter and process the results
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
            // Include only those results that exceed the confidence threshold
            if (confidence > conf_threshold)
            {
                // Calculate bounding-box co-ordinates
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

    // Perform non-maximim-suppression to remove overlapping boxes for the same object 
    vector<int> indexes;
    dnn::NMSBoxes(boxes, confidences, conf_threshold, nms, indexes);

    // Display results
    for (size_t i = 0; i < indexes.size(); ++i)
    {
        int idx = indexes[i];
        int classId = classIds[idx];
        string label = classes[classId];
        float confidence = confidences[idx];
        Rect box = boxes[idx];
        cout << label << "(" << confidence << "):";
        cout << "[" << box.x << "," << box.y << "," << box.width << "," << box.height << "]" << endl;
    }
}