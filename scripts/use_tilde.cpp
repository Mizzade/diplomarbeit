#include <opencv2/opencv.hpp>
#include "/home/tilde/TILDE/c++/Lib/src/libTILDE.hpp"
#include <getopt.h>

/**
 * How to compile inside docker image:
 * g++ -std=c++11 -o use_tilde use_tilde.cpp /home/tilde/TILDE/c++/build/Lib/libTILDE_static.a -L/usr/local/lib -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_features2d
 */

/**
 * Example usage:
 * ./use_tilde --imageDir /home/tilde/TILDE/data --outputDir output --fileName testImage.png --filterPath /home/tilde/TILDE/c++/Lib/filters --filterName Mexico.txt
 */
int main(int argc, char *argv[])
{
    using namespace cv;

    std::string imageDir;
    std::string outputDir;
    std::string fileName;
    std::string filterPath;
    std::string filterName;

    static const struct option long_options[] = {
      {"imageDir", required_argument, 0 , 'a'},
      {"outputDir", required_argument, 0, 'b'},
      {"fileName", required_argument, 0 , 'c'},
      {"filterPath", required_argument, 0, 'd'},
      {"filterName", required_argument, 0, 'e'},
      0
    };

    while (1) {
      int index = -1;
      struct option *opt = 0;
      int result = getopt_long_only(argc, argv, "abcde", long_options, &index);

      if (result == -1) break; /*end of list */

      switch (result) {
        case 'a':
          // printf("--imageDir was specified. Arg: <%s>\n", optarg);
          imageDir = optarg;
          break;
        case 'b':
          // printf("--outputDir was specified. Arg: <%s>\n", optarg);
          outputDir = optarg;
          break;
        case 'c':
          // printf("--fileName was specified. Arg: <%s>\n", optarg);
          fileName = optarg;
          break;
        case 'd':
          // printf("--filterPath was specified. Arg: <%s>\n", optarg);
          filterPath = optarg;
          break;
        case 'e':
          // printf("--filterName was specified. Arg: <%s>\n", optarg);
          filterName = optarg;
          break;
        default:
          break;
      }
    };

    while (optind < argc) {
      printf("other parameter: <%s>\n", argv[optind++]);
    }

    try {
      std::string pathToFilter = filterPath + "/" + filterName;
      std::string pathToImage = imageDir + "/" + fileName;
      std::string pathToOutputFile = outputDir + "/" + fileName + ".yml";

      // Load image
      Mat I = imread(pathToImage);
      if (I.data == 0) throw std::runtime_error("Image not found!");

      /*
      td::vector<cv::KeyPoint> getTILDEKeyPoints(cv::Mat image,
			  		  std::string pathFilter,
					    bool useApprox,
					    bool sortKeypoints,
					    bool keepPositiveOnly,
					    cv::Mat *score)

      <<input parameters>>:

      - image: a openCV U8C3 Mat object representing the image to process
      - pathFilter: a std string object giving the name of the filter to apply
      - useApprox:  a boolean  indicating to  use TILDE  (false) or  the approximated
        version of TILDE (true).
      - sortKeypoints:  (true) we  sort the  keypoints by  decreasing order  of their
        scores
      - keepPositiveOnly: (true) only the keypoints with positive score are returned
      - score: a  pointer to an  openCV Mat  image, if the  pointer is not  null, the
        score map is retuned in addition to the keypoints
      */
     // If you want the score map as well, you have to create an openCV Mat
     // mat object for it and pass it as last parameter to the function
     // as reference: (&score) instead of NULL
      vector<KeyPoint> kps = getTILDEKeyPoints(I, pathToFilter, false, true, true, NULL);

      // Der Ordner in dem die .yml File gespeichert wird, muss bereits exisiteren.
      FileStorage fs(pathToOutputFile, FileStorage::WRITE);

      // TODO: Schreibe zusätzlich noch Name des Bildes n die .yaml File
      // und den Namen des Detektors samt benutzten Filter.

      // A Keypoint has 7 values:
      // cv.KeyPoint(	x, y, _size[, _angle[, _response[, _octave[, _class_id]]]])
      // See for more details: https://docs.opencv.org/3.4/d2/d29/classcv_1_1KeyPoint.html

      write(fs, "keypoints", kps);
      fs.release();


      std::cout << "Executed TILDE for file " << pathToImage << ".\nResults are saved in " << pathToOutputFile << std::endl;
    }
    catch (std::exception &e) {
      std::cout << "ERROR: " << e.what() << "\n";
    }
    return 0;
}
