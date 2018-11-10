#include <opencv2/opencv.hpp>
#include "/home/tilde/TILDE/c++/Lib/src/libTILDE.hpp"
#include <getopt.h>
#include <fstream>

/**
 * How to compile inside docker image:
 * g++ -std=c++11 -o use_tilde use_tilde.cpp /home/tilde/TILDE/c++/build/Lib/libTILDE_static.a -L/usr/local/lib -lopencv_core -lopencv_imgcodecs -lopencv_highgui -lopencv_imgproc -lopencv_features2d
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
      size_t lastindex = fileName.find_last_of(".");
      string rawname = fileName.substr(0, lastindex);

      std::string pathToFilter = filterPath + "/" + filterName;
      std::string pathToImage = imageDir + "/" + fileName;
      std::string pathToOutputFile = outputDir + "/keypoints/kpts_tilde__" + rawname +
        "_TILDE.csv";

      // Filename for score map.
      std::string pathToScoreMap = outputDir + "/scores/scores_tilde__" + rawname +
        "_TILDE.csv";

      // Load image
      Mat I = imread(pathToImage);

      // Get width and height of the image
      int height = I.rows;
      int width = I.cols;

      // Create Mat for probability scores
      Mat scores = Mat::zeros(height, width, CV_32FC1);


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
      vector<KeyPoint> kpts = getTILDEKeyPoints(I, pathToFilter, false, true, true, &scores);

      std::vector<cv::Point2f> point2f_vector; //We define vector of point2f
      cv::KeyPoint::convert(kpts, point2f_vector, std::vector< int >()); //Then we use this nice function from OpenCV to directly convert from KeyPoint vector to Point2f vector
      cv::Mat kpts_coordinates(point2f_vector); //We simply cast the Point2f vector into a cv::Mat as Appleman1234 did

      cv::Size size_kpts = kpts_coordinates.size();
      int num_kpts = size_kpts.height;
      int num_cols = size_kpts.width;

      std::string meta_text = "# " + std::to_string(height) + ", " +
        std::to_string(width) + ", " + std::to_string(num_kpts) + ", " +
        std::to_string(num_cols + 1);


      // Save keypoints
      ofstream myfile;
      myfile.open(pathToOutputFile.c_str());
      myfile << meta_text << std::endl << cv::format(kpts_coordinates, cv::Formatter::FMT_CSV) << std::endl;
      myfile.close();

      // Save score map
      // The score map is exaclty as large as the image and contains the probability
      // value of that pixel to be a keypoint.
      std::string score_meta_text = "# " + std::to_string(height) + ", " +
        std::to_string(width);
      myfile.open(pathToScoreMap.c_str());
      myfile << score_meta_text << std::endl << cv::format(scores, cv::Formatter::FMT_CSV) << std::endl;
      myfile.close();

    }
    catch (std::exception &e) {
      std::cout << "ERROR: " << e.what() << "\n";
    }
    return 0;
}
