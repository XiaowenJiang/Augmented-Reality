#include<iostream>
#include<opencv2\calib3d\calib3d.hpp>
#include<core\core.hpp>
#include<opencv2\highgui\highgui.hpp>
#include<opencv2\imgproc\imgproc.hpp>
#include<features2d\features2d.hpp>
#include<opencv2\ml\ml.hpp>
#include<video\tracking.hpp>
#include<vector>
#include"Utils.h"
#include"Calibration.h"
#include"SURFMatch.h"
#include"EstimateCameraPose.h"
#include"ProjectPoints.h"
#include"DetectCorners.h"

using namespace std;
using namespace cv;

int main()
{
	try
	{
		//STEP 1 : Calibration
		/*Calibration calibration;
		calibration.FindCorners();
		calibration.Calibrate();
		calibration.Undistort();
		calibration.computeReprojectionErrors();*/

		//get intrinsic matrix from YML record
		Mat intrinsic = GetFromYML("CalibrationResults.yml", "intrinsic");
		Mat dist_coeff = GetFromYML("CalibrationResults.yml", "distortion_coeff");

		//STEP 2 : Read images and find matches
		int img_num = 2;
		vector<Mat> inputimgs;
		vector<DetectCorners> detectcorners;
		vector<vector<Point2f>> chess_all_corners;
		vector<vector<Point2f>> chess_four_corners;
		vector<vector<KeyPoint>> chess_all_corners_key;
		chess_all_corners_key.resize(2);

		//STEP 2.1 : Find chessboard corners
		for (int i = 1; i <= img_num; i++)
		{
			Mat input = ReadImg(i, "inputimgs");
			resize(input, input, Size(), 0.2, 0.2);
			inputimgs.push_back(input);
			detectcorners.push_back(DetectCorners(input));
			chess_all_corners.push_back(detectcorners[i - 1].getAllcorners());
			chess_four_corners.push_back(detectcorners[i - 1].getfourcorners());
			KeyPoint::convert(chess_all_corners[i-1], chess_all_corners_key[i-1]);
		}
		Mat img_matches;
	
		//STEP 2.2 : Find other match points using surf
		SURFMATCH match(inputimgs[0],inputimgs[1]);
		vector<int> empty;
		match.match(match.getKeypoints1(), empty);

		vector<Point2f> matchedpoints1 = match.getMatchedPoints1();
		vector<Point2f> matchedpoints2 = match.getMatchedPoints2();
		
		//STEP 2.3 : Conbine corners and other keypoints into matches
		vector<DMatch> allmatches;
		int i = 0;
		int originmatchedsize = matchedpoints1.size();
		for (i; i < originmatchedsize; i++)
		{
			DMatch m;
			m.queryIdx = i;
			m.trainIdx = i;
			allmatches.push_back(m);
		}
		
		for (i=originmatchedsize; i < chess_all_corners[0].size()+originmatchedsize; i++)
		{
			matchedpoints1.push_back(chess_all_corners[0][i-originmatchedsize]);
			matchedpoints2.push_back(chess_all_corners[1][i-originmatchedsize]);
			DMatch m;
			m.queryIdx = i;
			m.trainIdx = i;
			allmatches.push_back(m);
		}

		for (i = 0; i < chess_all_corners[0].size(); i++)
		{
			DMatch m;
			m.queryIdx = i;
			m.trainIdx = i;
			allmatches.push_back(m);
		}

		vector<vector<KeyPoint>> testpointsss;
		testpointsss.resize(2);
		KeyPoint::convert(matchedpoints1, testpointsss[0]);
		KeyPoint::convert(matchedpoints2, testpointsss[1]);
		
		drawMatches(inputimgs[0], testpointsss[0], inputimgs[1], testpointsss[1], allmatches, img_matches, cv::Scalar::all(-1), cv::Scalar::all(-1), std::vector< char >(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
		imshow("matched_points", img_matches);
		imwrite("matched_image.jpg", img_matches);
		waitKey();


		//STEP 3 : 3D reconstruct using the matchedpoints

		//STEP 3.1 : get 3D points
		EstimateCameraPose estimate(matchedpoints1, matchedpoints2, intrinsic);
		//EstimateCameraPose estimate(chess_all_corners[0], chess_all_corners[1], intrinsic);
		estimate.calcCameraPoseFromE();
		vector<Point3f> global3d = estimate.get3Dpoints();

		//STEP 3.2 :
		//find the four 3D corners of chessboard
		vector<Point3f> fourcorners3d;
		fourcorners3d.push_back(global3d[originmatchedsize]);
		fourcorners3d.push_back(global3d[originmatchedsize + 8]);
		fourcorners3d.push_back(global3d[originmatchedsize + 53]);
		fourcorners3d.push_back(global3d[originmatchedsize + 45]);
		
		//calculate the normal vector of the chessboard planar
		Point3f a = fourcorners3d[1] - fourcorners3d[0];
		Point3f b = fourcorners3d[2] - fourcorners3d[1];
		Point3f m = a.cross(b);
		vector<float> m_v = { m.x, m.y, m.z };
		//normalized normal vector of chessboard planar 
		normalize(m_v, m_v);
		m = -Point3f(m_v[0], m_v[1], m_v[2]);
		//width of chessboard in 3D coordinates
		double width = norm(Mat(fourcorners3d[1]), Mat(fourcorners3d[0]));
		//height of chessboard in 3D coordinates
		double height = norm(Mat(fourcorners3d[2]), Mat(fourcorners3d[1]));
		vector<Point3f> otherfourvertices;
		otherfourvertices.resize(4);

		//STEP 3.3 :
		//draw a 3D cuiboid with the height equals to 1/2 height of chessboard
		for (int i = 0; i < 4; i++)
		{
			otherfourvertices[i] = fourcorners3d[i] + m*height*0.5;
		}
		fourcorners3d.insert(fourcorners3d.end(), otherfourvertices.begin(), otherfourvertices.end());
		vector<Mat> all_rotations;
		vector<Mat> all_translations;
		/*Mat R, t;
		solvePnPRansac(global3d, matchedpoints1, intrinsic, dist_coeff,
		R, t);
		Mat rotation33;
		cv::Rodrigues(R, rotation33);*/
		//R , t of the first camera
		Mat Identity = (Mat)Matx33d(1, 0, 0, 0, 1, 0, 0, 0, 1);
		Mat zerotrans = (Mat)Matx13d(0, 0, 0);
		all_rotations.push_back(Identity);
		all_translations.push_back(zerotrans);
		all_rotations.push_back(estimate.getRotation());
		all_translations.push_back(estimate.getTranslation());
		for (int i = 0; i <= 1; i++)
		{
			ProjectPoints projectvertices(inputimgs[i], fourcorners3d, all_rotations[i], all_translations[i],
				intrinsic, dist_coeff);
			vector<Point2f> vertices = projectvertices.getProjectpoints();

			//show image on a face
			Mat picture = imread("tao.jpeg",0);
			//four corner points of the input picture
			vector<Point2f> picpoints(4);
			vector<Point2f> facepoints{ vertices[7], vertices[4], vertices[5], vertices[6] };

			picpoints[0] = Point2f(0, 0);
			picpoints[1] = Point2f(picture.cols, 0);
			picpoints[2] = Point2f(picture.cols, picture.rows);
			picpoints[3] = Point2f(0, picture.rows);
			//compute the transform matrix and warp the bird picture 
			Mat trans = getPerspectiveTransform(picpoints, facepoints);
			Mat warpedpic;
			warpPerspective(picture, warpedpic, trans, inputimgs[0].size());

			Mat image_to_draw = inputimgs[i].clone();
			for (int i = 0; i < vertices.size(); i++)
			{
				DrawCrossHair(image_to_draw, vertices[i]);
			}
			for (int i = 0; i < 4; i++)
			{
				line(image_to_draw, vertices[(i + 1) % 4], vertices[i], cv::Scalar(255, 0, 0), 1.2);
				line(image_to_draw, vertices[4 + (i + 1) % 4], vertices[4 + i], cv::Scalar(255, 0, 0), 1.2);
				line(image_to_draw, vertices[i], vertices[i + 4], cv::Scalar(255, 0, 0), 1.2);
			}

			//traverse through the image and change the 
			//chessboard part to the input cockatiel picture
			for (int j = 0; j < image_to_draw.rows; j++)
			{
				for (int i = 0; i < image_to_draw.cols; i++)
				{
					if (warpedpic.at<uchar>(j,i)!=0)
					{
						image_to_draw.at<uchar>(j, i) = warpedpic.at<uchar>(j, i);
					}
				}
			}
			string windowname = "result" + to_string(i);
			imshow(windowname, image_to_draw);
			WriteImg(i, "resultimgs", "", image_to_draw);
			waitKey();
		}
		
	}
	catch (Exception & e)
	{
		cout << e.msg << endl;
	}
	return 0;
}