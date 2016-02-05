#include"DetectCorners.h"

DetectCorners::DetectCorners()
{}

DetectCorners::~DetectCorners()
{}

DetectCorners::DetectCorners(cv::Mat img)
{
	inputimg = img.clone();
	bool result = findChessboardCorners(inputimg, ChessBoardSize, allcorners);
	//corner position in sub pixel
	cv::cornerSubPix(inputimg, allcorners, cv::Size(5, 5), cv::Size(-1, -1), cv::TermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 30, 0.01));
	fourcorners.resize(4);
	fourcorners[0] = allcorners[0];
	fourcorners[1] = allcorners[8];
	fourcorners[2] = allcorners[53];
	fourcorners[3] = allcorners[45];
	DrawCrossHair(inputimg, fourcorners[3]);
	cv::drawChessboardCorners(inputimg, ChessBoardSize, allcorners, result);
	cv::imshow("hehe", inputimg);
	cv::waitKey();
}

std::vector<cv::Point2f> DetectCorners::getAllcorners()
{
	return allcorners;
}

std::vector<cv::Point2f> DetectCorners::getfourcorners()
{
	return fourcorners;
}