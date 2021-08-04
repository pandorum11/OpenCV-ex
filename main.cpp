#include <iostream>
#include <exception>
#include <math.h>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/matx.hpp>

#include <deque>
#include <algorithm>
#include <memory>
#include <vector>
#include <stdio.h> 
#include <ctime>
#include <numeric>
#include <functional>


#include <fstream>

//test

void write_points(std::vector<std::vector<cv::Point>> contours, int counter, std::string file_) {
	std::ofstream out(file_ + ".txt", std::ios::app);
	out << std::to_string(counter) + "frame number" << std::endl;
	for (size_t i = 0; i < contours.size(); i++) {
		out << "Figure_number_" + std::to_string(i + 1) << std::endl;
		for (size_t j = 0; j < 4; j++) {
			out << "coords x-> " + std::to_string(contours[i][j].x) + " y-> " + std::to_string(contours[i][j].y) << std::endl;
		}
	}
	out.close();
}


void draw_points(std::vector<std::vector<cv::Point>> contours, int counter, cv::Mat frame) {
	for (size_t i = 0; i < contours.size(); i++) {
		for (size_t j = 0; j < 4; j++) {
			cv::Vec3b color = frame.at<cv::Vec3b>(cv::Point(contours[i][j].x, contours[i][j].y));
			color[0] = 255;
			color[1] = 255;
			color[2] = 255;
			frame.at<cv::Vec3b>(cv::Point(contours[i][j].x, contours[i][j].y)) = color;
		}
	}
	//cv::imwrite("E:/Vid1M_build/BTS_N1_TT/IMGCONTOURS1/Frame_" + std::to_string(counter) + ".png", frame);
}

/////////////////////////////////////////

extern const int count_top(4); // ���������� ������ ������
extern const int MIN_DISTANCE(35); // ����������� ��������� ����� ����� ������� ��� ����������� �������� �����������������
extern const int median_filter_window_width(13); // ������ ( � ������ ) �������� ���� ���������� �������, ��������
extern const int scalling(2); // ���������� ����������������
extern std::string video_path("video.mp4"); // ���� � ��������� �����
extern std::string img_path_1("beavis.png"); // ������
extern std::string img_path_2("butthead.png"); // � ������ ��������
extern std::string video_out_path("E:/Vid1M_build/BTS_N1_TT/VID/video_out_101010");

/////////////////////////////////////////

// ������ ��������� ����� �������
inline int getDistance(const cv::Point& a, const cv::Point& b) {
	const cv::Point d = a - b;
	return (int)sqrt((double)(d.x * d.x + d.y * d.y));
}

// ����� ���������� �������
class MedianProcessing {
private:
	unsigned long int full_lenth_counter_activations;	// ������ ��������� ���������� ������, � ���� ����� �� ������������
	size_t counter_flag;								// ������ �������� ���� (������� ������ ��� �������)
	size_t rect_count;									// ���������� ���������������
	std::deque<std::vector<std::vector<cv::Point>>> working_list; // ��������� �������� ����

public:
	// �����������: ������������� - (������ ����, ������ ����)
	MedianProcessing(const unsigned short weight_of, std::vector<std::vector<cv::Point>>& INPUT) {
		if (weight_of < 1 || weight_of > 254 || !(weight_of % 2))
			throw std::exception("Non correct size of floating window, should be non pair");
	
		this->full_lenth_counter_activations = 0;
		this->counter_flag = (size_t)weight_of;
		this->rect_count = INPUT.size();
		// ���������� ���������� ���������� ����
		for (size_t i = 0; i < counter_flag; i++) {
			working_list.push_front(INPUT);
		}
	}


	// ������� ������������ ������� �� ������ �����
	std::vector<std::vector<cv::Point>> FindMedian(std::vector<std::vector<cv::Point>>& INPUT) {

		int x, y;

		full_lenth_counter_activations++;

		working_list.pop_back();
		working_list.push_front(INPUT);
		
		
		// �������� ��������������� ������ ������ ��������������
		std::vector<std::vector<cv::Point>> final_for_curent;
		final_for_curent.reserve(2);
		// ��������� ������ ��������� 1 �����
		std::vector<cv::Point> processing_vector_Point;
		processing_vector_Point.reserve(4);

		// ���������� ��������� �������� �������� 1 ���������� ���� ��������������� �������� ����, (x � y)
		int temp_processing_array_x[median_filter_window_width];
		int temp_processing_array_y[median_filter_window_width];

		// �������� ���� ( � �������� ������� ���������� ��������������� )
		for (size_t i = 0; i < rect_count; i++) {
			// �������� �� ������ ��� ��������� �������
			auto final_for_curent_iterator = final_for_curent.begin();			
			// ���� �������� ������ ������
			for (size_t j = 0; j < count_top; j++) {
				// �������� ��� �������������� ������� 1-�� �������������� 
				auto processing_vector_Point_iterator = processing_vector_Point.begin();

				// ���� ������� �� ������ ����
				for (size_t k = 0; k < counter_flag; k++) {
					temp_processing_array_x[k] = working_list[k][i][j].x;
					temp_processing_array_y[k] = working_list[k][i][j].y;
				}
				// ������������� � ���������� ��������� ��� x � ��� y
				x = INPUT[i][j].x;
				y = INPUT[i][j].y;

				std::sort(std::begin(temp_processing_array_x), std::end(temp_processing_array_x));
				std::sort(std::begin(temp_processing_array_y), std::end(temp_processing_array_y));

				// �������� ����� � ������������� ������
				processing_vector_Point.emplace(processing_vector_Point_iterator + j, cv::Point(x, y));
			}
			// �������� ����� � ��������� ��������������� ������
			final_for_curent.emplace(final_for_curent_iterator + i, processing_vector_Point);
			processing_vector_Point.clear();
		}
		return final_for_curent;
	}
};

// ���������� ������� �����
void zooming_sides(std::vector<std::vector<cv::Point>>& squares) {
	for (size_t i = 0; i < squares.size(); i++) {

		squares[i][0].x -= scalling;
		squares[i][0].y -= scalling;

		squares[i][1].x -= scalling;
		squares[i][1].y += scalling;

		squares[i][2].x += scalling;
		squares[i][2].y += scalling;

		squares[i][3].x += scalling;
		squares[i][3].y -= scalling;

	}
}


// ������� ������ �������

struct x_sorter // ��������� �� x
{
	bool operator ()(const cv::Point& a, const cv::Point& b)
	{
		return (a.x < b.x);
	}
};

struct y_sorter // ��������� �� y
{
	bool operator ()(const cv::Point& a, const cv::Point& b)
	{
		return (a.y > b.y);
	}
};


//�������������� ���������� ��������������
std::vector<std::vector<cv::Point>> valid_horizontal_orientation(std::vector<std::vector<cv::Point>> squares) {

	std::vector<std::vector<cv::Point>> squares_final;
	squares_final.reserve(2);
	std::vector<cv::Point> squares_tmp;
	std::vector<cv::Point > temp_l, temp_r, temp_all;
	

	for (size_t i = 0; i < squares.size(); i++) {
		sort(squares[i].begin(), squares[i].end(), x_sorter());

		temp_l.push_back(squares[i][0]);
		temp_l.push_back(squares[i][1]);
		temp_r.push_back(squares[i][2]);
		temp_r.push_back(squares[i][3]);

		sort(temp_r.begin(), temp_r.end(), y_sorter());
		sort(temp_l.begin(), temp_l.end(), y_sorter());
		temp_all.push_back(temp_l[1]);
		temp_all.push_back(temp_l[0]);
		temp_all.push_back(temp_r[0]);
		temp_all.push_back(temp_r[1]);

		squares_final.push_back(temp_all);

		temp_l.clear();
		temp_r.clear();
		temp_all.clear();
	}
	return squares_final;
}


std::vector<std::vector<cv::Point>> getContours(const cv::Mat &mask, int counter) {

	// ��������� ����� ���������������
	std::vector<std::vector<cv::Point>> contours, squares;
	int squares_counter___ = 0;
	squares.reserve(15);
	
	// ������� ���������� ��������

	cv::findContours(mask, contours, \
		cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);

	if (counter > 1223) write_points(contours, counter, "test_start");

	std::vector<cv::Point> approx, temp_r;
	cv::Point temp_point;
	for (size_t i = 0; i < contours.size(); i++) {

		// ������������, ����������� ��������
		cv::approxPolyDP(contours[i], approx, \
			cv::arcLength(contours[i], true) * 0.08, true);

		// ���������� �����, isContourConvex ����������� ������� ����� � ��������� ���������� ������
		if (approx.size() == count_top && cv::isContourConvex(approx)) // ����������� ��������� ������� 
															   // ����� � �������� �� ����������
			if(getDistance(approx[0], approx[1]) >= MIN_DISTANCE || \
				getDistance(approx[0], approx[3]) >= MIN_DISTANCE)
				squares.push_back(approx);
	}
	return squares;
}

// ���������� �������� � ��������, � �������� ��������� - ������ �� ������� �������� 
inline std::vector<cv::Point2f> imageToContour(const cv::Mat &img) {

	if (img.empty())
		throw std::exception("empty Mat obj passed to imageToContour func");

	std::vector<cv::Point2f> result;
	result.reserve(4);
	result.push_back(cv::Point2f((float)img.cols, 0.0));
	result.push_back(cv::Point2f((float)img.cols, (float)img.rows));
	result.push_back(cv::Point2f(0, (float)img.rows));
	result.push_back(cv::Point2f(0.0, 0.0));

	return result;
}

// ����������� ���� ����������
inline std::vector<cv::Point2f> convertToPoint2f(const std::vector<cv::Point> &v) {

	std::vector<cv::Point2f> result;
	result.reserve(v.size());

	for (auto p = v.cbegin(); p != v.cend(); ++p)
		result.push_back(cv::Point2f((float)p->x, (float)p->y));

	return result;
}

// ������� ������� �������� � ����
 void warpImageToContour(const cv::Mat &img, \
							cv::Mat &warped, \
							const std::vector<cv::Point> &contour,
							const cv::Size &warped_size) {

	auto orig_contour = imageToContour(img);
	auto final_countour = convertToPoint2f(contour);

	// ������ �����������, ����������� �������� ������� ����������� ��������
	cv::Mat M = cv::findHomography(orig_contour, final_countour); // = cv::getPerspectiveTransform(orig_contour, final_countour);
	
	CV_Assert(!M.empty());

	cv::warpPerspective(img, warped, M, warped_size);
}

cv::VideoWriter createVideoWriter(const cv::VideoCapture &cap) {
	// ������� ���������� ��������� ����� � ����� �� ���������� � ��������
	cv::VideoWriter writer;
	time_t time_comp;
	// int frame_rate = (int)(cap.get(cv::CAP_PROP_FPS));
	// int codec = (int)(cap.get(cv::CAP_PROP_FOURCC));
	int codec = cv::VideoWriter::fourcc('F', 'M', 'P', '4');// ������ ������
	
	int width = (int)(cap.get(cv::CAP_PROP_FRAME_WIDTH)); // ������
	int height = (int)(cap.get(cv::CAP_PROP_FRAME_HEIGHT)); // ������
	// ������

	struct tm* newtime;
	time_t ltime;

	/* Get the time in seconds */
	time(&ltime);
	/* Convert it to the structure tm */
	newtime = localtime(&ltime);

	//asctime(newtime) +

	writer.open(video_out_path + ".mp4", codec, 46.875, cv::Size(width, height), true);
	if (!writer.isOpened()) 
		throw std::logic_error("Could not open the output video file for write");

	return writer;
}

// ��������� ������� �������� ������
inline void skipFrames(cv::VideoCapture &cap, const size_t skipped_count) {

	size_t frame_counter = 0;
	bool success = true;
	for (; success && frame_counter < skipped_count; ++frame_counter)
		success = cap.grab();
	std::cout << "Skipped " << frame_counter << " frames\n";
}

// ������ �����, ������������ ����� ��������� ������
//(52, 94, 53),(58, 119, 76)
cv::Mat geTmask(cv::Mat frame, int frame_counter) {
	cv::Mat mask, img_HSV;
	// ������� �� ������� BRG � HSV
	cv::cvtColor(frame, img_HSV, cv::COLOR_BGR2HSV);
	CV_Assert(!img_HSV.empty());

	// ��������� ��������� ����� ������
	if (frame_counter < 835)
		cv::inRange(img_HSV, cv::Scalar(39, 86, 56), \
			cv::Scalar(85, 255, 255), mask);
	else
		cv::inRange(img_HSV, cv::Scalar(48, 72, 84), \
			cv::Scalar(80, 255, 255), mask);
	CV_Assert(!mask.empty());
	return mask;
}

// ���������� ����� (��� ����������� ������� ���������� ����������������) 
struct contour_sorter
{
	bool operator ()(const std::vector<cv::Point>& a, const std::vector<cv::Point>& b)
	{
		// ������������� ����� ������� ������
		return (a[0].x < b[0].x);
	}
};


int main() {
	try {
		// ������ ����� � ������ cap
		cv::VideoCapture cap(video_path);

		if (!cap.isOpened())
			throw std::exception("file not found");
		
		// �������� ������������� �������
		auto writer = createVideoWriter(cap);

		// ��������� ������ ������������� ��������
		cv::Mat beavis = cv::imread(img_path_1);
		CV_Assert(!beavis.empty());
		cv::Mat butthead = cv::imread(img_path_2);
		CV_Assert(!butthead.empty());
		
		size_t frame_counter = 0;
		// skipFrames(cap, 835);
		cv::Mat frame;
		cap >> frame;

		cv::waitKey();
		// ������������� ������ ������ �������
		MedianProcessing MedianWorkingObject(median_filter_window_width, getContours(geTmask(frame, 0), frame_counter));

		cv::Mat mask, beavis_warped, butthead_warped, warped, result;

		while (!frame.empty()) {

			std::cout << "Extracted frame: " << frame_counter << std::endl;
			mask = geTmask(frame, frame_counter); // ������ �����
			
			auto contours = getContours(mask, frame_counter);	  // ������ ��������
			
			// ������� �� ���� ������ ��� ������� �� ������ ��������� ����� ���-�� �����
			if (contours.size() != 2) {
				//cv::imwrite("E:/Vid1M_build/BTS_N1_TT/IMG_LASTDIST/Frame_MMMMM_" + std::to_string(frame_counter) + ".png", frame);
				//cv::imwrite("E:/Vid1M_build/BTS_N1_TT/IMG_LASTDIST_MASK/Frame_MMMMM_" + std::to_string(frame_counter) + ".png", mask);
				cap >> frame;
				++frame_counter;
				std::cout << "skipped" << std::endl;
				continue;
			}
			
			// ��������� ������� ���������� �����
			std::sort(contours.begin(), contours.end(), contour_sorter());

			// ��������� ������� ���������� ������
			contours = valid_horizontal_orientation(contours);

			// ������ ����� ��������� ������
			contours = MedianWorkingObject.FindMedian(contours);

			// ����������� ��������
			zooming_sides(contours);

			// ���������� �������� � �������
			warpImageToContour(beavis, beavis_warped, contours[0], \
				cv::Size(mask.cols, mask.rows));
			warpImageToContour(butthead, butthead_warped, contours[1], \
				cv::Size(mask.cols, mask.rows));

			// ������� �������� � 1 ������������� �����
			cv::addWeighted(beavis_warped, 1.0, butthead_warped, 1.0, \
				0.0, warped);

			// ������� ��������� � ������� �������������� �� ������
			cv::fillPoly(frame, contours[0], cv::Scalar(0, 0, 0));
			cv::fillPoly(frame, contours[1], cv::Scalar(0, 0, 0));

			// ��������������� ������� � ����
			cv::addWeighted(frame, 1.0, warped, 1.0, \
				0.0, result);

			// ������ ����� � �������� �����
			//if (frame_counter > 650 && frame_counter < 750) {
			//	//cv::imwrite("E:/Vid1M_build/BTS_N1_TT/IMG_LASTDIST/Frame_ " + std::to_string(frame_counter) + ".png", result);
			//	draw_points(contours, frame_counter, frame);
			//}
			//if (frame_counter > 20) break;
			
			//if (frame_counter > 1223) {
			//	//cv::imwrite("E:/Vid1M_build/BTS_N1_TT/IMG_LASTDIST/Frame_ " + std::to_string(frame_counter) + ".png", result);
			//	write_points(contours, frame_counter, "test_final");
			//}
			writer.write(result);
			
			cap >> frame;
			++frame_counter;
		}
		std::cout << "Completed\n";
		// ������ �����
		writer.release();
	}
	catch (std::exception &e) {
		std::cerr << e.what();
	}
	std::cin.get();
	return 0;
}