// DIPhw1.cpp : 定義主控台應用程式的進入點。
//

#include "stdafx.h"
#include <iostream>
#include <iomanip>
#include <ctime>
#include <chrono>
#include <math.h>
#include "opencv2\opencv.hpp"

using namespace std;
using namespace cv;

inline Vec3b BilinearInterpolation(const Vec3b &f0, const Vec3b &f1, const Vec3b &f2, const Vec3b &f3, double alpha, double beta)
{
	return Vec3s(f0) + alpha * (Vec3s(f1) - Vec3s(f0)) + beta * (Vec3s(f2) - Vec3s(f0)) + alpha * beta * (Vec3s(f0) - Vec3s(f1) - Vec3s(f2) + Vec3s(f3));
}

inline void FiniteDeference(const Mat &mInput, Mat &mOutput, bool bDirection)
{
	mOutput = Mat(mInput.rows, mInput.cols, CV_64FC3);
	for (int u = 0; u < mOutput.rows; u++)
	{
		for (int v = 0; v < mOutput.cols; v++)
		{
			if (!bDirection) // u
			{
				int u_n = u - 1;
				int u_p = u + 1;
				if (u_n < 0 && u_p < mOutput.rows)
					mOutput.at<Vec3d>(u, v) = (mInput.at<Vec3d>(u_p, v) - mInput.at<Vec3d>(u, v));
				else if (u_p >= mOutput.rows && u_n >= 0)
					mOutput.at<Vec3d>(u, v) = (mInput.at<Vec3d>(u, v) - mInput.at<Vec3d>(u_n, v));
				else
					mOutput.at<Vec3d>(u, v) = (mInput.at<Vec3d>(u_p, v) - mInput.at<Vec3d>(u_n, v)) / 2.0;
			}
			else // v
			{
				int v_n = v - 1;
				int v_p = v + 1;
				if (v_n < 0 && v_p < mOutput.cols)
					mOutput.at<Vec3d>(u, v) = (mInput.at<Vec3d>(u, v_p) - mInput.at<Vec3d>(u, v));
				else if (v_p >= mOutput.cols && v_n >= 0)
					mOutput.at<Vec3d>(u, v) = (mInput.at<Vec3d>(u, v) - mInput.at<Vec3d>(u, v_n));
				else
					mOutput.at<Vec3d>(u, v) = (mInput.at<Vec3d>(u, v_p) - mInput.at<Vec3d>(u, v_n)) / 2.0;
			}
		}
	}
}

void Resize(const Mat &mInput, Mat &mOutput, double dScalingFactor, string sOption)
{
	int iHeight = round(mInput.rows * dScalingFactor);
	int iWidth = round(mInput.cols * dScalingFactor);
	mOutput = Mat(iHeight, iWidth, CV_8UC3);

	Mat A = (Mat_<double>(16, 16) <<
		1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		-3, 3, 0, 0, -2, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		2, -2, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, -3, 3, 0, 0, -2, -1, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 2, -2, 0, 0, 1, 1, 0, 0,
		-3, 0, 3, 0, 0, 0, 0, 0, -2, 0, -1, 0, 0, 0, 0, 0,
		0, 0, 0, 0, -3, 0, 3, 0, 0, 0, 0, 0, -2, 0, -1, 0,
		9, -9, -9, 9, 6, 3, -6, -3, 6, -6, 3, -3, 4, 2, 2, 1,
		-6, 6, 6, -6, -3, -3, 3, 3, -4, 4, -2, 2, -2, -2, -1, -1,
		2, 0, -2, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 2, 0, -2, 0, 0, 0, 0, 0, 1, 0, 1, 0,
		-6, 6, 6, -6, -4, -2, 4, 2, -3, 3, -3, 3, -2, -1, -2, -1,
		4, -4, -4, 4, 2, 2, -2, -2, 2, -2, 2, -2, 1, 1, 1, 1
		);

	Mat u_prime, v_prime, u_prime_v_prime;
	if (sOption == "bicubic")
	{
		Mat mInput64F;
		mInput.convertTo(mInput64F, CV_64FC3);
		FiniteDeference(mInput64F, u_prime, 0);
		FiniteDeference(mInput64F, v_prime, 1);
		FiniteDeference(u_prime, u_prime_v_prime, 1);
	}

	for (int u = 0; u < mOutput.rows; u++)
	{
		for (int v = 0; v < mOutput.cols; v++)
		{
			// Convert P to image_ratio to p
			double duInput = (1 + 2 * u - dScalingFactor) / (2 * dScalingFactor);
			double dvInput = (1 + 2 * v - dScalingFactor) / (2 * dScalingFactor);

			int iFlooruInput = floor(duInput);
			int iCeiluInput = iFlooruInput + 1;
			int iFloorvInput = floor(dvInput);
			int iCeilvInput = iFloorvInput + 1;

			double alpha = dvInput - iFloorvInput;
			double beta = duInput - iFlooruInput;

			// Boundary check
			iFlooruInput = (iFlooruInput < 0) ? iFlooruInput + 1 : iFlooruInput;
			iFloorvInput = (iFloorvInput < 0) ? iFloorvInput + 1 : iFloorvInput;
			iCeiluInput = (iCeiluInput >= mInput.rows) ? iCeiluInput - 1 : iCeiluInput;
			iCeilvInput = (iCeilvInput >= mInput.cols) ? iCeilvInput - 1 : iCeilvInput;

			if (sOption == "bilinear")
			{
				Vec3b f0 = mInput.at<Vec3b>(iFlooruInput, iFloorvInput);
				Vec3b f1 = mInput.at<Vec3b>(iFlooruInput, iCeilvInput);
				Vec3b f2 = mInput.at<Vec3b>(iCeiluInput, iFloorvInput);
				Vec3b f3 = mInput.at<Vec3b>(iCeiluInput, iCeilvInput);

				mOutput.at<Vec3b>(u, v) = BilinearInterpolation(f0, f1, f2, f3, alpha, beta);
			}
			else if (sOption == "bicubic")
			{
				Mat u_vec = (Mat_<double>(1, 4) << 1, beta, beta * beta, beta * beta * beta);
				Mat v_vec = (Mat_<double>(4, 1) << 1, alpha, alpha * alpha, alpha * alpha * alpha);
				for (int channel = 0; channel < 3; channel++)
				{
					Mat b = (Mat_<double>(16, 1) <<
						mInput.at<Vec3b>(iFlooruInput, iFloorvInput)[channel],
						mInput.at<Vec3b>(iCeiluInput, iFloorvInput)[channel],
						mInput.at<Vec3b>(iFlooruInput, iCeilvInput)[channel],
						mInput.at<Vec3b>(iCeiluInput, iCeilvInput)[channel],
						u_prime.at<Vec3d>(iFlooruInput, iFloorvInput)[channel],
						u_prime.at<Vec3d>(iCeiluInput, iFloorvInput)[channel],
						u_prime.at<Vec3d>(iFlooruInput, iCeilvInput)[channel],
						u_prime.at<Vec3d>(iCeiluInput, iCeilvInput)[channel],
						v_prime.at<Vec3d>(iFlooruInput, iFloorvInput)[channel],
						v_prime.at<Vec3d>(iCeiluInput, iFloorvInput)[channel],
						v_prime.at<Vec3d>(iFlooruInput, iCeilvInput)[channel],
						v_prime.at<Vec3d>(iCeiluInput, iCeilvInput)[channel],
						u_prime_v_prime.at<Vec3d>(iFlooruInput, iFloorvInput)[channel],
						u_prime_v_prime.at<Vec3d>(iCeiluInput, iFloorvInput)[channel],
						u_prime_v_prime.at<Vec3d>(iFlooruInput, iCeilvInput)[channel],
						u_prime_v_prime.at<Vec3d>(iCeiluInput, iCeilvInput)[channel]
						);

					Mat a = A * b;

					Mat out = u_vec * (a.reshape(0, 4).t()) * v_vec;
					// Clipping
					uchar p = (out.at<double>(0, 0) > 255) ? 255 : uchar(out.at<double>(0, 0));
					p = (out.at<double>(0, 0) < 0) ? 0 : p;
					mOutput.at<Vec3b>(u, v)[channel] = p;
				}
			}
		}
	}
}

int main(int argc, char* argv[])
{
	Mat mInput = imread(argv[1]);

	/*Mat mInput = Mat::zeros(5, 5, CV_8UC3);

	// Test pattern 1
	mInput.at<Vec3b>(0, 0) = Vec3b(41, 255, 205);
	mInput.at<Vec3b>(0, 1) = Vec3b(0, 196, 255);
	mInput.at<Vec3b>(0, 2) = Vec3b(0, 103, 255);
	mInput.at<Vec3b>(0, 3) = Vec3b(0, 103, 255);
	mInput.at<Vec3b>(0, 4) = Vec3b(255, 176, 0);

	mInput.at<Vec3b>(1, 0) = Vec3b(0, 7, 241);
	mInput.at<Vec3b>(1, 1) = Vec3b(0, 0, 127);
	mInput.at<Vec3b>(1, 2) = Vec3b(255, 76, 0);
	mInput.at<Vec3b>(1, 3) = Vec3b(205, 255, 41);
	mInput.at<Vec3b>(1, 4) = Vec3b(0, 196, 255);

	mInput.at<Vec3b>(2, 0) = Vec3b(255, 76, 0);
	mInput.at<Vec3b>(2, 1) = Vec3b(0, 103, 255);
	mInput.at<Vec3b>(2, 2) = Vec3b(255, 76, 0);
	mInput.at<Vec3b>(2, 3) = Vec3b(121, 255, 124);
	mInput.at<Vec3b>(2, 4) = Vec3b(0, 103, 255);

	mInput.at<Vec3b>(3, 0) = Vec3b(255, 76, 0);
	mInput.at<Vec3b>(3, 1) = Vec3b(0, 7, 241);
	mInput.at<Vec3b>(3, 2) = Vec3b(255, 176, 0);
	mInput.at<Vec3b>(3, 3) = Vec3b(0, 103, 255);
	mInput.at<Vec3b>(3, 4) = Vec3b(255, 76, 0);

	mInput.at<Vec3b>(4, 0) = Vec3b(0, 7, 241);
	mInput.at<Vec3b>(4, 1) = Vec3b(241, 0, 0);
	mInput.at<Vec3b>(4, 2) = Vec3b(0, 7, 241);
	mInput.at<Vec3b>(4, 3) = Vec3b(205, 255, 41);
	mInput.at<Vec3b>(4, 4) = Vec3b(205, 255, 41);

	// Test pattern 2
	mInput.at<Vec3b>(0, 0) = Vec3b(0, 0, 0);
	mInput.at<Vec3b>(0, 1) = Vec3b(255, 255, 255);
	mInput.at<Vec3b>(0, 2) = Vec3b(0, 0, 0);
	mInput.at<Vec3b>(0, 3) = Vec3b(255, 255, 255);
	mInput.at<Vec3b>(0, 4) = Vec3b(0, 0, 0);

	mInput.at<Vec3b>(1, 0) = Vec3b(255, 255, 255);
	mInput.at<Vec3b>(1, 1) = Vec3b(0, 0, 0);
	mInput.at<Vec3b>(1, 2) = Vec3b(255, 255, 255);
	mInput.at<Vec3b>(1, 3) = Vec3b(0, 0, 0);
	mInput.at<Vec3b>(1, 4) = Vec3b(255, 255, 255);

	mInput.at<Vec3b>(2, 0) = Vec3b(0, 0, 0);
	mInput.at<Vec3b>(2, 1) = Vec3b(255, 255, 255);
	mInput.at<Vec3b>(2, 2) = Vec3b(0, 0, 0);
	mInput.at<Vec3b>(2, 3) = Vec3b(255, 255, 255);
	mInput.at<Vec3b>(2, 4) = Vec3b(0, 0, 0);

	mInput.at<Vec3b>(3, 0) = Vec3b(255, 255, 255);
	mInput.at<Vec3b>(3, 1) = Vec3b(0, 0, 0);
	mInput.at<Vec3b>(3, 2) = Vec3b(255, 255, 255);
	mInput.at<Vec3b>(3, 3) = Vec3b(0, 0, 0);
	mInput.at<Vec3b>(3, 4) = Vec3b(255, 255, 255);

	mInput.at<Vec3b>(4, 0) = Vec3b(0, 0, 0);
	mInput.at<Vec3b>(4, 1) = Vec3b(255, 255, 255);
	mInput.at<Vec3b>(4, 2) = Vec3b(0, 0, 0);
	mInput.at<Vec3b>(4, 3) = Vec3b(255, 255, 255);
	mInput.at<Vec3b>(4, 4) = Vec3b(0, 0, 0);*/

	Mat mOutput;
	std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
	Resize(mInput, mOutput, stod(argv[3]), argv[4]);
	std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
	cout << argv[4] << ", scaling factor = " << argv[3] << " took " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << "us." << endl;
	imwrite(argv[2], mOutput);

	/*Resize(mInput, mOutput, 100, "bicubic");
	imwrite("testori.png", mOutput);*/

	// Output OpneCV resize for comparison
	/*Mat mOpenCVOutput;
	cv::resize(mInput, mOpenCVOutput, Size(0, 0), 100, 100, INTER_CUBIC);
	imwrite("bilinearOpenCV.png", mOpenCVOutput);*/

	return 0;
}

