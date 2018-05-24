#include "cuda_runtime.h"  
#include "device_launch_parameters.h"  
#include <cuda.h>
#include"device_functions.h"

#include <stdio.h> 
#include<iostream>
#include <fstream>
#include <string>
#include "math.h"
#include<time.h>


//skiplist
#include <time.h>
#include <malloc.h>

#include <unordered_map>



using namespace std;
#define NEI_MAX	300	//最大允许的相似矩阵大小
#define PEOPLE_CAP 2500  //总人数
#define MOVIE_CAP  2500 //总电影数目
#define TEST_DATA_CAP 30000 //可分析的电影最大数
#define THREDS_NUM 1024 //块的最大线程数

double approximate(double a)
{
	if (a == 0)
		return 1;

	if (a > 5)
		return 5;

	//if (a < 2.5)
	//	return 1;

//	return a;
	int b = (int)a;

	if (a - (double)b > 0.6)
		return (double)+ 1;
	if (a - (double)b > 0.3)
		return (double)b + 0.5;

	return (double)b;
}

//计算相似度
__global__ void getSim(const int testData[], const double rating_map[], double sim[], const int USER_NUM, const int MOVIE_NUM, const int offeset)
{
	int bid = blockIdx.x;   
	int tid = threadIdx.x;  
	
	int movie1Id = bid + 1;
	int movie2Id = ((tid > MOVIE_NUM >> 2) ? MOVIE_NUM - tid : tid) + 1;

	double movie1Sum = 0;
	double movie2Sum = 0;

	int kernelNum = 0;

	double movie1[PEOPLE_CAP];
	double movie2[PEOPLE_CAP];
	for (int i = 1; i <= USER_NUM; i++)
		if (rating_map[movie1Id * PEOPLE_CAP + i] && rating_map[movie2Id * PEOPLE_CAP + i])
		{
			movie1Sum += rating_map[movie1Id * PEOPLE_CAP + i];
			movie2Sum += rating_map[movie2Id * PEOPLE_CAP + i];
			movie1[kernelNum] = rating_map[movie1Id * PEOPLE_CAP + i];
			movie2[kernelNum] = rating_map[movie2Id * PEOPLE_CAP + i];
			kernelNum++;
		}
	if (kernelNum)
	{
		double bar1 = movie1Sum / kernelNum;
		double bar2 = movie2Sum / kernelNum;

		double temp1 = 0;
		double temp2 = 0;
		double temp3 = 0;

		for (int i = 0; i < kernelNum; i++)
		{
			temp1 += (movie1[i] - bar1)*(movie2[i] - bar2);
			temp2 += (movie1[i] - bar1)*(movie1[i] - bar1);
			temp3 += (movie2[i] - bar2)*(movie2[i] - bar2);
		}
		if (!temp2 || !temp1)
		{
			//sim[movie1Id * MOVIE_NUM + movie2Id] = 0;
			//sim[movie2Id * MOVIE_NUM + movie1Id] = 0;
		}
		else
		{
			double result = temp1 / sqrt(temp2 * temp3);
			//sim[movie1Id * MOVIE_NUM + movie2Id] = result > 0 ? result : 0;
			//sim[movie2Id * MOVIE_NUM + movie1Id] = result > 0 ? result : 0;
			sim[movie1Id * MOVIE_NUM + movie2Id] = result;
			sim[movie2Id * MOVIE_NUM + movie1Id] = result;
			//行是被预测的movie															 
			//sim[testMovieId * MOVIE_NUM + tid] = 10;
		}

	}
	//else
	//{
	//	//sim[movie1Id * MOVIE_NUM + movie2Id] = 0;
	//	//sim[movie2Id * MOVIE_NUM + movie1Id] = 0;
	//}


}
__global__ void conclude(double result[], const double rating_map[], double sim[], const int TEST_NUM, const int MOVIE_NUM, const int testData[])
{
	int tid = threadIdx.x;
	int bid = blockIdx.x;
	int index = bid * blockDim.x + tid;

	if (index < TEST_NUM)
	{
		int testMovieId = testData[index] & 0xFFFF;   //需要测量的电影id
		int userId = testData[index] >> 16;

		double a = 0;
		double b = 0;
		double t1, t2, t3, t4;
		t1 = t2 = t3 = t4 = 0;
		int length = 0;
		double rate;
		double similarity;
		for (int i = 0; i < MOVIE_NUM; i++)
			if ( (rate = rating_map[i * PEOPLE_CAP + userId]) && (similarity = sim[testMovieId * MOVIE_NUM + i]))
			{
				t1 += similarity * similarity;
				t2 += similarity;
				t3 += similarity * rate;
				t4 += rate;
				length++;
			}

		a = (t3*length - t2*t4) / (t1*length - t2*t2);
		b = (t1*t4 - t2*t3) / (t1*length - t2*t2);

		result[index] = (a + b) > 0 ? a + b : 1;

		//double sum1 = 0;
		//double sum2 = 0;
		//double rate;
		//double similarity;
		//int length = 0;

		//double MaxSim = -1;
		//int MaxIndex = 0;
		//for (int i = 0; i < MOVIE_NUM; i++)
		//	if ((rate = rating_map[i * PEOPLE_CAP + userId]) && (similarity = sim[testMovieId * MOVIE_NUM + i]) && similarity > 0.8)
		//	{
		//		if ( similarity > MaxSim && i != testMovieId)
		//		{
		//			MaxSim = similarity;
		//			MaxIndex = i;
		//		}
		//		sum1 += rate * similarity;
		//		sum2 += similarity;
		//		length++;
		//	}
		//for (int i = 0; i < MOVIE_NUM; i++)
		//	if ((rate = rating_map[i * PEOPLE_CAP + userId]) && (similarity = sim[testMovieId * MOVIE_NUM + i]) && similarity > 0.7)
		//	{
		//		length++;
		//		if (length == 30)
		//			break;
		//		if (similarity > MaxSim && i != testMovieId)
		//		{
		//			MaxSim = similarity;
		//			MaxIndex = i;
		//		}
		//		sum1 += rate * similarity;
		//		sum2 += similarity;
		//	}
		//for (int i = 0; i < MOVIE_NUM; i++)
		//	if ((rate = rating_map[i * PEOPLE_CAP + userId]) && (similarity = sim[testMovieId * MOVIE_NUM + i]) && similarity > 0.6)
		//	{
		//		length++;
		//		if (length == 30)
		//			break;
		//		if (similarity > MaxSim && i != testMovieId)
		//		{
		//			MaxSim = similarity;
		//			MaxIndex = i;
		//		}
		//		sum1 += rate * similarity;
		//		sum2 += similarity;
		//	}

		//if (length < 3)
		//	result[index] = rating_map[MaxIndex * PEOPLE_CAP + userId];
		//else
		//	result[index] = sum1/sum2;
	}

}

double *dev_rating_map = 0;
double *temp_dev_rating_map = 0;
double *dev_sim = 0;
int *dev_test_data = 0;
double *dev_result = 0;


int main()
{
	//测试结果
	int testData[TEST_DATA_CAP];
	double testRating[TEST_DATA_CAP];

	int startClock = clock();

	cudaSetDevice(0);

	//数据文件
	int ReadingClock = clock();
	string fileName = "data/u1.base";
	ifstream ratingFile(fileName);
	if (!ratingFile.is_open())
	{
		std::cout << "Error opening " + fileName;
		exit(1);
	}

	//测试文件
	fileName = "data/u1.test";
	ifstream testFile(fileName);
	if (!testFile.is_open())
	{
		std::cout << "Error opening " + fileName;
		exit(1);
	}

	//结果文件
	fileName = "data/result.txt";
	ofstream resultFile;
	resultFile.open(fileName);
	if (!resultFile.is_open())
	{
		std::cout << "Error opening " + fileName;
		exit(1);
	}

	//读入测试文件
	double rating;
	int userId, movieId;
	long long int timeStamp;

	int TEST_NUM = 0;
	while (!testFile.eof())
	{
		testFile >> userId >> movieId >> rating >> timeStamp;
		//int temp = userId << 16 | movieId;
		//cout <<  (temp >> 16) << " " << (temp & 0xFFFF) << endl;
		testRating[TEST_NUM] = rating;
		testData[TEST_NUM++] = userId << 16 | movieId;   //假设id值都不大于65 536
	}
	cudaMalloc((void**)&dev_test_data, TEST_NUM * sizeof(int));
	cudaMemcpy(dev_test_data, testData, TEST_NUM * sizeof(int), cudaMemcpyHostToDevice);
	testFile.close();

	//读入评分数据
	cudaMalloc((void**)&dev_rating_map, PEOPLE_CAP * MOVIE_CAP * sizeof(double));   //行是电影,列是人

	int MOVIE_NUM = 0;
	int PEOPLE_NUM = 0;
	while (!ratingFile.eof())
	{
		ratingFile >> userId >> movieId >> rating >> timeStamp;
		if (PEOPLE_NUM < userId)
			PEOPLE_NUM = userId;
		if (movieId > MOVIE_NUM)
			MOVIE_NUM = movieId;
		cudaMemcpy(dev_rating_map + userId + movieId * PEOPLE_CAP, &rating, sizeof(double), cudaMemcpyHostToDevice);
	}
	ratingFile.close();

	//double temp[20000];
	//for (int i = 0; i <  MOVIE_CAP; i++)
	//{
	//	cout << i << endl;
	//	cudaMemcpy(temp, dev_rating_map + PEOPLE_CAP * i, PEOPLE_CAP * sizeof(double), cudaMemcpyDeviceToHost);
	//	for (int j = 0; j < PEOPLE_CAP; j++)
	//		if (temp[j] != 0)
	//			cout << temp[j] << " ";
	//	cout << endl;
	//}

	cout << "There are " << MOVIE_NUM << " movies amd " << PEOPLE_NUM << " peoples" << endl;
	cout << TEST_NUM << " data need to be predicted" << endl;
	std::cout << "ReadFile use " << clock() - ReadingClock << "ms" << endl;


	cudaMalloc((void**)&dev_sim, MOVIE_NUM * MOVIE_NUM * sizeof(double));   //sim值
	int threadNum = MOVIE_NUM > 1024 ? 1024 : MOVIE_NUM;

	getSim <<< MOVIE_NUM, MOVIE_NUM /2>>> (dev_test_data, dev_rating_map, dev_sim, PEOPLE_NUM, MOVIE_NUM, 0);
	cudaThreadSynchronize();

	double temp[20000];
	for (int i = 1; i <= MOVIE_NUM; i++)
	{
		cudaMemcpy(temp, dev_sim + MOVIE_NUM * i, MOVIE_NUM * sizeof(double), cudaMemcpyDeviceToHost);
		for (int j = 0; j < MOVIE_NUM; j++)
			//if (temp[j] != 0)
			cout << i << " " << j << " " << temp[j] << endl;
	}

	cudaMalloc((void**)&dev_result, TEST_NUM * sizeof(double));   //sim值
	conclude << < 20, 1000 >> > (dev_result, dev_rating_map, dev_sim, TEST_NUM, MOVIE_NUM, dev_test_data);

	double *result = new double[TEST_NUM];
	cudaMemcpy(result, dev_result, TEST_NUM * sizeof(double), cudaMemcpyDeviceToHost);


	double d = 0;
	int realNum = 0;
	for (int i = 0; i < TEST_NUM; i++)
	{
		double predict = approximate(result[i]);
		//cout << (testData[i] >> 16) << " "  << (testData[i] & 0xFFFF) << " " << testRating[i] << " " << predict << endl;
		if ( predict != 1)
		{
			//cout << (testData[i] >> 16) << " " << (testData[i] & 0xFFFF) << " " << testRating[i] << " " << predict << endl;
			double c = fabs(testRating[i] - predict);
			d += c;
			realNum++;
		}

	}
		
	cout << "MAE为" << d / realNum << endl;

	cudaFree(dev_rating_map);
	cudaFree(dev_test_data);
	cudaFree(dev_result);
	cudaFree(dev_sim);
	std::cout << "total use " << clock() - startClock << "ms" << endl;
	return 0;
}

