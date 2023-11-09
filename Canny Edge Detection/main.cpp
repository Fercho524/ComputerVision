#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <math.h>

#define VERBOSE true

using namespace cv;
using namespace std;

double normalEstandar(double x, double y, double sigma)
{
  return (1 / (2 * M_PI * sigma * sigma)) * pow(M_E, (-(x * x + y * y) / (2 * sigma * sigma)));
}

double **gaussKernel(int n, double sigma)
{
  if(VERBOSE){
    cout << "Generating Gaussian kernel with size " << n << " and sigma " << sigma << endl;
  }

  double **matrix = (double **)malloc(sizeof(double) * n);

  for (int i = 0, x = -((n - 1) / 2); i < n && x < ((n - 1) / 2) + 1; i++, x++)
  {
    matrix[i] = (double *)malloc(sizeof(double) * n);
    for (int j = 0, y = -((n - 1) / 2); j < n && y < ((n - 1) / 2) + 1; j++, y++)
    {
      matrix[i][j] = normalEstandar(x, y, sigma);
    }
  }

  return matrix;
}

double **zeros(int m, int n)
{
  double **matrix = (double **)malloc(sizeof(double) * m);

  for (int i = 0; i < m; i++)
  {
    matrix[i] = (double *)malloc(sizeof(double) * n);
    for (int j = 0; j < n; j++)
    {
      matrix[i][j] = 0;
    }
  }
  return matrix;
}

double **Vecindad(Mat image, int large, int x, int y)
{
  double **vec = zeros(large, large);

  int m = image.rows;
  int n = image.cols;

  for (int i = x - ((large - 1) / 2), row = 0; row < large && i < x + ((large - 1) / 2) + 1; row++, i++)
  {
    for (int j = y - ((large - 1) / 2), col = 0; col < large && j < y + ((large - 1) / 2) + 1; col++, j++)
    {
      bool condition = (i >= 0) && (j >= 0) && (i < m) && (j < n);

      if (condition)
      {
        // cout << " \n(" << i << " , " << j << ")";
        vec[row][col] = (double)image.at<uchar>(Point(j, i));
      }
      else
      {
        vec[row][col] = 0;
      }
    }
  }
  return vec;
}

void printMatrix(double **matrix, int m, int n)
{
  cout << "\n";
  for (int i = 0; i < m; i++)
  {
    cout << "[";

    for (int j = 0; j < n; j++)
    {
      cout << " " << matrix[i][j] << " ";
    }
    cout << "]" << endl;
  }
}

double Convolucion(double **a, double **b, int m, int n)
{
  double result = 0;
  double prom = 1;

  for (int i = 0; i < m; i++)
  {
    for (int j = 0; j < n; j++)
    {
      result += a[i][j] * b[i][j];
      if (result <= 0)
        prom += 128;
    }
  }

  if (result < 0)
    result *= -1;

  return result;
}

void Filtro2D(Mat image, Mat result, double **kern, int n)
{
  int m = image.rows;
  int c = image.cols;
  double gray_m;

  for (int i = 0; i < m; i++)
  {
    for (int j = 0; j < c; j++)
    {
      double **vec0 = Vecindad(image, n, i, j);
      result.at<uchar>(Point(j, i)) = (uchar)Convolucion(vec0, kern, n, n);
    }
  }
}

double **VerticalSobel()
{
  double **kern = zeros(3, 3);
  kern[0][0] = -1;
  kern[0][1] = 0;
  kern[0][2] = 1;
  kern[1][0] = -2;
  kern[1][1] = 0;
  kern[1][2] = 2;
  kern[2][0] = -1;
  kern[2][1] = 0;
  kern[2][2] = 1;
  return kern;
}

double **HorizontalSobel()
{
  if(VERBOSE){
    cout << "Horizontal Sobel Gx Generated" << endl;
  }
  double **kern = zeros(3, 3);
  kern[0][0] = -1;
  kern[0][1] = -2;
  kern[0][2] = -1;
  kern[1][0] = 0;
  kern[1][1] = 0;
  kern[1][2] = 0;
  kern[2][0] = 1;
  kern[2][1] = 2;
  kern[2][2] = 1;
  return kern;
}

void NormaImagenes(Mat a, Mat b, Mat result)
{
  if(VERBOSE){
    cout << "Getting | G | = sqrt(Gx^2 + Gy^2)" << endl;
  }
  int m = result.rows;
  int n = result.cols;

  for (int i = 0; i < m; i++)
  {
    for (int j = 0; j < n; j++)
    {
      int b1 = (int)a.at<uchar>(Point(j, i));
      int b2 = (int)b.at<uchar>(Point(j, i));

      result.at<uchar>(Point(j, i)) = (double)sqrt(b1 * b1 + b2 * b2);
    }
  }
}

void DirectionEdges(Mat x, Mat y, double **result)
{
  if(VERBOSE){
    cout << "Getting Edges Direction with tan^-1(Gy/Gx)" << endl;
  }
  int m = x.rows;
  int n = x.cols;

  for (int i = 0; i < m; i++)
  {
    for (int j = 0; j < n; j++)
    {
      float X = (float)x.at<uchar>(Point(i, j));
      float Y = (float)y.at<uchar>(Point(i, j));
      float angle = (atan(Y / X) * 180) / M_PI;

      if (angle == 0 || angle == 180)
      {
        result[i][j] = 0;
      }
      else if (angle == 90 || angle == 270)
      {
        result[i][j] = 90;
      }
      else if (angle > 0 && angle < 45 || angle > 180 && angle < 225)
      {
        result[i][j] = 45;
      }
      else if (angle > 135 && angle < 180 || angle > 270 && angle < 315)
      {
        result[i][j] = 135;
      }
    }
  }
}

void printImage(Mat image, int m, int n)
{
  for (int i = 0; i < m; i++)
  {
    cout << "[";
    for (int j = 0; j < n; j++)
    {
      double pixel = (double)image.at<uchar>(Point(i, i));
      cout << " " << pixel << " ";
    }
    cout << "]\n";
  }
}

void printImageInfo(Mat a, string id)
{
  cout << "" << id << endl;
  cout << "( " << a.rows << " x " << a.cols << " )" << endl;
}

void CannyEdgeDetection(Mat G, double **Direction, Mat result)
{
  if (VERBOSE){
    cout << "Making Canny Edge Detection\n"; 
  }
  int m = G.rows;
  int n = G.cols;

  for (int i = 0; i < m; i++)
  {
    for (int j = 0; j < n; j++)
    {
      double **vec = Vecindad(G, 3, i, j);
      double angle = Direction[i][j];
      double lineG[3];

      // Obteniendo la Línea del Borde
      if (angle == 0)
      {
        lineG[0] = vec[1][0];
        lineG[1] = vec[1][1];
        lineG[2] = vec[1][2];
      }
      else if (angle == 90)
      {
        lineG[0] = vec[0][1];
        lineG[1] = vec[1][1];
        lineG[2] = vec[2][1];
      }
      else if (angle == 45)
      {
        lineG[0] = vec[0][0];
        lineG[1] = vec[1][1];
        lineG[2] = vec[2][2];
      }
      else if (angle == 135)
      {
        lineG[0] = vec[2][0];
        lineG[1] = vec[1][1];
        lineG[2] = vec[0][2];
      }

      // Calculando si es máximo
      double maximo = 0;

      for (int k = 0; k < 3; k++)
      {
        if (lineG[k] > maximo)
        {
          maximo = lineG[k];
        }
      }

      if (maximo == vec[1][1])
      {
        result.at<uchar>(Point(j, i)) = vec[1][1];
      }
      else
      {
        result.at<uchar>(Point(j, i)) = 0;
      }
    }
  }
}

void Umbralar(Mat image, Mat result)
{
  int m = image.rows;
  int n = image.cols;

  float max=0;
  float min=255;

  // Hayando máximos y Mínimos
  for (int i = 0; i < m; i++)
  {
    for (int j = 0; j < n; j++)
    {
      double pixel = (double)image.at<uchar>(Point(i, j));
      if (pixel > max){
        max = pixel;
      }
      if (pixel<min){
        min = pixel;
      }
    }
  }

  if (VERBOSE){
    cout << "Thresolding Image with max of : " << max <<"\n" ;
  }

  // Umbralando
  for (int i = 0; i < m; i++)
  {
    for (int j = 0; j < n; j++)
    {
      double pixel = (double)image.at<uchar>(Point(j, i));
      if (pixel > 0.9*max)
      {
        result.at<uchar>(Point(j, i)) = 255;
      }else if (pixel < 0.3*max){
        result.at<uchar>(Point(j, i)) = 0;
      }else{
        result.at<uchar>(Point(j, i)) = pixel;
      }
    }
  }
}

int main(int argc, char **argv)
{
  // IMAGEN
  Mat image = imread("image.jpg");

  if (!image.data)
    exit(1);

  int m = image.rows;
  int n = image.cols;
  printImageInfo(image, "Original");

  // GRAYSCALE
  Mat grayScale(m, n, CV_8UC1);
  cvtColor(image, grayScale, COLOR_BGR2GRAY);
  printImageInfo(grayScale,"Step 1 Converted to Grayscale");

  // GAUSSIAN FILTER
  int N = 5;
  double sigma = 1.5;
  double **smoothKernel = gaussKernel(N, sigma);
  printMatrix(smoothKernel,N,N);
  Mat gaussianFilter(m, n, CV_8UC1);
  Filtro2D(grayScale, gaussianFilter, smoothKernel, N);
  printImageInfo(gaussianFilter,"Step 2 Filtered with a Gauss Kernel");
  

  // ECUALIZACIÓN
  Mat equalized(m, n, CV_8UC1);
  equalizeHist(gaussianFilter, equalized);
  printImageInfo(equalized,"Step 3 Equalized Image");

  // FILTRO SOBEL
  Mat vsobel(m, n, CV_8UC1);
  Mat hsobel(m, n, CV_8UC1);
  Mat sobel(m, n, CV_8UC1);
  double **anglesSobel = zeros(m, n);

  Filtro2D(equalized, vsobel, VerticalSobel(), 3);
  Filtro2D(equalized, hsobel, HorizontalSobel(), 3);
  NormaImagenes(vsobel, hsobel, sobel);
  printImageInfo(sobel,"Step 4 With Full Sobel Filter");
  DirectionEdges(vsobel, hsobel, anglesSobel);
  cout << "-> Edges Direction matrix has stored";

  // CANNY
  Mat cannyFilter(m, n, CV_8UC1);
  CannyEdgeDetection(sobel, anglesSobel, cannyFilter);
  printImageInfo(cannyFilter,"Step 5 With Canny Edge Detection Aplied");

  // UMBRALADO
  Mat finalImage(m,n,CV_8UC1);
  Umbralar(cannyFilter,finalImage);
  printImageInfo(finalImage,"Step 6 Final Image with All Borders Detected");

  // RESULTADOS
  imshow("Original", grayScale);
  imshow("Gauss", gaussianFilter);
  imshow("Equalized", equalized);
  imshow("Sobel", sobel);
  imshow("Canny", cannyFilter);
  imshow("UmbralFinal",finalImage);

  waitKey(0);
  destroyAllWindows();

  return 0;
}