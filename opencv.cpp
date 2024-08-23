#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <malloc.h>

#include <opencv2/opencv.hpp>

using namespace cv;

typedef struct {
    int r, g, b;
}int_rgb;

#define imax(x,y)((x > y) ? x : y)
#define imin(x,y)((x < y) ? x : y)
#define clipping(x) imin(imax(x, 0), 255)
#define roundup(x) ((int)(x + 0.5))






int** IntAlloc2(int height, int width)
{
    int** tmp;
    tmp = (int**)calloc(height, sizeof(int*));
    for (int i = 0; i < height; i++)
        tmp[i] = (int*)calloc(width, sizeof(int));
    return(tmp);
}

void IntFree2(int** image, int height, int width)
{
    for (int i = 0; i < height; i++)
        free(image[i]);

    free(image);
}


float** FloatAlloc2(int height, int width)
{
    float** tmp;
    tmp = (float**)calloc(height, sizeof(float*));
    for (int i = 0; i < height; i++)
        tmp[i] = (float*)calloc(width, sizeof(float));
    return(tmp);
}

void FloatFree2(float** image, int height, int width)
{
    for (int i = 0; i < height; i++)
        free(image[i]);

    free(image);
}

int_rgb** IntColorAlloc2(int height, int width)
{
    int_rgb** tmp;
    tmp = (int_rgb**)calloc(height, sizeof(int_rgb*));
    for (int i = 0; i < height; i++)
        tmp[i] = (int_rgb*)calloc(width, sizeof(int_rgb));
    return(tmp);
}

void IntColorFree2(int_rgb** image, int height, int width)
{
    for (int i = 0; i < height; i++)
        free(image[i]);

    free(image);
}

int** ReadImage(char* name, int* height, int* width)
{
    Mat img = imread(name, IMREAD_GRAYSCALE);
    int** image = (int**)IntAlloc2(img.rows, img.cols);

    *width = img.cols;
    *height = img.rows;

    for (int i = 0; i < img.rows; i++)
        for (int j = 0; j < img.cols; j++)
            image[i][j] = img.at<unsigned char>(i, j);

    return(image);
}

void WriteImage(char* name, int** image, int height, int width)
{
    Mat img(height, width, CV_8UC1);
    for (int i = 0; i < height; i++)
        for (int j = 0; j < width; j++)
            img.at<unsigned char>(i, j) = (unsigned char)image[i][j];

    imwrite(name, img);
}


void ImageShow(char* winname, int** image, int height, int width)
{
    Mat img(height, width, CV_8UC1);
    for (int i = 0; i < height; i++)
        for (int j = 0; j < width; j++)
            img.at<unsigned char>(i, j) = (unsigned char)image[i][j];


    imshow(winname, img);
    waitKey(0);
}



int_rgb** ReadColorImage(char* name, int* height, int* width)
{
    Mat img = imread(name, IMREAD_COLOR);
    int_rgb** image = (int_rgb**)IntColorAlloc2(img.rows, img.cols);

    *width = img.cols;
    *height = img.rows;

    for (int i = 0; i < img.rows; i++)
        for (int j = 0; j < img.cols; j++) {
            image[i][j].b = img.at<Vec3b>(i, j)[0];
            image[i][j].g = img.at<Vec3b>(i, j)[1];
            image[i][j].r = img.at<Vec3b>(i, j)[2];
        }

    return(image);
}

void WriteColorImage(char* name, int_rgb** image, int height, int width)
{
    Mat img(height, width, CV_8UC3);
    for (int i = 0; i < height; i++)
        for (int j = 0; j < width; j++) {
            img.at<Vec3b>(i, j)[0] = (unsigned char)image[i][j].b;
            img.at<Vec3b>(i, j)[1] = (unsigned char)image[i][j].g;
            img.at<Vec3b>(i, j)[2] = (unsigned char)image[i][j].r;
        }

    imwrite(name, img);
}

void ColorImageShow(char* winname, int_rgb** image, int height, int width)
{
    Mat img(height, width, CV_8UC3);
    for (int i = 0; i < height; i++)
        for (int j = 0; j < width; j++) {
            img.at<Vec3b>(i, j)[0] = (unsigned char)image[i][j].b;
            img.at<Vec3b>(i, j)[1] = (unsigned char)image[i][j].g;
            img.at<Vec3b>(i, j)[2] = (unsigned char)image[i][j].r;
        }
    imshow(winname, img);

}

template <typename _TP>
void ConnectedComponentLabeling(_TP** seg, int height, int width, int** label, int* no_label)
{

    //Mat bw = threshval < 128 ? (img < threshval) : (img > threshval);
    Mat bw(height, width, CV_8U);

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++)
            bw.at<unsigned char>(i, j) = (unsigned char)seg[i][j];
    }
    Mat labelImage(bw.size(), CV_32S);
    *no_label = connectedComponents(bw, labelImage, 8); // 0        Ե        

    (*no_label)--;

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++)
            label[i][j] = labelImage.at<int>(i, j);
    }
}

#define imax(x, y) ((x)>(y) ? x : y)
#define imin(x, y) ((x)<(y) ? x : y)

int BilinearInterpolation(int** image, int width, int height, double x, double y)
{
    int x_int = (int)x;
    int y_int = (int)y;

    int A = image[imin(imax(y_int, 0), height - 1)][imin(imax(x_int, 0), width - 1)];
    int B = image[imin(imax(y_int, 0), height - 1)][imin(imax(x_int + 1, 0), width - 1)];
    int C = image[imin(imax(y_int + 1, 0), height - 1)][imin(imax(x_int, 0), width - 1)];
    int D = image[imin(imax(y_int + 1, 0), height - 1)][imin(imax(x_int + 1, 0), width - 1)];

    double dx = x - x_int;
    double dy = y - y_int;

    double value
        = (1.0 - dx) * (1.0 - dy) * A + dx * (1.0 - dy) * B
        + (1.0 - dx) * dy * C + dx * dy * D;

    return((int)(value + 0.5));
}


void DrawHistogram(char* comments, int* Hist)
{
    int histSize = 256; /// Establish the number of bins
    // Draw the histograms for B, G and R
    int hist_w = 512; int hist_h = 512;
    int bin_w = cvRound((double)hist_w / histSize);

    Mat histImage(hist_h, hist_w, CV_8UC3, Scalar(255, 255, 255));
    Mat r_hist(histSize, 1, CV_32FC1);
    for (int i = 0; i < histSize; i++)
        r_hist.at<float>(i, 0) = Hist[i];
    /// Normalize the result to [ 0, histImage.rows ]
    normalize(r_hist, r_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());

    /// Draw for each channel
    for (int i = 1; i < histSize; i++)
    {
        line(histImage, Point(bin_w * (i - 1), hist_h - cvRound(r_hist.at<float>(i - 1))),
            Point(bin_w * (i), hist_h - cvRound(r_hist.at<float>(i))),
            Scalar(255, 0, 0), 2, 8, 0);
    }

    /// Display
    namedWindow(comments, WINDOW_AUTOSIZE);
    imshow(comments, histImage);

    waitKey(0);

}

void ex0625()
{
    int height, width;
    int** image = ReadImage((char*)"./TestImages/lena.png", &height, &width);

    printf("%d ", image[5][9]);

    ImageShow((char*)"test", image, height, width);


}

void DrawRect(int** img, int height, int width)
{

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            img[y][x] = 0;
        }
    }

    for (int y = height / 4; y < height * 3 / 4; y++) {
        for (int x = width / 4; x < width * 3 / 4; x++) {
            img[y][x] = 255;
        }
    }
}

void DRect(int**img,int height, int width)
{
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            img[y][x] = 0;
        }
    }

    for (int y = height / 4; y <= height / 4 * 3; y++) {
        for (int x = width / 4; x <= width / 4 * 3; x++) {
            img[y][x] = 255;
        }
    }
}

void DCircle(int**img,int height, int width, int x0, int y0, int R)
{
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int dist2 = (x - x0) * (x - x0) + (y - y0) * (y - y0);
            if (dist2 < (R * R))
                img[y][x] = 255;
            else
                img[y][x] = 0;
        }
    }
}


void test1_2()
{
    int height = 500, width = 500;
    int** img = (int**)IntAlloc2(height, width);

    int x0 = 250, y0 = 250;
    int R = 100;
    
    DCircle(img, height, width, x0, y0, R);

    ImageShow((char*)"output", img, height, width);
    IntFree2(img, height, width);

}




void DrawCircle(int** img, int height, int width, int x0, int y0, int R)
{
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int dist2 = (x - x0) * (x - x0) + (y - y0) * (y - y0);
            if (dist2 < R * R) {
                img[y][x] = 255;
            }
            else {
                //img[y][x] = 0;
            }
        }
    }
}

void Thres(int thres,int **img, int**img_out, int height,int width)
{
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            if (img[y][x] >= thres)
                img_out[y][x] = 255;
            else
                img_out[y][x] = 0;
        }
    }

}

void Thres2(int thres, int y_st,int y_end, int x_st, int x_end,
    int**img, int**img_out)
{
    for (int y = y_st; y < y_end; y++) {
        for (int x = x_st; x < x_end; x++) {
            if (img[y][x] >= thres)
                img_out[y][x] = 255;
            else
                img_out[y][x] = 0;
        }
    }
}

void Thress2(int thress,int y_st, int y_end, int x_st, int x_end, 
    int**img, int**img_out) ////////// 끝나는 점에 =를 넣으면 초과로 에러가 난다
{
    for (int y = y_st; y <= y_end ; y++) {
        for (int x = x_st; x <=x_end; x++) {
            if (img[y][x] >= thress)
                img_out[y][x] = 255;
            else
                img_out[y][x] = 0;
        }
    }
}


void test1_3()
{

    int height,  width;
    int** img = (int**)ReadImage((char*)"./TestImages/lena.png", &height, &width);
    int** img_out = (int**)IntAlloc2(height, width);

    int thress;

    Thress2(50, 0, height/2, 0, width/2, img, img_out);
    Thress2(100, 0, height / 2, width/2, width, img, img_out);
    Thress2(150, height/2, height, 0, width / 2, img, img_out);
    Thress2(200, height/2, height, width/2, width , img, img_out);

    ImageShow((char*)"input", img, height, width);
    ImageShow((char*)"output", img_out, height, width);

    IntFree2(img, height, width);
    IntFree2(img_out, height, width);

}




void test1()
{
    int height, width;
    int** img = (int**)ReadImage((char*)"./TestImages/barbara.png", &height, &width);
    int** img_out = (int**)IntAlloc2(height, width);

    int thres;
    Thres2(50, 0, height/2, 0, width/2, img, img_out);
    Thres2(100, 0, height / 2, width/2, width, img, img_out);
    Thres2(150, height/2, height , 0, width / 2, img, img_out);
    Thres2(200, height/2, height, width/2, width , img, img_out);

    ImageShow((char*)"input", img, height, width);
    ImageShow((char*)"output", img_out, height, width);

    IntFree2(img, height, width);
    IntFree2(img_out, height, width);
}




int ex0625_1()
{
    int height = 1000;
    int width = 1000;

    int** img = (int**)IntAlloc2(height, width);

    //DrawRect( img, height, width);

    int y0 = 500, x0 = 500;
    int R = 300;

    DrawCircle(img, height, width, x0, y0, R);

    ImageShow((char*)"test", img, height, width);

    IntFree2(img, height, width);

    return 0;


}

int ex0625_2()
{
    int height, width;
    int** img = ReadImage((char*)"barbara.png", &height, &width);

    DrawCircle(img, height, width, 256, 256, 100);

    WriteImage((char*)"test_out.jpg", img, height, width);

    ImageShow((char*)"test", img, height, width);

    IntFree2(img, height, width);

    return 0;
}

void Binarization(int threshold, int** img, int height, int width, int** img_out) {
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            if (img[y][x] > threshold) {
                img_out[y][x] = 255;
            }
            else {
                img_out[y][x] = 0;
            }
        }
    }
}
int main_1() {

    int height, width;
    int** img = ReadImage((char*)"barbara.png", &height, &width);

    int** img_out = (int**)IntAlloc2(height, width);
    int x = 0, y = 0;

    for (int th = 50; th < 250; th += 50) {
        Binarization(th, img, height, width, img_out);
        ImageShow((char*)"test", img_out, height, width);
    }
    IntFree2(img, height, width);

    return 0;
}


void Binarization2(int threshold, int** img, int start_y, int end_y, int start_x, int end_x, int** img_out) {
    for (int y = start_y; y < end_y; y++) {
        for (int x = start_x; x < end_x; x++) {
            if (img[y][x] > threshold) {
                img_out[y][x] = 255;
            }
            else {
                img_out[y][x] = 0;
            }
        }
    }
}

int ex_0625_4() {
    int height, width;
    int** img = ReadImage((char*)"barbara.png", &height, &width);

    int** img_out = (int**)IntAlloc2(height, width);


    int mid_y = height / 2;
    int mid_x = width / 2;


    Binarization2(50, img, 0, mid_y, 0, mid_x, img_out);
    Binarization2(100, img, 0, mid_y, mid_x, width, img_out);
    Binarization2(150, img, mid_y, height, 0, mid_x, img_out);
    Binarization2(200, img, mid_y, height, mid_x, width, img_out);

    ImageShow((char*)"test", img_out, height, width);
    IntFree2(img, height, width);
    IntFree2(img_out, height, width);

    return 0;
}


void Clipping2(int**img_out, int height, int width)
{
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            img_out[y][x] = imax(imin(img_out[y][x], 255), 0);
        }
    }
}


void AddValue2(int addvalue, int** img_in, int height, int width, int** img_out)
{
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            img_out[y][x] = img_in[y][x] + addvalue;
        }
    }
    Clipping2(img_out, height, width);

}

void Mixing(float alpha,int**img1, int **img2, int height, int width, int **img_out)
{/////////////////// alpha를 float로 해야한다. int말고
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            img_out[y][x] = alpha * img1[y][x] + (1 - alpha) * img2[y][x];
        }
    }
}

void test2()
{
    int height, width;
    int** img1 = (int**)ReadImage((char*)"./TestImages/lena.png", &height, &width);
    int** img2 = (int**)ReadImage((char*)"./TestImages/barbara.png", &height, &width);
    int** img_out = (int**)IntAlloc2(height, width);

    ImageShow((char*)"intput1", img1, height, width);
    ImageShow((char*)"intput2", img2, height, width);

    for (float alpha = 0.1; alpha <= 0.8; alpha += 0.1) {
        Mixing(alpha, img1, img2, height, width, img_out);
        ImageShow((char*)"output1", img_out, height, width);
    }
    

    
   

    IntFree2(img1, height, width);
    IntFree2(img2, height, width);
    IntFree2(img_out, height, width);
  

}





void Clipping(int** img, int height, int width)
{
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {

            //img[y][x] = imin(img[y][x], 255);
            //img[y][x] = imax(img[y][x], 0);


            img[y][x] = imax(imin(img[y][x], 255), 0);


            //if (img[y][x] > 255)  img[y][x] = 255;
            //else if (img[y][x] < 0)  img[y][x] = 0;
            //else img[y][x]=img[y][x];
        }
    }
}

void AddValue(int additiveValue, int** img_in, int height, int width, int** img_out)
{
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            img_out[y][x] = img_in[y][x] + additiveValue;
        }
    }
    Clipping(img_out, height, width);

}


void ex0626_01()
{
    int height, width;
    int** img = ReadImage((char*)"./TestImages/barbara.png", &height, &width);
    int** img_out1 = (int**)IntAlloc2(height, width);
    int** img_out2 = (int**)IntAlloc2(height, width);

    int additiveValue = 50;
    AddValue(additiveValue, img, height, width, img_out1);


    additiveValue = -50;
    AddValue(additiveValue, img, height, width, img_out2);

    //함수화해보라 Clipping()
    //Clipping(img_out1, height, width);
    //Clipping(img_out2, height, width);

    ImageShow((char*)"intput", img, height, width);
    ImageShow((char*)"output2", img_out1, height, width);
    ImageShow((char*)"output3", img_out2, height, width);

    IntFree2(img, height, width);
    IntFree2(img_out1, height, width);
    IntFree2(img_out2, height, width);



}



int FindMaxvalue(int** img, int height, int width)

{
    int maxvalue = -1;
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            maxvalue = imax(img[y][x], maxvalue);


        }
    }
    return maxvalue;
}

int FindMinvalue(int** img, int height, int width)
{
    int minvalue = 300;
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            minvalue = imin(img[y][x], minvalue);


        }
    }
    return minvalue;
}

int ex0626_2()
{
    {
        int height, width;
        int** img = ReadImage((char*)"./TestImages/barbara.png", &height, &width);

        int maxvalue = FindMaxvalue(img, height, width);
        int minvalue = FindMinvalue(img, height, width);
        printf("\ maxvalue = %d, minvalue = %d", maxvalue, minvalue);

        IntFree2(img, height, width);

        return 0;
    }


}

void ImageMixing(double alpha, int** img_1, int** img_2, int height, int width, int** img_out) {
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            img_out[y][x] = roundup(alpha * img_1[y][x] + (1.0 - alpha) * img_2[y][x] + 0.5);
        }/// alpha가 소수이므로 정수값이 안나와서 roundup을 하는거다.
    }
}

int ex0626_3() {
    int height, width;
    int** img_1 = ReadImage((char*)"./TestImages/barbara.png", &height, &width);
    int** img_2 = ReadImage((char*)"./TestImages/lena.png", &height, &width);
    int** img_out = (int**)IntAlloc2(height, width);

    double alpha = 0.5;
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            img_out[y][x] = roundup(alpha * img_1[y][x] + (1.0 - alpha) * img_2[y][x] + 0.5);
        }
    }

    ImageShow((char*)"input 1", img_1, height, width);
    ImageShow((char*)"input 2", img_2, height, width);
    ImageShow((char*)"output", img_out, height, width);

    IntFree2(img_1, height, width);
    IntFree2(img_2, height, width);
    IntFree2(img_out, height, width);

    destroyAllWindows(); //모든 창을 닫는다
    return 0;
}
int ex0626_4() {
    int height, width;
    int** img_1 = ReadImage((char*)"./TestImages/barbara.png", &height, &width);
    int** img_2 = ReadImage((char*)"./TestImages/lena.png", &height, &width);
    int** img_out = (int**)IntAlloc2(height, width);

    double alpha = 0.5;

    ImageMixing(alpha, img_1, img_2, height, width, img_out);

    ImageShow((char*)"input 1", img_1, height, width);
    ImageShow((char*)"input 2", img_2, height, width);
    ImageShow((char*)"output", img_out, height, width);

    IntFree2(img_1, height, width);
    IntFree2(img_2, height, width);
    IntFree2(img_out, height, width);

    destroyAllWindows(); //모든 창을 닫는다
    return 0;
}
int ex0626_5() {
    int height, width;
    int** img_1 = ReadImage((char*)"./TestImages/barbara.png", &height, &width);
    int** img_2 = ReadImage((char*)"./TestImages/lena.png", &height, &width);
    int** img_out = (int**)IntAlloc2(height, width);

    double alpha = 0.5;

    ImageShow((char*)"input 1", img_1, height, width);
    ImageShow((char*)"input 2", img_2, height, width);

    for (alpha = 0.1; alpha < 0.9; alpha += 0.1) {
        ImageMixing(alpha, img_1, img_2, height, width, img_out);
        ImageShow((char*)"output", img_out, height, width);
    }
    IntFree2(img_1, height, width);
    IntFree2(img_2, height, width);
    IntFree2(img_out, height, width);

    destroyAllWindows(); //모든 창을 닫는다
    return 0;
}
int ex0625_6() {

    ex0626_3();
    ex0626_4();
    ex0626_5();
    return 0;
}
int main_roundup() {
    float a = 10.499;
    int c = roundup(a);

    printf("\n c = %d \n", c); //반올림

    return 0;
}

int main_test()
{
    int xxx = 300, yyy = 200;
    int maxvalue, minvalue;


    //maxvalue = (x > y) ? x : y;
    //minvalue = (x < y) ? x : y;
    maxvalue = imax(xxx, yyy);
    minvalue = imin(xxx, yyy);

    return 0;
}

void Stretch(int a, int b, int c, int d, int**img, int **img_out, int height,int width)
{
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            if (img[y][x] < a)
                img_out[y][x] = (float)c / a * img[y][x];
            else if (img[y][x] < b)
                img_out[y][x] = ((float)d - c) / (b - a) * (img[y][x] - a) + c;
            else
                img_out[y][x] = (255.0 - d) / (255.0 - b) * (img[y][x] - b) + d;
        }
    }
}

void Stretch2(int a, int b, int c, int d, int** img, int** img_out, int height, int width)
{  // 반올림 한게 더 좋다  roundup
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            if (img[y][x] < a)
                img_out[y][x] = roundup((float)c / a * img[y][x]);
            else if (img[y][x] < b)
                img_out[y][x] = roundup(((float)d - c) / (b - a) * (img[y][x] - a) + c);
            else
                img_out[y][x] = roundup((255.0 - d) / (255.0 - b) * (img[y][x] - b) + d);
        }
    }
}


void test3_1()
{
    int height, width;
    int** img = (int**)ReadImage((char*)"./TestImages/barbara.png", &height, &width);
    int** img_out1 = (int**)IntAlloc2(height, width);
    int** img_out2 = (int**)IntAlloc2(height, width);

    int a = 100, b = 150, c = 50, d = 200;

    Stretch(a, b, c, d, img, img_out1, height, width);
    Stretch2(a, b, c, d, img, img_out2, height, width);
    
    ImageShow((char*)"input", img, height, width);
    ImageShow((char*)"output1", img_out1, height, width);
    ImageShow((char*)"output2", img_out2, height, width);

    IntFree2(img, height, width);
    IntFree2(img_out1, height, width);
    IntFree2(img_out2, height, width);

}

void Histo(int**img, int height, int width, int*Hist)
{
    for (int i = 0; i < 256; i++) {
        Hist[i] = 0;

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                Hist[img[y][x]]++;
            }
    }
    }
}

void c_Histto(int**img, int height, int width, int*c_Hist)
{
    int Hist[256] = { 0 };
    Histo(img, height, width, Hist);

    for (int i = 0; i < 256; i++) {
        c_Hist[i] = 0;
    }
    c_Hist[0] = Hist[0];
    for (int i = 0; i < 256; i++) {
        c_Hist[i] = Hist[i] + c_Hist[i - 1];
    }
    
}

void c_norm(int **img, int height, int width, int* NC_Hist)
{
    int c_Hist[256] = { 0 };
    c_Histto(img, height, width, c_Hist);

    int N = height * width;
    for (int i = 0; i < 256; i++) {
        NC_Hist[i] = c_Hist[i] * 255.0 / N;
    }
}

void test3_2()
{
    int height, width;
    int** img = (int**)ReadImage((char*)"./TestImages/lenax0.5.png", &height, &width);
    int** img_out = (int**)IntAlloc2(height, width);
    int Hist[256] = { 0 };

    Histo(img, height, width, Hist);

    int c_Hist[256] = { 0 };

    c_Histto(img, height, width, c_Hist);

    int NC_Hist[256] = { 0 };
    c_norm(img, height, width, NC_Hist);

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            img_out[y][x] = NC_Hist[img[y][x]];
        }
    }


    ImageShow((char*)"input", img, height, width);
    DrawHistogram((char*)"hist", Hist);
    DrawHistogram((char*)"누적", c_Hist);
    DrawHistogram((char*)"정규", NC_Hist);
    ImageShow((char*)"output", img_out, height, width);


    IntFree2(img, height, width);
    IntFree2(img_out, height, width);

}



void ImageStretch(int** img, int** img_out, int a, int b, int c, int d, int height, int width) {
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int I = img[y][x];
            if (I <= a) {
                img_out[y][x] = (c / a) * I;
            }
            else if (a < I && I <= b) { // else if (I<=b)
                img_out[y][x] = (d - c) / (b - a) * (I - a) + c;
            }
            else {
                img_out[y][x] = (255 - d) / (255 - b) * (I - b) + d;
            }
        }
    }
}
void ImageStretch2(int** img, int** img_out, int a, int b, int c, int d, int height, int width) {
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int I = img[y][x];
            if (I <= a) {
                img_out[y][x] = ((float)c / a) * I;
            }
            else if (a < I && I <= b) { // else if (I<=b)
                img_out[y][x] = ((float)d - c) / (b - a) * (I - a) + c;
            }
            else {
                img_out[y][x] = (255 - (float)d) / (255 - b) * (I - b) + d;
            }
        }
    }
}




int LineFn(int x, int x1, int y1, int x2, int y2) {
    int y;
    y = roundup((double)y2 - y1) / (x2 - x1) * (x - x1) + y1;
    return y;
}
void ImageStretch3(int** img, int** img_out, int a, int b, int c, int d, int height, int width) {
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int I = img[y][x];
            if (I <= a) {
                //img_out[y][x] = ((float)c / a) * I;
                img_out[y][x] = LineFn(I, 0, 0, a, c);
            }
            else if (a < I && I <= b) { // else if (I<=b)
                //img_out[y][x] = ((float)d - c) / (b - a) * (I - a) + c;
                img_out[y][x] = LineFn(I, b, d, a, c);
            }
            else {
                //img_out[y][x] = (255 - (float)d) / (255 - b)*(I - b) + d;
                img_out[y][x] = LineFn(I, b, d, 255, 255);
            }
        }
    }
}
struct PARAM {
    int a; // x1
    int b; //x2
    int c; //y1
    int d; //y2
};




void ImageStretch4(struct PARAM p, int** img, int height, int width, int** img_out /*int a, int b, int c, int d*/) {
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int I = img[y][x];
            if (I <= p.a) {
                //img_out[y][x] = ((float)c / a) * I;
                img_out[y][x] = LineFn(I, 0, 0, p.a, p.c);
            }
            else if (p.a < I && I <= p.b) { // else if (I<=b)
                //img_out[y][x] = ((float)d - c) / (b - a) * (I - a) + c;
                img_out[y][x] = LineFn(I, p.b, p.d, p.a, p.c);
            }
            else {
                //img_out[y][x] = (255 - (float)d) / (255 - b)*(I - b) + d;
                img_out[y][x] = LineFn(I, p.b, p.d, 255, 255);
            }
        }
    }
}
int ex0701_1() {
    int height, width;
    int** img = ReadImage((char*)"./TestImages/barbara.png", &height, &width);
    int** img_out1 = (int**)IntAlloc2(height, width);
    int** img_out2 = (int**)IntAlloc2(height, width);

    int a = 100, b = 150, c = 50, d = 200;
    ImageStretch(img, img_out1, a, b, c, d, height, width);
    ImageStretch2(img, img_out2, a, b, c, d, height, width);
    ImageShow((char*)"input", img, height, width);

    ImageShow((char*)"output", img_out1, height, width);
    ImageShow((char*)"output2", img_out2, height, width);
    IntFree2(img, height, width);




    return 0;
}

int ex0701_2() {
    int x1 = 100, y1 = 150, x2 = 50, y2 = 200;
    int x, y;
    x = 120;
    y = LineFn(x, x1, y1, x2, y2);
    return 0;
}
int ex0701_3() {
    int height, width;
    int** img = ReadImage((char*)"./TestImages/barbara.png", &height, &width);
    int** img_out1 = (int**)IntAlloc2(height, width);
    int** img_out2 = (int**)IntAlloc2(height, width);

    int a = 100, b = 150, c = 50, d = 200;
    ImageStretch2(img, img_out1, a, b, c, d, height, width);
    ImageStretch3(img, img_out2, a, b, c, d, height, width);
    ImageShow((char*)"input", img, height, width);

    ImageShow((char*)"output", img_out1, height, width);
    ImageShow((char*)"output2", img_out2, height, width);
    IntFree2(img, height, width);




    return 0;
}
struct IMAGE {
    int** img;
    int height;
    int width;
};

int ex0701_8() {
    int height, width;
    int** img = ReadImage((char*)"./TestImages/barbara.png", &height, &width);
    int** img_out = (int**)IntAlloc2(height, width);


    //int a = 100, b = 150, c = 50, d = 200;
    struct PARAM p;
    p.a = 100;
    p.b = 150;
    p.c = 50;
    p.d = 200;
    //ImageStretch3( img, img_out,p.a,p.b,p.c,p.d, height, width);
    struct IMAGE image;
    image.img = img;
    image.height = height;
    image.width = width;
    ImageStretch4(p, img, height, width, img_out);
    ImageShow((char*)"input", img, height, width);

    ImageShow((char*)"output", img_out, height, width);

    IntFree2(img, height, width);




    return 0;
}
int GetCount(int value, int** img, int height, int width) {
    int count = 0;
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            if (img[y][x] == value) count++;


        }
    }
    return count;
}
//h[], h[256]
void GetHistogram(int* h, int** img, int height, int width) {
    {
        for (int value = 0; value < 256; value++) {
            h[value] = GetCount(value, img, height, width);
        }
    }
}
int GetHistogram2(int h[256], int** img, int height, int width) {
    {
        int count = 0;
        for (int value = 0; value < 256; value++) {
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    if (img[y][x] == value) count++;
                }
            }

        }
        return count;
    }
}
void GetHistogram3(int* h, int** img, int height, int width) {
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            h[img[y][x]]++;
        }
    }

}
int ex0702_1()
{
    int height, width;
    int** img = ReadImage((char*)"./TestImages/lena.png", &height, &width);

    int h[256]; //h[0]  = ? 
    GetHistogram3(h, img, height, width);


    DrawHistogram((char*)"histogram", h);

    //ImageShow((char*)"test", img, height, width);

    IntFree2(img, height, width);

    return 0;
}
void C_Histogram(int* ch, int** img, int height, int width) {
    int h[256] = { 0 };
    GetHistogram3(h, img, height, width);
    ch[0] = h[0];
    for (int k = 1; k < 256; k++) {
        ch[k] = h[k] + ch[k - 1];
    }
}
void NC_histogram(int* n_ch, int** img, int height, int width) {
    int ch[256] = { 0 };
    C_Histogram(ch, img, height, width);
    for (int n = 0; n < 256; n++) {
        n_ch[n] = roundup(ch[n] * 255.0 / ch[255]);
    }
}
void HistogramEqualization(int** img, int height, int width, int** img_out) {
    int n_ch[256] = { 0 };

    NC_histogram(n_ch, img, height, width);
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            img_out[y][x] = n_ch[img[y][x]];
        }
    }
}
int ex0702_11()
{
    int height, width;
    int** img = ReadImage((char*)"./TestImages/lenax0.5.png", &height, &width);
    int** img_out = (int**)IntAlloc2(height, width);
    int h_in[256] = { 0 }, h_out[256] = { 0 };
    HistogramEqualization(img, height, width, img_out);
    GetHistogram3(h_in, img, height, width);
    GetHistogram3(h_out, img_out, height, width);

    DrawHistogram((char*)"histogram(input)", h_in);
    DrawHistogram((char*)"histogram(output)", h_out);

    ImageShow((char*)"test", img, height, width);
    ImageShow((char*)"output", img_out, height, width);

    IntFree2(img, height, width);
    IntFree2(img_out, height, width);

    return 0;
}

int getavg(int **img, int y, int x)
{
    int p_p;
    p_p= img[y - 1][x - 1] + img[y - 1][x] + img[y + 1][x]
        + img[y][x - 1] + img[y][x] + img[y][x + 1]
        + img[y + 1][x - 1] + img[y + 1][x] + img[y + 1][x + 1];

    p_p = roundup(p_p / 9.0);
    return p_p;
}


int GetAvgAtyx(int y, int x, int** img) {
    int p_prime = img[y - 1][x - 1] + img[y - 1][x] + img[y + 1][x]
        + img[y][x - 1] + img[y][x] + img[y][x + 1]
        + img[y + 1][x - 1] + img[y + 1][x] + img[y + 1][x + 1];

    p_prime = roundup(p_prime / 9.0);
    return p_prime;
}


void avg3x3(int **img, int height, int width, int **img_out)
{
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            if (x == 0 || x == width - 1 || y == 0 || y == height - 1) {
                img_out[y][x] = img[y][x];
            }
            else
                img_out[y][x] = getavg(img, y, x);
        }
    }

}

void avg3x3_2(int** img, int height, int width, int** img_out)
{
    for (int x = 0; x < width; x++) {
        img_out[0][x] = img[0][x];
        img_out[height - 1][x] = img[height - 1][x];
    }

    for (int y = 0; y < height; y++) {
        img_out[y][0] = img[y][0];
        img_out[y][width - 1] = img[y][width - 1];
    }

    for (int y = 1; y < height - 1; y++) {
        for (int x = 1; x < width - 1; x++) {
            img_out[y][x] = getavg( img,  y,  x);
        }
    }

}

void test4()
{
    int height,  width;
    int** img = (int**)ReadImage((char*)"./TestImages/barbara.png", &height, &width);
    int** img_out = (int**)IntAlloc2(height, width);

    
    avg3x3_2(img, height, width, img_out);
    

    ImageShow((char*)"intput", img, height, width);
    ImageShow((char*)"output", img_out, height, width);

    IntFree2(img, height, width);
    IntFree2(img_out, height, width);

    
}

int ex0703_1() {
    int height, width;
    int** img = ReadImage((char*)"./TestImages/barbara.png", &height, &width);
    int** img_out = (int**)IntAlloc2(height, width);

    int y = 100, x = 200;
    for (int y = 1; y < height - 1; y++) {
        for (int x = 1; x < width - 1; x++) {
            img_out[y][x] = GetAvgAtyx(y, x, img);
        }
    }

    ImageShow((char*)"input", img, height, width);
    ImageShow((char*)"평균필터", img_out, height, width);

    IntFree2(img, height, width);
    IntFree2(img_out, height, width);

    return 0;
}
int GetAvgAtyx2(int y, int x, int** img, int height, int width) {
    int p_prime;
    if (x < 1 || y < 1 || x >= (width - 1) || y >= (height - 1)) {
        return img[y][x];
    }
    else {
        p_prime = img[y - 1][x - 1] + img[y - 1][x] + img[y + 1][x]
            + img[y][x - 1] + img[y][x] + img[y][x + 1]
            + img[y + 1][x - 1] + img[y + 1][x] + img[y + 1][x + 1];
        p_prime = roundup(p_prime / 9.0);
    }


    return p_prime;
}

int GetAvgAtyx3(int y, int x, int** img, int height, int width) {
    int p_prime;
    if (x < 1 || y < 1 || x >= (width - 1) || y >= (height - 1)) {
        return img[y][x];
    }
    else {
        /*p_prime = img[y - 1][x - 1] + img[y - 1][x] + img[y + 1][x]
           + img[y][x - 1] + img[y][x] + img[y][x + 1]
           + img[y + 1][x - 1] + img[y + 1][x] + img[y + 1][x + 1];
        p_prime = roundup(p_prime / 9.0);
        */
        p_prime = 0;
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                p_prime += img[y + dy][x + dx];
            }
        }
        /*for (int yy = y-1; yy <= y+1;yy++) {
           for (int xx = x-1; xx <=x+ 1; xx++) {
              p_prime += img[yy][xx];
           }
        }*/
        p_prime = roundup(p_prime / 9.0);
    }


    return p_prime;
}
int ex0703_2() { //3x3 평균필터
    int height, width;
    int** img = ReadImage((char*)"./TestImages/barbara.png", &height, &width);
    int** img_out = (int**)IntAlloc2(height, width);

    int y = 100, x = 200;
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            img_out[y][x] = GetAvgAtyx3(y, x, img, height, width);
        }
    }

    ImageShow((char*)"input", img, height, width);
    ImageShow((char*)"평균필터", img_out, height, width);

    IntFree2(img, height, width);
    IntFree2(img_out, height, width);

    return 0;
}

int GetAvgAtyx3_5x5(int y, int x, int** img, int height, int width) {
    int p_prime;
    if (x < 2 || y < 2 || x >= (width - 2) || y >= (height - 2)) {
        return img[y][x];
    }
    else {

        p_prime = 0;
        for (int dy = -2; dy <= 2; dy++) {
            for (int dx = -2; dx <= 2; dx++) {
                p_prime += img[y + dy][x + dx];
            }
        }

        p_prime = roundup(p_prime / 25.0);
    }


    return p_prime;
}


int getavg_NxN(int N, int y, int x, int** img, int height, int width)
{
    int p_prime;
    int b = (N - 1) / 2;
    if (x < b || y < b || x >= (width - b) || y >= (height - b))
    {
        return img[y][x];
    }
    else
    {
        p_prime = 0;
        for (int dy = -b; dy <= b; dy++) {
            for (int dx = -b; dx <= b; dx++) {
                p_prime += img[y + dy][x + dx];
            }
        }
        p_prime = roundup((float)p_prime / (N * N));
    }
    return p_prime;
}



int GetAvgAtyx3_NxN(int N, int y, int x, int** img, int height, int width) {
    int p_prime;
    int b = (N - 1) / 2;
    if (x < b || y < b || x >= (width - b) || y >= (height - b)) {
        return img[y][x];
    }
    else {

        p_prime = 0;
        for (int dy = -b; dy <= b; dy++) {
            for (int dx = -b; dx <= b; dx++) {
                p_prime += img[y + dy][x + dx];
            }
        }

        p_prime = roundup((float)p_prime / (N * N));
    }


    return p_prime;
}
int ex0703_3() { //5x5 평균필터
    int height, width;
    int** img = ReadImage((char*)"./TestImages/barbara.png", &height, &width);
    int** img_out = (int**)IntAlloc2(height, width);

    int y = 100, x = 200;
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            img_out[y][x] = GetAvgAtyx3_5x5(y, x, img, height, width);
        }
    }

    ImageShow((char*)"input", img, height, width);
    ImageShow((char*)"평균필터", img_out, height, width);

    IntFree2(img, height, width);
    IntFree2(img_out, height, width);

    return 0;
}
void MeanFiltering(int N, int** img, int** img_out, int height, int width) {
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            img_out[y][x] = GetAvgAtyx3_NxN(N, y, x, img, height, width);
        }
    }

}
int GetAvgAtyx3_NxN_2(int N, int y, int x, int** img, int height, int width) {
    int p_prime;
    int b = (N - 1) / 2;
    //좌표가 음수이면 x=imax(x,0), y =imax(y,0)
    //좌표가 오른바깥이나 하단 바깥의 경우
    // x= imin(x,width-1), y=imin(y,height-1)
    if (x < b || y < b || x >= (width - b) || y >= (height - b)) {
        return img[y][x];
    }
    else {

        p_prime = 0;
        for (int dy = -b; dy <= b; dy++) {
            for (int dx = -b; dx <= b; dx++) {
                int y_new = imin(imax((y + dy), 0), height - 1);
                int x_new = imin(imax((x + dx), 0), width - 1);
                p_prime += img[y_new][x_new];
            }
        }

        p_prime = roundup((float)p_prime / (N * N));
    }


    return p_prime;
}
void MeanFiltering2(int N, int** img, int** img_out, int height, int width) {
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            img_out[y][x] = GetAvgAtyx3_NxN_2(N, y, x, img, height, width);
        }
    }

}
int ex0703_5() {
    int height, width;
    int** img = ReadImage((char*)"./TestImages/barbara.png", &height, &width);
    int** img_out = (int**)IntAlloc2(height, width);

    int N = 5;
    for (int N = 3; N < 15; N += 2) {
        MeanFiltering2(N, img, img_out, height, width);
        ImageShow((char*)"평균필터", img_out, height, width);

    }


    IntFree2(img, height, width);
    IntFree2(img_out, height, width);

    return 0;
}
struct POS2d {
    int y, x;
};
int GetFiltering(float** mask, int y, int x, int** img) {
    //mask[2][0] * img[y + 1][x - 1] + mask[2][1] * img[y + 1][x] + mask[2][2] * img[y + 1][x + 1];
    float out = 0.0;
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            out += mask[dy + 1][dx + 1] * img[y + dy][x + dx];
        }
    }
    return (int)out;
}
int ex0704_1() {
    int height, width;
    int** img = ReadImage((char*)"./TestImages/barbara.png", &height, &width);
    int** img_out = (int**)IntAlloc2(height, width);
    float** mask = (float**)FloatAlloc2(3, 3);

    mask[0][0] = 1.0 / 9; mask[0][1] = 1.0 / 9; mask[0][2] = 1.0 / 9;
    mask[1][0] = 1.0 / 9; mask[1][1] = 1.0 / 9; mask[1][2] = 1.0 / 9;
    mask[2][0] = 1.0 / 9; mask[2][1] = 1.0 / 9; mask[2][2] = 1.0 / 9;

    int y = 100, x = 200;
    //int out=mask[0][0] * img[y - 1][x - 1] + mask[0][1] * img[y - 1][x] + mask[0][2] * img[y - 1][x + 1] +
       //mask[1][0] * img[y][x - 1] + mask[1][1] * img[y][x] + mask[1][2] * img[y][x + 1] +

    int out = GetFiltering(mask, y, x, img);

    ImageShow((char*)"input", img, height, width);
    ImageShow((char*)"output", img_out, height, width);

    IntFree2(img, height, width);
    IntFree2(img_out, height, width);

    return 0;
}
void MaskFiltering(int** img, int** img_out, float** mask, int height, int width) {
    for (int y = 1; y < height - 1; y++) {
        for (int x = 1; x < width - 1; x++) {
            img_out[y][x] = GetFiltering(mask, y, x, img);

        }
    }
}
int GetFiltering2(float** mask, int y, int x, int** img, int height, int width) {
    //mask[2][0] * img[y + 1][x - 1] + mask[2][1] * img[y + 1][x] + mask[2][2] * img[y + 1][x + 1];
    float out = 0.0;
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            int x_new = imin(imax(x + dx, 0), width - 1);
            int y_new = imin(imax(y + dy, 0), height - 1);
            out += mask[dy + 1][dx + 1] * img[y_new][x_new];
        }
    }
    return (int)roundup(out);
}
void MaskFiltering2(int** img, int** img_out, float** mask, int height, int width) {
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            img_out[y][x] = GetFiltering2(mask, y, x, img, height, width);
            img_out[y][x] = imin(imax(img_out[y][x], 0), 255);
        }
    }
}
void MaskFiltering_ABS(int** img, int** img_out, float** mask, int height, int width) {
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            img_out[y][x] = GetFiltering2(mask, y, x, img, height, width);
            img_out[y][x] = abs(img_out[y][x]);
        }
    }
}
int ex0704_5() {
    int height, width;
    int** img = ReadImage((char*)"./TestImages/lena.png", &height, &width);
    int** img_out = (int**)IntAlloc2(height, width);
    float** mask = (float**)FloatAlloc2(3, 3);

    //mask[0][0] = 1.0 / 9; mask[0][1] = 1.0 / 9; mask[0][2] = 1.0 / 9;
    //mask[1][0] = 1.0 / 9; mask[1][1] = 1.0 / 9; mask[1][2] = 1.0 / 9;
    //mask[2][0] = 1.0 / 9; mask[2][1] = 1.0 / 9; mask[2][2] = 1.0 / 9;

    mask[0][0] = -1.0; mask[0][1] = -1.0; mask[0][2] = -1.0;
    mask[1][0] = -1.0; mask[1][1] = 9.0; mask[1][2] = -1.0;
    mask[2][0] = -1.0; mask[2][1] = -1.0; mask[2][2] = -1.0;

    int y = 100, x = 200;
    MaskFiltering2(img, img_out, mask, height, width);

    ImageShow((char*)"input", img, height, width);
    ImageShow((char*)"output", img_out, height, width);

    IntFree2(img, height, width);
    IntFree2(img_out, height, width);

    return 0;
}
int GetGradientMag(int y, int x, int** img, int height, int width) {
    int x_new = imin(imax(x + 1, 0), width - 1);
    int y_new = imin(imax(y + 1, 0), height - 1);
    int fx = img[y][x_new] - img[y][x];
    int fy = img[y_new][x] - img[y][x];
    int mag = imin(abs(fx) + abs(fy), 255);
    return mag;
}
int GetGradientMag_X(int y, int x, int** img, int width) {
    int x_new = imin(imax(x + 1, 0), width - 1);

    int fx = img[y][x_new] - img[y][x];

    int mag = abs(fx);
    return mag;
}
int GetGradientMag_Y(int y, int x, int** img, int height) {

    int y_new = imin(imax(y + 1, 0), height - 1);

    int fy = img[y_new][x] - img[y][x];
    int mag = abs(fy);
    return mag;
}
int ex0705_1() {
    int height, width;
    int** img = ReadImage((char*)"./TestImages/lena.png", &height, &width);
    int** img_out = (int**)IntAlloc2(height, width);

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            img_out[y][x] = GetGradientMag(y, x, img, height, width);
        }
    }


    ImageShow((char*)"input", img, height, width);
    ImageShow((char*)"output", img_out, height, width);

    IntFree2(img, height, width);
    IntFree2(img_out, height, width);



    return 0;
}
int ex_0705_2() {
    int height, width;
    int** img = ReadImage((char*)"./TestImages/바둑판.png", &height, &width);
    int** img_out = (int**)IntAlloc2(height, width);
    int** img_out_x = (int**)IntAlloc2(height, width);
    int** img_out_y = (int**)IntAlloc2(height, width);

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            img_out[y][x] = GetGradientMag(y, x, img, height, width);
            img_out_x[y][x] = GetGradientMag_X(y, x, img, width);
            img_out_y[y][x] = GetGradientMag_Y(y, x, img, height);
        }
    }


    ImageShow((char*)"input", img, height, width);
    ImageShow((char*)"output", img_out, height, width);
    ImageShow((char*)"output_x", img_out_x, height, width);
    ImageShow((char*)"output_y", img_out_y, height, width);
    IntFree2(img, height, width);
    IntFree2(img_out, height, width);



    return 0;
}

int ex0705_3() {
    int A[5] = { 3,7,8,9,11 };
    int maxvalue = A[0];
    for (int n = 1; n < 5; n++) {
        maxvalue = imax(maxvalue, A[n]);
    }



    return 0;
}
int FindMaxValue(int** img, int height, int width) {
    int maxvalue = img[0][0];
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            maxvalue = imax(maxvalue, img[y][x]);
        }
    }
    return maxvalue;
}
int FindMinValue(int** img, int height, int width) {
    int minvalue = img[0][0];
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {

            minvalue = imin(minvalue, img[y][x]);
        }
    }
    return minvalue;
}
int ex0705_4() {
    int height, width;
    int** img = ReadImage((char*)"./TestImages/barbara.png", &height, &width);
    int** img_out = (int**)IntAlloc2(height, width);

    FindMaxValue(img, height, width);
    FindMinValue(img, height, width);

    ImageShow((char*)"input", img, height, width);
    ImageShow((char*)"output", img_out, height, width);

    IntFree2(img, height, width);




    return 0;
}
int ex_0705_5() {
    int height, width;
    int** img = ReadImage((char*)"./TestImages/lena.png", &height, &width);
    int** img_out = (int**)IntAlloc2(height, width);
    int** img_out_x = (int**)IntAlloc2(height, width);
    int** img_out_y = (int**)IntAlloc2(height, width);

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            img_out[y][x] = GetGradientMag(y, x, img, height, width);
            img_out_x[y][x] = GetGradientMag_X(y, x, img, width);
            img_out_y[y][x] = GetGradientMag_Y(y, x, img, height);
        }
    }
    int maxvalue = FindMaxValue(img_out, height, width);
    int maxvalue_x = FindMaxValue(img_out_x, height, width);
    int maxvalue_y = FindMaxValue(img_out_y, height, width);

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            img_out[y][x] = (float)img_out[y][x] / maxvalue * 255.0;
            img_out_x[y][x] = (float)img_out_x[y][x] / maxvalue_x * 255.0;
            img_out_y[y][x] = (float)img_out_y[y][x] / maxvalue_y * 255.0;
        }
    }

    ImageShow((char*)"input", img, height, width);
    ImageShow((char*)"output", img_out, height, width);
    ImageShow((char*)"output_x", img_out_x, height, width);
    ImageShow((char*)"output_y", img_out_y, height, width);
    IntFree2(img, height, width);
    IntFree2(img_out, height, width);



    return 0;
}
int NormalizeByMaxValue(int** img, int height, int width) {
    int maxvalue = FindMaxValue(img, height, width);


    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            img[y][x] = (float)img[y][x] / maxvalue * 255.0;

        }
    }
    return maxvalue;
}
int ex0705_6() {
    int height, width;
    int** img = ReadImage((char*)"./TestImages/lena.png", &height, &width);
    int** img_out = (int**)IntAlloc2(height, width);
    int** img_out_x = (int**)IntAlloc2(height, width);
    int** img_out_y = (int**)IntAlloc2(height, width);

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            img_out[y][x] = GetGradientMag(y, x, img, height, width);
            img_out_x[y][x] = GetGradientMag_X(y, x, img, width);
            img_out_y[y][x] = GetGradientMag_Y(y, x, img, height);
        }
    }
    NormalizeByMaxValue(img_out, height, width);
    NormalizeByMaxValue(img_out_x, height, width);
    NormalizeByMaxValue(img_out_y, height, width);

    ImageShow((char*)"input", img, height, width);
    ImageShow((char*)"output", img_out, height, width);
    ImageShow((char*)"output_x", img_out_x, height, width);
    ImageShow((char*)"output_y", img_out_y, height, width);
    IntFree2(img, height, width);
    IntFree2(img_out, height, width);



    return 0;
}
int ex0705_7() { //라플라시안
    int height, width;
    int** img = ReadImage((char*)"./TestImages/lena.png", &height, &width);
    int** img_out = (int**)IntAlloc2(height, width);
    float** mask = (float**)FloatAlloc2(3, 3);

    //mask[0][0] = 0.0; mask[0][1] = -1.0; mask[0][2] = 0.0;
    //mask[1][0] = -1.0; mask[1][1] = 4.0; mask[1][2] = -1.0;
    //mask[2][0] = 0.0; mask[2][1] =-1.0; mask[2][2] = 0.0;

    mask[0][0] = -1.0; mask[0][1] = -1.0; mask[0][2] = -1.0;
    mask[1][0] = -1.0; mask[1][1] = 8.0; mask[1][2] = -1.0;
    mask[2][0] = -1.0; mask[2][1] = -1.0; mask[2][2] = -1.0;

    MaskFiltering_ABS(img, img_out, mask, height, width);
    int maxvalue = NormalizeByMaxValue(img_out, height, width);
    ImageShow((char*)"input", img, height, width);
    ImageShow((char*)"output", img_out, height, width);

    IntFree2(img, height, width);
    IntFree2(img_out, height, width);

    return 0;
}
int ex0705_8() { //소벨
    int height, width;
    int** img = ReadImage((char*)"./TestImages/lena.png", &height, &width);
    int** img_out = (int**)IntAlloc2(height, width);
    int** img_out_x = (int**)IntAlloc2(height, width);
    int** img_out_y = (int**)IntAlloc2(height, width);
    float** mask_x = (float**)FloatAlloc2(3, 3);
    float** mask_y = (float**)FloatAlloc2(3, 3);
    mask_x[0][0] = 1; mask_x[0][1] = 0; mask_x[0][2] = -1;
    mask_x[1][0] = 2; mask_x[1][1] = 0; mask_x[1][2] = -2;
    mask_x[2][0] = 1; mask_x[2][1] = 0; mask_x[2][2] = -1;

    mask_y[0][0] = 1; mask_y[0][1] = 2; mask_y[0][2] = 1;
    mask_y[1][0] = 0; mask_y[1][1] = 0; mask_y[1][2] = 0;
    mask_y[2][0] = -1; mask_y[2][1] = -2.0; mask_y[2][2] = -1;

    MaskFiltering_ABS(img, img_out_x, mask_x, height, width);
    NormalizeByMaxValue(img_out_x, height, width);

    MaskFiltering_ABS(img, img_out_y, mask_y, height, width);
    NormalizeByMaxValue(img_out_y, height, width);


    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            img_out[y][x] = abs(img_out_x[y][x]) + abs(img_out_y[y][x]);
        }
    }
    NormalizeByMaxValue(img_out, height, width);

    ImageShow((char*)"input", img, height, width);
    ImageShow((char*)"outputx", img_out_x, height, width);
    ImageShow((char*)"outputy", img_out_y, height, width);
    ImageShow((char*)"output", img_out, height, width);
    IntFree2(img, height, width);
    IntFree2(img_out, height, width);

    return 0;
}
void Sobel(int** img, int height, int width, int** img_out) {
    int** img_out_x = (int**)IntAlloc2(height, width);
    int** img_out_y = (int**)IntAlloc2(height, width);
    float** mask_x = (float**)FloatAlloc2(3, 3);
    float** mask_y = (float**)FloatAlloc2(3, 3);
    mask_x[0][0] = 1; mask_x[0][1] = 0; mask_x[0][2] = -1;
    mask_x[1][0] = 2; mask_x[1][1] = 0; mask_x[1][2] = -2;
    mask_x[2][0] = 1; mask_x[2][1] = 0; mask_x[2][2] = -1;

    mask_y[0][0] = 1; mask_y[0][1] = 2; mask_y[0][2] = 1;
    mask_y[1][0] = 0; mask_y[1][1] = 0; mask_y[1][2] = 0;
    mask_y[2][0] = -1; mask_y[2][1] = -2.0; mask_y[2][2] = -1;

    MaskFiltering_ABS(img, img_out_x, mask_x, height, width);
    MaskFiltering_ABS(img, img_out_y, mask_y, height, width);


    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            img_out[y][x] = abs(img_out_x[y][x]) + abs(img_out_y[y][x]);
        }
    }
    NormalizeByMaxValue(img_out, height, width);

    IntFree2(img_out_x, height, width);
    IntFree2(img_out_y, height, width);
    FloatFree2(mask_x, 3, 3);
    FloatFree2(mask_y, 3, 3);
}
int ex0706_9() { //소벨
    int height, width;
    int** img = ReadImage((char*)"./TestImages/lena.png", &height, &width);
    int** img_out = (int**)IntAlloc2(height, width);

    Sobel(img, height, width, img_out);
    ImageShow((char*)"input", img, height, width);

    ImageShow((char*)"output", img_out, height, width);
    IntFree2(img, height, width);
    IntFree2(img_out, height, width);

    return 0;
}
int ex0706_1() {
    int height, width;
    int** img = ReadImage((char*)"./TestImages/lena.png", &height, &width);
    int** img_out1 = (int**)IntAlloc2(height, width);
    int** img_out2 = (int**)IntAlloc2(height, width);
    float** mask1 = (float**)FloatAlloc2(3, 3);
    float** mask2 = (float**)FloatAlloc2(3, 3);
    //mask[0][0] = 1.0 / 9; mask[0][1] = 1.0 / 9; mask[0][2] = 1.0 / 9;
    //mask[1][0] = 1.0 / 9; mask[1][1] = 1.0 / 9; mask[1][2] = 1.0 / 9;
    //mask[2][0] = 1.0 / 9; mask[2][1] = 1.0 / 9; mask[2][2] = 1.0 / 9;

    mask1[0][0] = -1.0; mask1[0][1] = -1.0; mask1[0][2] = -1.0;
    mask1[1][0] = -1.0; mask1[1][1] = 9.0; mask1[1][2] = -1.0;
    mask1[2][0] = -1.0; mask1[2][1] = -1.0; mask1[2][2] = -1.0;

    MaskFiltering2(img, img_out1, mask1, height, width);

    mask2[0][0] = 0; mask2[0][1] = -1.0; mask2[0][2] = 0;
    mask2[1][0] = -1.0; mask2[1][1] = 5.0; mask2[1][2] = -1.0;
    mask2[2][0] = 0; mask2[2][1] = -1.0; mask2[2][2] = 0;

    MaskFiltering2(img, img_out2, mask2, height, width);

    ImageShow((char*)"input", img, height, width);
    ImageShow((char*)"output1", img_out1, height, width);
    ImageShow((char*)"output2", img_out2, height, width);
    IntFree2(img, height, width);
    IntFree2(img_out1, height, width);
    IntFree2(img_out2, height, width);
    return 0;
}
int ex0706_2() {
    int height, width;
    int** img = ReadImage((char*)"./TestImages/lena.png", &height, &width);
    int** img_out = (int**)IntAlloc2(height, width);

    float** mask = (float**)FloatAlloc2(3, 3);

    ImageShow((char*)"input", img, height, width);
    //mask[0][0] = 1.0 / 9; mask[0][1] = 1.0 / 9; mask[0][2] = 1.0 / 9;
    //mask[1][0] = 1.0 / 9; mask[1][1] = 1.0 / 9; mask[1][2] = 1.0 / 9;
    //mask[2][0] = 1.0 / 9; mask[2][1] = 1.0 / 9; mask[2][2] = 1.0 / 9;
    float a = 0.25;
    for (float a = 0.0; a < 1.0; a += 0.1) {
        mask[0][0] = -a; mask[0][1] = -a; mask[0][2] = -a;
        mask[1][0] = -a; mask[1][1] = 1 + 8 * a; mask[1][2] = -a;
        mask[2][0] = -a; mask[2][1] = -a; mask[2][2] = -a;

        MaskFiltering2(img, img_out, mask, height, width);


        ImageShow((char*)"output", img_out, height, width);
    }


    IntFree2(img, height, width);
    IntFree2(img_out, height, width);

    return 0;
}
int ex0706_3() {
    int height, width;
    int** img = ReadImage((char*)"./TestImages/lena.png", &height, &width);
    int** img_out = (int**)IntAlloc2(height, width);

    float** mask = (float**)FloatAlloc2(3, 3);

    ImageShow((char*)"input", img, height, width);
    //mask[0][0] = 1.0 / 9; mask[0][1] = 1.0 / 9; mask[0][2] = 1.0 / 9;
    //mask[1][0] = 1.0 / 9; mask[1][1] = 1.0 / 9; mask[1][2] = 1.0 / 9;
    //mask[2][0] = 1.0 / 9; mask[2][1] = 1.0 / 9; mask[2][2] = 1.0 / 9;
    float a = 0.25;
    for (float a = 0.0; a < 1.0; a += 0.1) {
        mask[0][0] = 0; mask[0][1] = -a; mask[0][2] = 0;
        mask[1][0] = -a; mask[1][1] = 1 + 4 * a; mask[1][2] = -a;
        mask[2][0] = 0; mask[2][1] = -a; mask[2][2] = 0;

        MaskFiltering2(img, img_out, mask, height, width);


        ImageShow((char*)"output", img_out, height, width);
    }


    IntFree2(img, height, width);
    IntFree2(img_out, height, width);

    return 0;
}
void Swap(int* A, int* B) {
    int tmp;

    tmp = *A;
    *A = *B;
    *B = tmp;
}
void Bubbling(int* A, int num) {
    for (int n = 0; n < num - 1; n++) {

        if (A[n] > A[n + 1]) {
            Swap(&A[n], &A[n + 1]);
        }

    }
}
void BubbleSort(int* A, int num) {
    for (int n = 0; n < num - 1; n++) {
        Bubbling(A, num - n);
    }
}
#define NUM 5
int ex0706_4() {
    int A[NUM] = { 7,3,2,5,1 };
    BubbleSort(A, NUM);


    return 0;
}
void GetBlock1D(int y, int x, int** img, int* data, int height, int width) {
    int cnt = 0;
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            int y_new = imin(imax(y + dy, 0), height - 1);
            int x_new = imin(imax(x + dx, 0), width - 1);
            data[cnt] = img[y_new][x_new];
            cnt++;
        }
    }
}
int GetMedianValue(int y, int x, int** img, int height, int width) {
    int data[9] = { 0 };
    GetBlock1D(y, x, img, data, height, width);
    BubbleSort(data, 3 * 3);
    return data[4];; //(N-1)/2
}
void MedianFiltering(int** img, int height, int width, int** img_out) {
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            img_out[y][x] = GetMedianValue(y, x, img, height, width);
        }
    }
}
int ex0706_11() {
    int height, width;
    int** img = ReadImage((char*)"./TestImages/lenaSP20.png", &height, &width);
    int** img_out = (int**)IntAlloc2(height, width);
    int** img_out2 = (int**)IntAlloc2(height, width);
    MedianFiltering(img, height, width, img_out);
    MedianFiltering(img_out, height, width, img_out2);


    ImageShow((char*)"input", img, height, width);
    ImageShow((char*)"output1", img_out, height, width);
    ImageShow((char*)"output2", img_out2, height, width);
    IntFree2(img, height, width);
    return 0;
}
void ResizeWithRepeat(int y, int x, int** img, int height, int width, int** img_out)
{
    int x_out = 2 * x;
    int y_out = 2 * y;
    img_out[y_out][x_out] = img[y][x];
    img_out[y_out][x_out + 1] = img[y][x];  //R
    img_out[y_out + 1][x_out] = img[y][x];  //B
    img_out[y_out + 1][x_out + 1] = img[y][x];  //C
}
void ResizeMethod1(int height, int width, int** img, int** img_out) {
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            ResizeWithRepeat(y, x, img, height, width, img_out);
        }
    }
}
int es0708_1() {

    int height, width;
    int** img = ReadImage((char*)"./TestImages/s_barbara.png", &height, &width);
    int height2 = 2 * height;
    int width2 = 2 * width;
    int** img_out = (int**)IntAlloc2(height2, width2);

    ResizeMethod1(height, width, img, img_out);


    ImageShow((char*)"input", img, height, width);
    ImageShow((char*)"output", img_out, height2, width2);

    IntFree2(img, height, width);

    return 0;
}
void FillEvenPixels(int** img, int height, int width, int** img_out) {
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int x_out = 2 * x;
            int y_out = 2 * y;
            img_out[y_out][x_out] = img[y][x];
        }
    }
}
void Fill_EF_Pixels(int** img, int height, int width, int** img_out) {
    int height2 = height * 2;
    int width2 = width * 2;
    for (int y = 0; y < height - 1; y++) {
        for (int x = 0; x < width - 1; x++) {

            int x_out = 2 * x;
            int y_out = 2 * y;
            int A = img_out[y_out][x_out];
            int B = img_out[y_out][x_out + 2];
            int C = img_out[y_out + 2][x_out];
            int E = roundup((A + B) / 2.0); //-- > (y_out, x_out + 1)
            int F = roundup((A + C) / 2.0);//-- > (y_out+1, x_out )
            img_out[y_out][x_out + 1] = E;
            img_out[y_out + 1][x_out] = F;
        }
    }
    // 가장자리 E를 채우기
    int y_out = 0;
    int x_out = width2 - 2;
    for (y_out = 0; y_out < height2; y_out += 2) {
        img_out[y_out][x_out + 1] = img_out[y_out][x_out];
        img_out[y_out + 1][x_out] = img_out[y_out][x_out];
    }
    // 가장자리 F를 채우기
    y_out = height2 - 2;

    for (x_out = 0; x_out < width2; x_out += 2) {
        img_out[y_out + 1][x_out] = img_out[y_out][x_out];
        img_out[y_out][x_out + 1] = img_out[y_out][x_out];
    }
}
void Fill_I_Pixels(int** img, int height, int width, int** img_out) {
    int height2 = height * 2;
    int width2 = width * 2;
    //I 채우기
    for (int y = 0; y < height - 1; y++) {
        for (int x = 0; x < width - 1; x++) {

            int x_out = 2 * x;
            int y_out = 2 * y;
            int E = img_out[y_out + 1][x_out + 2];
            int F = img_out[y_out + 1][x_out];

            img_out[y_out + 1][x_out + 1] = roundup((E + F) / 2.0);
        }
    }
    //I 가장자리 채우기
    int y_out = 0;
    int x_out = width2 - 2;
    for (y_out = 0; y_out < height2; y_out += 2) {
        img_out[y_out + 1][x_out + 1] = img_out[y_out + 1][x_out];
    }
    y_out = height - 2;
    for (x_out = 0; x_out < width2; x_out += 2) {
        img_out[y_out + 1][x_out + 1] = img_out[y_out + 1][x_out];

    }
}
void ResizeMethod2(int** img, int height, int width, int** img_out, int height2, int width2) {
    if (height2 != 2 * height || width2 != 2 * width) {
        printf("\n 출력영상 크기가 두배가 아니다");
        return;
    }

    FillEvenPixels(img, height, width, img_out);
    Fill_EF_Pixels(img, height, width, img_out);
    Fill_I_Pixels(img, height, width, img_out);
}

int ex0708_2() {

    int height, width;
    int** img = ReadImage((char*)"./TestImages/barbara.png", &height, &width);
    int height2 = height / 2;
    int width2 = width / 2;
    int** img_out = (int**)IntAlloc2(height2, width2);
    ResizeMethod2(img, height, width, img_out, height2, width2);

    ImageShow((char*)"input", img, height, width);
    ImageShow((char*)"output", img_out, height2, width2);

    IntFree2(img, height, width);

    return 0;
}
void DownSize(int** img, int** img_out, int height2, int width2) {
    for (int y_out = 0; y_out < height2; y_out++) {
        for (int x_out = 0; x_out < width2; x_out++) {
            int y = 2 * y_out;
            int x = 2 * x_out;
            img_out[y_out][x_out] = img[y][x];
        }
    }
}
int ex0708_3() {

    int height, width;
    int** img = ReadImage((char*)"./TestImages/barbara.png", &height, &width);
    int height2 = height / 2;
    int width2 = width / 2;
    int** img_out = (int**)IntAlloc2(height2, width2);
    int y_out = 100, x_out = 200;

    DownSize(img, img_out, height2, width2);
    //img_out[y_out][x_out] = img[2 * y_out][2 * x_out];

    ImageShow((char*)"input", img, height, width);
    ImageShow((char*)"output", img_out, height2, width2);

    IntFree2(img, height, width);

    return 0;
}
int BiLinear(float y_in, float x_in,
    int** img, int height, int width)
{
    int y = (int)y_in;
    int x = (int)x_in;

    float dy = y_in - y;
    float dx = x_in - x;

    int y_new = imin(imax((y + 1), 0), height - 1);
    int x_new = imin(imax((x + 1), 0), width - 1);

    int y_1_new = imin(imax((y + 1), 0), height - 1);
    int x_1_new = imin(imax((x + 1), 0), width - 1);

    int A = img[y_new][x_new];
    int B = img[y_new][x_1_new];
    int C = img[y_1_new][x_new];
    int D = img[y_1_new][x_1_new];

    int I = roundup((1 - dx) * (1 - dy) * A + dx * (1 - dy) * B
        + (1 - dx) * dy * C + dx * dy * D);

    return I;
}

int BiLinear2(float y_in, float x_in,
    int** img, int height, int width) //가장자리 부분을 검정, 유용 변환을 주기위함
{
    int y = (int)y_in;
    int x = (int)x_in;

    float dy = y_in - y;
    float dx = x_in - x;

    if (x < 0 || y < 0 || x + 1 >= width || y + 1 >= height)
        return 0;
    int A = img[y][x];
    int B = img[y][x + 1];
    int C = img[y + 1][x];
    int D = img[y + 1][x + 1];

    int I = roundup((1 - dx) * (1 - dy) * A + dx * (1 - dy) * B
        + (1 - dx) * dy * C + dx * dy * D);

    return I;
}
int ex0709_1() //2배로 크게
{

    int height, width;
    int** img = (int**)ReadImage((char*)"./TestImages/s_lena.png", &height, &width);
    int height2 = height * 2;
    int width2 = width * 2;
    int** img_out = (int**)IntAlloc2(height2, width2);


    for (int y = 0; y < height2; y++) {
        for (int x = 0; x < width2; x++) {
            img_out[y][x] = BiLinear(y / 2.0, x / 2.0, img, height, width);
        }
    }


    ImageShow((char*)"input", img, height, width);
    ImageShow((char*)"output", img_out, height2, width2);

    IntFree2(img, height, width);
    IntFree2(img_out, height2, width2);
    return 0;

}
int ex0709_2()
{

    int height, width;
    int** img = (int**)ReadImage((char*)"./TestImages/s_lena.png", &height, &width);
    float n = 1.5;
    float heightn = height * n;
    float widthn = width * n;
    int** img_out = (int**)IntAlloc2(heightn, widthn);



    for (int y = 0; y < heightn; y++) {
        for (int x = 0; x < widthn; x++) {
            img_out[y][x] = BiLinear(y / n, x / n, img, height, width);
        }
    }

    ImageShow((char*)"input", img, height, width);
    ImageShow((char*)"output", img_out, heightn, widthn);

    IntFree2(img, height, width);
    IntFree2(img_out, heightn, widthn);
    return 0;

}
int ex0709_3() // 영상이동
{
    int height, width;
    int** img = (int**)ReadImage((char*)"./TestImages/barbara.png", &height, &width);
    int** img_out = (int**)IntAlloc2(height, width);

    //x0 = x +tx ==> x = x0 - tx
    //y0 = y +ty ==> y = y0 - ty
    float ty = 30.5;
    float tx = 20.7;

    for (int y0 = 0; y0 < height; y0++) {
        for (int x0 = 0; x0 < width; x0++) {
            float y = y0 - ty;
            float x = x0 - tx;
            img_out[y0][x0] = BiLinear2(y, x, img, height, width);
        }
    }

    ImageShow((char*)"input", img, height, width);
    ImageShow((char*)"output", img_out, height, width);

    IntFree2(img, height, width);
    IntFree2(img_out, height, width);
    return 0;


}
int ex0709_4() // 영상회전 (중앙 기준)
{
    int height, width;
    int** img = (int**)ReadImage((char*)"./TestImages/barbara.png", &height, &width);
    int** img_out = (int**)IntAlloc2(height, width);

    // x = cos*x0 + sin*y0
    // y = -sin*x0+ cos*y0
    float theta = -15.0 * (CV_PI / 180.0);

    for (int y0 = 0; y0 < height; y0++) {
        for (int x0 = 0; x0 < width; x0++) {
            float y = -sin(theta) * x0 + cos(theta) * y0;
            float x = cos(theta) * x0 + sin(theta) * y0;
            img_out[y0][x0] = BiLinear2(y, x, img, height, width);
        }
    }
    /*double theta = 45.0;
    Rotation(theta, img, height, width, img_out);
 */
    ImageShow((char*)"input", img, height, width);
    ImageShow((char*)"output", img_out, height, width);

    IntFree2(img, height, width);
    IntFree2(img_out, height, width);
    return 0;


}
int ex0709_5() // 영상회전
{
    int height, width;
    int** img = (int**)ReadImage((char*)"./TestImages/barbara.png", &height, &width);
    int** img_out = (int**)IntAlloc2(height, width);

    // x = cos*x0 + sin*y0
    // y = -sin*x0+ cos*y0
    float theta = 45.0 * (CV_PI / 180.0);
    int yc = 256, xc = 256;

    for (int y0 = 0; y0 < height; y0++) {
        for (int x0 = 0; x0 < width; x0++) {
            float y = -sin(theta) * (x0 - xc) + cos(theta) * (y0 - yc) + yc;
            float x = cos(theta) * (x0 - xc) + sin(theta) * (y0 - yc) + xc;
            img_out[y0][x0] = BiLinear2(y, x, img, height, width);
        }
    }


    /*double theta = 45.0;
    Rotation(theta, img, height, width, img_out);
 */

    ImageShow((char*)"input", img, height, width);
    ImageShow((char*)"output", img_out, height, width);

    IntFree2(img, height, width);
    IntFree2(img_out, height, width);
    return 0;


}
void ScalingPlueRotation(float alpha, float theta,
    int** img, int height, int width, int** img_out)
{
    int xc = width / 2;
    int yc = height / 2;
    for (int y0 = 0; y0 < height; y0++) {
        for (int x0 = 0; x0 < width; x0++) {
            float y = 1.0 / alpha * (-sin(theta) * (x0 - xc) + cos(theta) * (y0 - yc)) + yc;
            float x = 1.0 / alpha * (cos(theta) * (x0 - xc) + sin(theta) * (y0 - yc)) + xc;
            img_out[y0][x0] = BiLinear2(y, x, img, height, width);
        }
    }
}

int ex0709_6() // 영상회전 스케일링
{
    int height, width;
    int** img = (int**)ReadImage((char*)"./TestImages/barbara.png", &height, &width);
    int** img_out = (int**)IntAlloc2(height, width);

    // x = cos*x0 + sin*y0
    // y = -sin*x0+ cos*y0
    float theta = 45.0 * (CV_PI / 180.0);
    int yc = 256, xc = 256;

    float alpha = 0.8;

    ScalingPlueRotation(alpha, theta, img, height, width, img_out);

    ImageShow((char*)"input", img, height, width);
    ImageShow((char*)"output", img_out, height, width);

    IntFree2(img, height, width);
    IntFree2(img_out, height, width);
    return 0;


}
void AffineTransform(float a, float b, float c, float d,
    float ty, float tx,
    int** img, int height, int width, int** img_out)
{
    float det = a * d - b * c;
    float a_p = d / det;
    float b_p = -b / det;
    float c_p = -c / det;
    float d_p = a / det;

    for (int y0 = 0; y0 < height; y0++) {
        for (int x0 = 0; x0 < width; x0++) {
            float x = a_p * (x0 - tx) + b_p * (y0 - ty);
            float y = c_p * (x0 - tx) + d_p * (y0 - ty);
            img_out[y0][x0] = BiLinear2(y, x, img, height, width);
        }
    }
}
int ex0710_1() //어파인 변환
{
    int height, width;
    int** img = (int**)ReadImage((char*)"./TestImages/barbara.png", &height, &width);
    int** img_out = (int**)IntAlloc2(height, width);

    int yc = 256, xc = 256;

    float a = 1.0, b = 1.0;
    float c = 0.0, d = 1.0;
    float tx = 10.0, ty = 20.0;

    AffineTransform(a, b, c, d, ty, tx, img, height, width, img_out);

    ImageShow((char*)"input", img, height, width);
    ImageShow((char*)"output", img_out, height, width);

    IntFree2(img, height, width);
    IntFree2(img_out, height, width);
    return 0;

}

//구조체 만드는 법
struct AffinePara {
    float a; //설명
    float b;
    float c;
    float d;
    float ty;
    float tx;
};

void AffineTransform2(struct AffinePara para,
    int** img, int height, int width, int** img_out)
{

    float det = para.a * para.d - para.b * para.c;
    float a_p = para.d / det;
    float b_p = -para.b / det;
    float c_p = -para.c / det;
    float d_p = para.a / det;

    for (int y0 = 0; y0 < height; y0++) {
        for (int x0 = 0; x0 < width; x0++) {
            float x = a_p * (x0 - para.tx) + b_p * (y0 - para.ty);
            float y = c_p * (x0 - para.tx) + d_p * (y0 - para.ty);
            img_out[y0][x0] = BiLinear2(y, x, img, height, width);
        }
    }
}
int ex0710_2()
{
    int height, width;
    int** img = (int**)ReadImage((char*)"./TestImages/barbara.png", &height, &width);
    int** img_out = (int**)IntAlloc2(height, width);

    int yc = 256, xc = 256;

    float a = 1.0, b = 1.0;
    float c = 0.0, d = 1.0;
    float tx = 10.0, ty = 20.0;

    AffinePara para; //C 에서는 struct AffinePara para ->struct 필수
    para.a = a;
    para.b = b;
    para.c = c;
    para.d = d;
    para.tx = tx;
    para.ty = ty;

    AffineTransform2(para, img, height, width, img_out);

    ImageShow((char*)"input", img, height, width);
    ImageShow((char*)"output", img_out, height, width);

    IntFree2(img, height, width);
    IntFree2(img_out, height, width);
    return 0;

}
struct Parameters {
    float a, b, c, d, tx, ty;
    int** img;
    int** img_out;
    int height, width;
};

void AffineTransform3(struct Parameters para)
{

    float det = para.a * para.d - para.b * para.c;
    float a_p = para.d / det;
    float b_p = -para.b / det;
    float c_p = -para.c / det;
    float d_p = para.a / det;

    for (int y0 = 0; y0 < para.height; y0++) {
        for (int x0 = 0; x0 < para.width; x0++) {
            float x = a_p * (x0 - para.tx) + b_p * (y0 - para.ty);
            float y = c_p * (x0 - para.tx) + d_p * (y0 - para.ty);
            para.img_out[y0][x0] = BiLinear2(y, x, para.img, para.height, para.width);
        }
    }
}
int ex0710_3()
{
    int height, width;
    int** img = (int**)ReadImage((char*)"./TestImages/barbara.png", &height, &width);
    int** img_out = (int**)IntAlloc2(height, width);

    int yc = 256, xc = 256;

    float a = 1.0, b = 0.0;
    float c = 0.0, d = 1.0;
    float tx = 10.0, ty = 10.0;

    Parameters para; //C 에서는 struct AffinePara para ->struct 필수
    para.a = a;
    para.b = b;
    para.c = c;
    para.d = d;
    para.tx = tx;
    para.ty = ty;
    para.img = img;
    para.img_out = img_out;
    para.height = height;
    para.width = width;
    AffineTransform3(para);

    ImageShow((char*)"input", img, height, width);
    ImageShow((char*)"output", img_out, height, width);

    IntFree2(img, height, width);
    IntFree2(img_out, height, width);
    return 0;

}
float ComputeMAD(int** blockA, int** blockB, int by, int bx)
{
    float mad = 0;
    for (int y = 0; y < by; y++) {
        for (int x = 0; x < bx; x++) {
            mad += abs(blockA[y][x] - blockB[y][x]);
        }
    }
    mad = mad / (by * bx);

    return mad;
}
#define SQ(x) ((x)*(x))    //매크로 할 때 반드시 괄호 꼭
float ComputeMSE(int** blockA, int** blockB, int by, int bx)
{
    float mse = 0;
    for (int y = 0; y < by; y++) {
        for (int x = 0; x < bx; x++) {
            mse += SQ(blockA[y][x] - blockB[y][x]);
        }
    }
    mse = mse / (by * bx);

    return mse;
}
int ex0710_4()
{

    int height, width;
    int** img1 = (int**)ReadImage((char*)"./TestImages/barbara.png", &height, &width);
    int** img2 = (int**)ReadImage((char*)"./TestImages/barbaraGN10.png", &height, &width);
    //int** img_out = (int**)IntAlloc2(height, width);

    float mad = ComputeMAD(img1, img2, height, width);
    float mse = ComputeMSE(img1, img2, height, width);

    printf("\n mad = %f", mad);
    printf("\n mse = %f", mse);
    ImageShow((char*)"input1", img1, height, width);
    ImageShow((char*)"input2", img2, height, width);
    //ImageShow((char*)"output", img_out, height, width);

    //IntFree2(img1, height, width);
    //IntFree2(img2, height, width);
    //IntFree2(img_out, height, width);
    return 0;

}

void ReadBlock(int yp, int xp, int** block, int** img, int by, int bx) {
    for (int y = 0; y < by; y++) {
        for (int x = 0; x < bx; x++) {
            block[y][x] = img[y + yp][x + xp];
        }
    }
}

void WriteBlock(int yp, int xp, int** block, int** img, int by, int bx) {
    for (int y = 0; y < by; y++) {
        for (int x = 0; x < bx; x++) {
            img[y + yp][x + xp]= block[y][x];
        }
    }
}

void TemplateMatching(
    int** block, int by, int bx,
    int** img, int height, int width,
    int* yp_min_out, int* xp_min_out, float* mad_min_out)
{
    float mad = 0;
    float mad_min = FLT_MAX;
    int yp_min = 0, xp_min = 0;

    int** block_img = (int**)IntAlloc2(by, bx);

    for (int yp = 0; yp < height - by; yp++) {
        for (int xp = 0; xp < width - bx; xp++) {
            ReadBlock(yp, xp, block_img, img, by, bx);
            mad = ComputeMAD(block, block_img, by, bx);


            if (mad < mad_min) {
                mad_min = mad;
                yp_min = yp;
                xp_min = xp;
            }
        }
    }
    IntFree2(block_img, by, bx);

    *yp_min_out = yp_min;
    *xp_min_out = xp_min;
    *mad_min_out = mad_min;
}

int ex0710_05()
{

    int height, width;
    int** img = (int**)ReadImage((char*)"./TestImages/barbara.png", &height, &width);
    int by, bx;
    int** block = ReadImage((char*)"./TestImages/barbara_template.png", &by, &bx);

    int yp_min, xp_min;
    float mad_min;

    TemplateMatching(block, by, bx, img, height, width, &yp_min, &xp_min, &mad_min);

    printf("\n (x , y) = (%d, %d) ,mad = %f", xp_min, yp_min, mad_min);

    ImageShow((char*)"input", img, height, width);
    ImageShow((char*)"input2", block, by, bx);
    //ImageShow((char*)"output", img_out, height, width);

    IntFree2(img, height, width);
    IntFree2(block, by, bx);
    //IntFree2(img_out, height, width);
    return 0;

}

#define NUM_DB 510

void FindDBindex(int y, int x, int** img, int by, int bx, int** db[], int* index_out) {
   
    int** block = (int**)IntAlloc2(by, bx);
    int n_min = 0;
    float mad = 0.0, mad_min = FLT_MAX;

    ReadBlock(y, x,block, img, by, bx);
    for (int n = 0; n < NUM_DB; n++) {
        mad = ComputeMAD(block, db[n], by, bx);
        if (mad < mad_min) {
            mad_min = mad;
            n_min = n;
        }
    }
    *index_out = n_min;
    IntFree2(block, by, bx);

}

void MakeMosaicImage(int**img, int height, int width , 
    int** db[], int by, int bx, int **img_out)
{
    for (int y = 0; y < height; y += by) {
        for (int x = 0; x < width; x += bx) {
            int index_out = 0;
            FindDBindex(y, x, img, by, bx, db, &index_out);
            WriteBlock(y, x, db[index_out], img_out, by, bx);
        }
    }
}

int ex0711_01()
{
    int height, width;
    int** img = ReadImage((char*)"./TestImages/barbara.png", &height, &width);
    int** img_out = (int**)IntAlloc2(height, width);

    char name[100];
    int by, bx;

    
    int** db[NUM_DB];
    for (int n = 0; n < NUM_DB; n++) {
        sprintf_s(name, "./face/dbs%04d.jpg", n);
        db[n] = ReadImage(name, &by, &bx);
    }

    MakeMosaicImage(img, height, width, db, by, bx, img_out);
    
    

    ImageShow((char*)"input", img, height, width);
    ImageShow((char*)"output", img_out, height, width);

    IntFree2(img, height, width);




    return 0;

}


int main()
{
    int height, width;
    int** img = ReadImage((char*)"./TestImages/barbara.png", &height, &width);
    int** img_out = (int**)IntAlloc2(height, width);

    char name[100];
    int by, bx;


    int** db[NUM_DB];
    for (int n = 0; n < NUM_DB; n++) {
        sprintf_s(name, "./face/dbs%04d.jpg", n);
        db[n] = ReadImage(name, &by, &bx);
    }

    int** db_small[NUM_DB];
    int by_s = by / 2;
    int bx_s = bx / 2;
    for (int n = 0; n < NUM_DB; n++) {
        db_small[n] = (int**)IntAlloc2(by_s, bx_s);
        DownSize(db[n], db_small[n], by_s, bx_s);
    }

    MakeMosaicImage(img, height, width, db_small, by_s, bx_s, img_out);



    ImageShow((char*)"input", img, height, width);
    ImageShow((char*)"output", img_out, height, width);

    IntFree2(img, height, width);




    return 0;

}







int TTTT() {
    int height, width;
    int** img = ReadImage((char*)"./TestImages/barbara.png", &height, &width);
    int** img_out = (int**)IntAlloc2(height, width);

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {

        }
    }

    ImageShow((char*)"input", img, height, width);
    ImageShow((char*)"output", img_out, height, width);

    IntFree2(img, height, width);




    return 0;
}




