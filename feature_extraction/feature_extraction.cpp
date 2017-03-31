#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/features2d.hpp>


using namespace std;
using namespace cv;

int main(int argc, char** argv)
{
  if( argc != 3 ){
    cout<<" usage: feature_extracion src_Image1 src_Image2 "<<endl;
    return 1;
  }
  //读取图片
  cv::Mat src_Image1 = imread( argv[1], CV_LOAD_IMAGE_COLOR );
  cv::Mat src_Image2 = imread( argv[2], CV_LOAD_IMAGE_COLOR );

  //初始化
  std::vector<KeyPoint>keypoint_1,keypoint_2;
  Mat descriptors_1,descriptors_2;
  Ptr<ORB> orb = ORB::create( 500, 1.2f, 8, 31, 0, 2, ORB::HARRIS_SCORE, 31, 20 );

  //--第一步检测Oriented FAST角点位置
  orb->detect( src_Image1, keypoint_1 );
  orb->detect( src_Image2, keypoint_2 );

  //--第二步:根据角点位置计算BRIEF描述子
  orb->compute( src_Image1, keypoint_1, descriptors_1 );
  orb->compute( src_Image2, keypoint_2, descriptors_2 );

  Mat outing1;
  drawKeypoints( src_Image1, keypoint_1, outing1, Scalar::all(-1), DrawMatchesFlags::DEFAULT );
  imshow( "characteristic points ORB",outing1 );

  //--第三步:对两幅图中的描述子进行匹配,使用BRIEF Hamming 距离
  vector<DMatch> matches;
  BFMatcher matcher ( NORM_HAMMING );
  matcher.match( descriptors_1, descriptors_2, matches );

  //-- 第四步匹配点对筛选
  double min_dist = 10000, max_dist = 0;

  //找出所有的匹配之间的最小距离和最大距离,即是最相似的和最不相似的两组点之间的距离
  for( int i = 0; i <descriptors_1.rows; i++)
  {
    double dist = matches[i].distance;
    if( dist < min_dist )min_dist = dist;
    if( dist > max_dist )max_dist = dist;
    
  }

  printf( "-- MAX dist : %f\n",max_dist );
  printf( "-- MIN dist : %f\n",min_dist );
  
  //当前描述子之间的距离大于两倍的最小距离时,即认为匹配有误
  //但有时候最小距离会非常小,设置一个经验值作为下限
  std::vector<DMatch> good_matches;
  for( int i = 0 ; i < descriptors_1.rows; i++)
  {
    if ( matches[i].distance <= max( 2*min_dist, 30.0 ))
      good_matches.push_back( matches[i] );

  }

  //第五步:绘制匹配结果
  Mat img_match,img_goodmatch;
  drawMatches( src_Image1, keypoint_1, src_Image2, keypoint_2, matches, img_match );
  drawMatches( src_Image1, keypoint_1, src_Image2, keypoint_2, good_matches, img_goodmatch );


  imshow( "All matching points", img_match );
  imshow( "Optimized matching point", img_goodmatch );

  waitKey( 0 );

  return 0;
}
