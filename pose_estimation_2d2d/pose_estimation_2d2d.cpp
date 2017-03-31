#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>

using namespace std;
using namespace cv;


/***********************
 * 本程序主要实现2D-2D的特征匹配估计相机运动
 * ***************************/
//全局函数声明部分
//进行图片的特征点匹配
void find_feature_matches(
  const Mat& srcImage_1,const Mat& srcImage_2,
  std::vector<KeyPoint>& Keypoints_1,
  std::vector<KeyPoint>& Keypoints_2,
  std::vector<DMatch>& matches );


//2D-2D图像坐标转换
void pose_estimation_2d2d(
  std::vector<KeyPoint> Keypoints_1,
  std::vector<KeyPoint> Keypoints_2,
  std::vector<DMatch> matches,
  Mat& R,Mat& t );

//像素坐标转相机归一化坐标
Point2d pixel2cam( const Point2d& p, const Mat& K );

int main(int argc, char** argv)
{
  if( argc != 3 )
  {
    cout<<" usage: pose_estimation_2d2d srcImage_1 srcImage_2 "<<endl;
    return 1; 
  } 


  //--读取图像
  Mat srcImage_1 = imread( argv[1], CV_LOAD_IMAGE_COLOR );
  Mat srcImage_2 = imread( argv[2], CV_LOAD_IMAGE_COLOR );

  vector<KeyPoint> Keypoints_1, Keypoints_2;
  vector<DMatch> matches;
  find_feature_matches( srcImage_1, srcImage_2, Keypoints_1, Keypoints_2, matches );
  cout<<"一共找到了"<<matches.size()<<"组匹配点"<<endl;

  //-- 估计两张图想之间的运动
  Mat R, t;
  pose_estimation_2d2d( Keypoints_1, Keypoints_2, matches, R, t );

  //-- 验证E=t^Rscale
  Mat t_x =( Mat_<double>( 3,3 )<<
            0,                     -t.at<double>( 2,0 ),   t.at<double>( 1,0 ),
            t.at<double>( 2,0 ),   0,                      -t.at<double>( 0,0 ),
            -t.at<double>( 1,0 ),  t.at<double>( 0,0 ),    0 );
  cout<<"t^R="<<endl<<t_x*R<<endl;

  //-- 验证对极约束
  Mat K = ( Mat_<double>( 3,3 )<< 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1 );
  for( DMatch m: matches )
  {
    Point2d pt1 = pixel2cam( Keypoints_1[ m.queryIdx ].pt, K );
    Mat y1 = ( Mat_<double> ( 3,1 )<< pt1.x, pt1.y, 1 );
    Point2d pt2 = pixel2cam( Keypoints_2[ m.trainIdx ].pt, K );
    Mat y2 = ( Mat_<double> ( 3,1 )<< pt2.x, pt2.y, 1 );
    Mat d = y2.t() * t_x * R * y1;
    cout<<"epipolar constraint = "<< d <<endl;
     
  }
  return 0;
}


void find_feature_matches(const Mat& srcImage_1,const Mat& srcImage_2,
                          std::vector<KeyPoint>& Keypoints_1,
                          std::vector<KeyPoint>& Keypoints_2,
                          std::vector<DMatch>& matches )
{
    //初始化
    Mat descripitors_1, descripitors_2;
   //used in OpenCV3
   Ptr<FeatureDetector> detector = ORB::create();
   Ptr<DescriptorExtractor> descripitor = ORB::create();
   Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create( "BruteForce-Hamming" );

   //-- 第一步:检测Oriented FAST角点检测
   detector -> detect ( srcImage_1, Keypoints_1 );
   detector -> detect ( srcImage_2, Keypoints_2 );

   //--第二步:根据角点位置计算BRIEF描述子
   descripitor -> compute( srcImage_1, Keypoints_1, descripitors_1 );
   descripitor -> compute( srcImage_2, Keypoints_2, descripitors_2 );

   //--对两幅图中的BRIEF描述子进行匹配,使用Hamming距离
   vector<DMatch> match;
   matcher -> match ( descripitors_1, descripitors_2, match );

   //匹配点对筛选
   double min_dist = 10000, max_dist = 0;
   
   //找出所有匹配之间的最小距离和最大距离,即是最相似的和最不相似的两组点之间的距离
   for( int i =0; i < descripitors_1.rows; i++ )
   {
      double dist = match[i].distance;
      if( dist < min_dist )min_dist = dist;
      if( dist > max_dist )max_dist = dist;
      
   }
   printf( "-- Max dist : %f \n",max_dist );
   printf( "-- Min dist :%f \n", min_dist );

   //当描述子之间距离大于两倍的最小距离时,即认为匹配有误,但有时候最小距离会非常小,设置一个经验值30作为下限
   for( int i = 0; i < descripitors_1.rows; i++ )
   {
      if( match[i].distance <= max( 2*min_dist, 30.0 ) )
        matches.push_back( match[i] );
   }

}
Point2d pixel2cam( const Point2d& p, const Mat& K )
  {
    return Point2f
     (
       ( p.x - K.at<double> ( 0,2 ) )/ K.at<double>( 0,0 ),
       ( p.y - K.at<double> ( 1,2 ) )/ K.at<double>( 1,1 )
       ); 
  }
void pose_estimation_2d2d(
                         std::vector<KeyPoint> Keypoints_1,
                         std::vector<KeyPoint> Keypoints_2,
                         std::vector<DMatch> matches,
                         Mat& R,Mat& t )
{
  //相机内参,TUMFreiburg2
  Mat K = ( Mat_<double> ( 3, 3 ) << 520,9, 0, 325.1, 0, 521, 249.7, 0, 0, 1 );

  //--把匹配点转换为vector<Point2f>的形式
  vector<Point2f> points_1;
  vector<Point2f> points_2;

  for( int i = 0; i < (int)matches.size(); i++)
  {
    points_1.push_back( Keypoints_1[matches[i].queryIdx].pt);
    points_2.push_back( Keypoints_2[matches[i].trainIdx].pt);
  
  }

  //计算基础矩阵
  Mat fundamental_matrix;
  fundamental_matrix = findFundamentalMat( points_1, points_2, CV_FM_8POINT );
  cout<<"fundamental_matrix is "<<endl<< fundamental_matrix<<endl;

  //计算本质矩阵
  Point2d principal_point ( 325.1,249.7 );//相机光心,TUM dataset标定值
  double focal_length = 521;
  Mat essential_matrix;
  essential_matrix = findEssentialMat( points_1, points_2, focal_length, principal_point );
  cout<<"essential_matrix is "<<endl<<essential_matrix<<endl;

  //计算单应矩阵
  Mat homography_matrix;
  homography_matrix = findHomography ( points_1, points_2, RANSAC, 3 );
  cout<<"homography_matrix is "<<endl<<homography_matrix<<endl;

  //-- 从本质矩阵中恢复旋转和平移信息
  recoverPose( essential_matrix, points_1, points_2, R, t, focal_length, principal_point );
  cout<<"R is "<<endl<<R<<endl;
  cout<<"t is "<<endl<<t<<endl;
}
