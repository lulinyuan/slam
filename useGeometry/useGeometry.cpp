#include <iostream>
#include <cmath>
#include <Eigen/Core>
#include <Eigen/Geometry>

using namespace std ;

int main ()
{
  //Eigen/Geometry模块提供各种各种旋转和平移表示
  //3D旋转矩阵直接使用Matric3d或Matrix3f
  Eigen::Matrix3d rotation_matrix = Eigen::Matrix3d::Identity();
  //旋转向量使用AangleAxis，它底层不直接是Matrix，但运算可以当作矩阵（因为重载了运算）
  Eigen:: AngleAxisd rotation_vector( M_PI/4, Eigen::Vector3d( 0, 0, 1));//沿z轴旋转45度
  cout .precision(3);
  //用matrix()转化成矩阵
  cout <<"rotation_matrix =\n "<<rotation_vector.matrix() <<endl;
  //也可以赋值
  rotation_matrix = rotation_vector.toRotationMatrix();
  //用AngleAxis 可以进行坐标变换
  Eigen::Vector3d v( 1, 0, 0);
  Eigen::Vector3d v_rotated = rotation_vector * v;
  cout <<"( 1, 0, 0) after rotation ="<< v_rotated.transpose()<<endl;
  //使用旋转矩阵
  v_rotated = rotation_matrix * v;
  cout <<"( 1, 0, 0) after rotation = "<< v_rotated.transpose()<<endl;


  //欧拉角：可以将旋转矩阵直接转化成欧拉角
  Eigen::Vector3d euler_angles = rotation_matrix.eulerAngles( 2, 1, 0);//ZXY顺序即yaw pitch roll 顺序
  cout <<"yaw pitch roll = " << euler_angles.transpose() <<endl;


  //欧式变换使用Eigen ::Isometry
  Eigen::Isometry3d T = Eigen::Isometry3d::Identity();//虽然称为3d，实质上是4×4的矩阵
  T.rotate( rotation_vector );// 按照rotation进行旋转
  T.pretranslate ( Eigen::Vector3d ( 1, 3, 4)); //把平移向量设成（ 1， 3, 4 ）
  cout << "Transform matrix = \n "<< T.matrix() << endl;

  //用变换矩阵进行坐标变换
  Eigen::Vector3d v_transformed = T * v ;//相当于R*v+t
  cout << "Transformed = " << v_transformed.transpose()<< endl;
  
  //对于仿射和射影变换，使用Eigen::Affine3d 和 Eigen::Projective3d即可
  
  //四元数
  //可以直接把AngleAxis赋值给四元数，反之亦然
  Eigen::Quaterniond q = Eigen::Quaterniond( rotation_vector );
  cout << "quaternion = \n " << q.coeffs() << endl;//注意coeffs 的顺序是（ x， y， z， w）W为实部，前三者为虚部
  //也可以把旋转矩阵赋给它
  q = Eigen::Quaterniond ( rotation_matrix );
  cout << "quaternion = \n" << q.coeffs() <<endl;
  //使用四元数旋转一个向量， 使用重载到1乘法即可
  v_rotated = q * v;
  cout << " ( 1, 0, 0) after rotation = "<< v_rotated.transpose()<<endl;


  return 0;

;
}
