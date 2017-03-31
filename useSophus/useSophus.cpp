#include <iostream>
#include <cmath>
using namespace std;

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "sophus/so3.h"
#include "sophus/se3.h"

int main( int argc, char** argv )
{
  //沿Ｚ轴旋转９０度的旋转矩阵
  Eigen::Matrix3d R = Eigen::AngleAxisd ( M_PI/2, Eigen::Vector3d( 0,0,1 )).toRotationMatrix();

  //Sophus::SO(2)可以直接从旋转矩阵构造
  Sophus::SO3 SO3_R(R);
  Sophus::SO3 SO3_v( 0, 0, M_PI/2);//亦可从旋转向量构造
  Eigen::Quaterniond q( q );
  Sophus::SO3 SO3_q( q );
  //上述表达式都是等价的
  //输出SO(３)时，以so(3)形式输出
  std::cout << "SO(3) from matrix: "<<SO3_R << std::endl;
  std::cout << "SO(3) from vector:  "<<SO3_v << std::endl;
  std::cout << "SO(3) from quaternion:"<<SO3_q << std::endl;

  //使用对数映射获得它的李代数
  Eigen::Vector3d so3 = SO3_R.log();
  std::cout << "so3 = "<<so3.transpose() << std::endl;
  //hat为向量到反对称矩阵
  std::cout << "so3 hat="<<Sophus::SO3::hat(so3) << std::endl;
  //相对的，vee为反对称到向量
  std::cout << "so3 hat vee= "<<Sophus::SO3::vee( Sophus::SO3::hat(so3)).transpose() << std::endl;

  //增量扰动模型的更新
  Eigen::Vector3d update_so3( 1e-4, 0, 0);//假设更新有这么多
  Sophus::SO3 SO3_updated = Sophus::SO3::exp( update_so3 )*SO3_R;//左乘更新
  std::cout << "SO3 updated =  "<< SO3_updated << std::endl;
  std::cout << "*******************" << std::endl;


  //对SE(3)操作大同小异
  Eigen::Vector3d t(1, 0, 0 );
  Sophus::SE3 SE3_Rt( R, t);
  Sophus::SE3 SE3_qt( q, t );
  std::cout << "SE3 from R,t = "<<endl<< SE3_Rt<< std::endl;
  //李代数se(3)是一个六维向量，方便起见先typedef一下
  typedef Eigen::Matrix<double, 6, 1>Vector6d;
  Vector6d se3 = SE3_Rt.log();
  std::cout << " se3 = "<<se3.transpose() << std::endl;
  //观察输出，会发现Sophus中，se(3)平移在前，旋转在后，
  //同样的，有hat和vee两个算符
  std::cout << "se3 hat = "<<endl<< Sophus::SE3::hat(se3) << std::endl;
  //演示更新
  Vector6d update_se3;//更新量
  update_se3.setZero();
  update_se3(0,0) = 1e-4d;
  Sophus::SE3 SE3_updated = Sophus::SE3::exp( update_se3)*SE3_Rt;
  std::cout << "SE3 updated = "<<endl<<SE3_updated.matrix() << std::endl;

  return 0;

}
