#include <iostream>
#include <ctime>
#include <Eigen/Core>
#include <Eigen/Dense>
using namespace std;

#define MATRIX_SIZE 50

//演示Eigen基本类型到1使用

int main ()
{
  //Eigen 以矩阵为基本数据单元。它是一个模板类。前三个参数为：数据类型，行，列
  //声明一个2×3的float矩阵
  Eigen::Matrix<float,2,3>matrix_23;
  //Eigen通过typedef提供许多内置类型，不过底层让仍是Eigen：：Matrix
  //例如Vector3d实质上是Eigen::Matrix<double, 3, 1>
  Eigen::Vector3d v_3d;
  //还有Matrix3d实质是上Eigen::Matrixe<double, 3,3>
  Eigen::Matrix3d matrix_33 = Eigen::Matrix3d::Zero();
  //如果不确认矩阵的大小， 可以使用动态数组大小
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic > matrix_dynamic;
  //跟简单类型
  Eigen::MatrixXd matrix_x;

  //输入数据
  matrix_23 << 1, 2, 3, 4, 5, 6;
  //输出
  cout << matrix_23 << endl;

  //用（）访问矩阵中欧给你的元素
  for (int i = 0; i < 1; i++)
    for(int j = 0; j < 2; j++)
      cout << matrix_23(i,j)<<endl;

  v_3d << 3, 2, 1;
  //矩阵和向量相乘
  //但是在这里不能混合来两种不同不懂类型的矩阵，像这样是错的
  //Eigen::Matrix<double, 2, 1>result_wrong_type = matrix_23 * v_3d
  
  Eigen::Matrix<double, 2, 1> result = matrix_23.cast<double>() * v_3d;
  cout << result << endl;

  //一些矩阵的运算
  matrix_33 = Eigen::Matrix3d::Random();
  cout << matrix_33 <<endl<<endl;

  cout << matrix_33.transpose()<<endl;//矩阵的转置
  cout << matrix_33.sum()<<endl;//矩阵的各元素的和
  cout << matrix_33.trace()<<endl;//矩阵的迹
  cout << 10*matrix_33 <<endl;//数乘
  cout << matrix_33.inverse()<<endl;//矩阵的逆
  cout << matrix_33.determinant()<<endl;//矩阵的行列式

  //矩阵的特征值
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigen_solver( matrix_33 );
  cout << "Eigen values = " << eigen_solver.eigenvalues()<<endl;
  cout << "Eigen vectors = " << eigen_solver.eigenvectors()<<endl; 
//解方程
//求解matrix_NN * x = v_Vd
//N的大小在前面的宏定义，矩阵由随机数生成 
//直接求解矩阵的逆是最直接的，但是求逆运算量大 
  Eigen::Matrix< double, MATRIX_SIZE, MATRIX_SIZE>Matrix_NN;
  Matrix_NN = Eigen::MatrixXd::Random( MATRIX_SIZE, MATRIX_SIZE);
  Eigen::Matrix< double, MATRIX_SIZE, 1> v_Nd;
  v_Nd = Eigen::MatrixXd::Random( MATRIX_SIZE, 1);

  clock_t time_stt = clock();//计时
  Eigen::Matrix<double,MATRIX_SIZE,1> x= Matrix_NN.inverse()*v_Nd;
  cout << "time use in normal invers is "<<1000*(clock() - time_stt)/(double)CLOCKS_PER_SEC <<"ms"<<endl;
  
  //用矩阵分解来求解 
  time_stt = clock();
  x = Matrix_NN.colPivHouseholderQr().solve(v_Nd);
  cout << "time use in Qr compsition is "<<1000*(clock() - time_stt)/(double)CLOCKS_PER_SEC <<"ms"<<endl;

return 0;
}
