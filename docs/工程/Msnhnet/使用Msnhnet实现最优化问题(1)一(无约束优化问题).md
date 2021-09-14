### 使用Msnhnet实现最优化问题(1)一(无约束优化问题)

#### 1. 预备知识

#####  范数

$ L_1范数: \Vert{x}\Vert_1=\sum_{i=1}^n \mid{x_i}\mid $

$ L_2范数: \Vert{x}\Vert_2=\sqrt{\sum_{i=1}^n \mid{x_i}\mid }$

$ L_\infty 范数: \Vert{x}\Vert_\infty= \max{\mid x_i \mid, i\in\{1,2,3,...,n\}} $

$ L_p 范数: \Vert{x}\Vert_p= (\sum_{i=1}^n \mid{x_i}\mid^p)^{\frac{1}{p}}, p\in[1,\infty)$

$范数之间的关系: \Vert{x}\Vert_\infty \le \Vert{x}\Vert_2 \le \Vert{x}\Vert_1$

#####  梯度、Jacobian矩阵和Hessian矩阵

1.梯度: f(x)多元标量函数一阶连续可微
$$ \nabla f(x) = \left[ \frac{\partial{f}}{\partial x_1},\frac{\partial{f}}{\partial x_2}, ...,\frac{\partial{f}}{\partial x_n} \right]^T $$

2.Jacobian矩阵: f(x)多元向量函数一阶连续可微
$$ J(x) = \left[ \begin{matrix} \frac{\partial f_1}{\partial x_1}  &\dots & \frac{\partial f_1}{\partial x_n} \\ \vdots  & \ddots & \vdots \\ \frac{\partial f_m}{\partial x_1} & \dots & \frac{\partial f_m}{\partial x_n} \end{matrix}\right] $$

3.Hessian矩阵: f(x)二阶连续可微
$$ H(x) = \left[ \begin{matrix} \frac{\partial^2 f}{\partial x_1^2} & \frac{\partial^2 f}{\partial x_1\partial x_2} & \dots & \frac{\partial^2 f}{\partial x_1\partial x_n} \\ \frac{\partial^2 f}{\partial x_2 \partial x_1} & \frac{\partial^2 f}{\partial x_2^2} & \dots & \frac{\partial^2 f}{\partial x_2\partial x_n} \\ \vdots & \dots & \ddots & \vdots \\ \frac{\partial^2 f}{\partial x_n \partial x_1} & \frac{\partial^2 f}{{\partial x_n \partial x_2}} & \dots & \frac{\partial^2 f}{\partial x_n^2} \end{matrix}\right]$$

注: 二次函数$f(x)=\frac{1}{2}x^TAx+x^Tb+c$,其中$A\in R^{n\times n},b\in R^N, c \in R,对称矩阵则$：

$$ \nabla f(x)=Ax+b, \nabla^2f(x)=A $$

##### Taylor公式

如果$f(x)$在$x_k$处是一阶连续可微，令$x-x_k=\delta$,则其Maclaurin余项的一阶Taylor展开式为L:

$$f(x_k+\delta)=f(x_k)+\nabla f(x_k)^T(\delta)+O(\Vert\delta\Vert)$$

如果$f(x)$在$x_k$处是二阶连续可微，令$x-x_k=\delta$,则其Maclaurin余项的二阶Taylor展开式为:

 $$f(x_k+\delta)=f(x_k)+\nabla f(x_k)^T(\delta)+\frac{1}{2}\delta^T\nabla^2f(x_k)\delta+O(\Vert\delta\Vert)$$
 或者:
 $$f(x_k+\delta)=f(x_k)+\nabla f(x_k)^T(\delta)+\frac{1}{2}\delta^TH(x_k)f(x_k)\delta+O(\Vert\delta\Vert)$$

#### 2. 凸函数判别准则

#####  一阶判别定理

$设在开凸集F\subseteq R^n内函数f(x)一阶可微,有:$

1.f(x)在凸集F上为凸函数,则对于任意$x,y\in F$ ,有:

$f(y)−f(x) \ge \nabla f(x)^T(y−x)$

2.f(x)在凸集F上为严格凸函数,则对于任意$x,y\in F$有

$f(y)−f(x)\ge \nabla f(x)^T(y−x)$


#####  二阶判别定理

设在开凸集$F\subseteq R^n$内函数$f(x)$二阶可微,有:

1.f(x)在凸集F上为凸函数,则对于任意$x\in F$,Hessian矩阵半正定

2.f(x)在凸集F上为严格凸函数,则对于$\forall x \in F$,Hessian矩阵正定

#####  矩阵正定判定

1.若所有特征值均大于零，则称为正定

2.若所有特征值均大于等于零，则称为半正定


#### 3. 无约束优化

#####  无约束优化基本架构

$$ x_\ast=argmin f(x)$$

**step1**.给定初始点$x_0 \in R^n$, k=0以及最小误差$\xi$

**step2**.判断$x_k$是否满足终止条件，是则终止

**step3**.确定f(x)在点x_k的下降方向$d_k$

**step4**.确定下降步长$\alpha_k$, $\alpha_k>0$, 计算$f_{k+1}=f(x_k+\alpha_k d_k)$，满足 $f_{k+1} < f_k$

**step5**.令$x_{k+1}=x_k+\alpha_kd_k,k=k+1$ ; 返回step2

**终止条件**: $\Vert\nabla f(x_k)\Vert_2\le\xi\quad or\quad \Vert f_k-f_{k+1}\Vert\le\xi\quad or \quad \Vert x_k - x_{k+1}\Vert_2 \le \xi$

#### 4. 梯度下降法

$$(x_k+1)=f(x_k)−\alpha \nabla f(x_k)$$

$\alpha$在梯度下降算法中被称作为学习率或者步长;$\nabla f(x_k)$梯度的方向

梯度下降不一定能够找到全局最优解，有可能是局部最优解。当然，如果损失函数是凸函数，梯度下降法得到的解就一定是全局最优解。

$step1.给定初始点x_0 \in R^n, k=0,学习率\alpha和最小误差\xi ;$

$step2.判断x_k是否满足终止条件，是则终止;$

$step3.确定f(x)在点x_k的下降方向d_k = -\Delta f(x_k);$

$step5.令x_{k+1}=x_k+\alpha d_k,k=k+1;返回step2$

##### 举例

**$y = 3x_1^2+3x_2^2-x_1^2+x_2,初始点(1.5,1.5),\xi=10^{-3}$**

```c++
#include <Msnhnet/math/MsnhMatrixS.h>
#include <Msnhnet/cv/MsnhCVGui.h>
#include <iostream>

using namespace Msnhnet;

class SteepestDescent
{
public:
    SteepestDescent(double learningRate, int maxIter, double eps):_learningRate(learningRate),_maxIter(maxIter),_eps(eps){}

    void setLearningRate(double learningRate)
    {
        _learningRate = learningRate;
    }

    void setMaxIter(int maxIter)
    {
        _maxIter = maxIter;
    }

    virtual int solve(MatSDS &startPoint) = 0;

    void setEps(double eps)
    {
        _eps = eps;
    }

    const std::vector<Vec2F32> &getXStep() const
    {
        return _xStep;
    }

protected:
    double _learningRate = 0;
    int _maxIter = 100;
    double _eps = 0.00001;
    std::vector<Vec2F32> _xStep;
protected:
    virtual MatSDS calGradient(const MatSDS& point) = 0;
    virtual MatSDS function(const MatSDS& point) = 0;
};


class NewtonProblem1:public SteepestDescent
{
public:
    NewtonProblem1(double learningRate, int maxIter, double eps):SteepestDescent(learningRate, maxIter, eps){}

    MatSDS calGradient(const MatSDS &point) override
    {
        MatSDS dk(1,2);
        // df(x) = (2x_1,2x_2)^T
        double x1 = point(0,0);
        double x2 = point(0,1);

        dk(0,0) = 6*x1 - 2*x1*x2;
        dk(0,1) = 6*x2 - x1*x1;

        dk = -1*dk;
        return dk;
    }

    MatSDS function(const MatSDS &point) override
    {
        MatSDS f(1,1);
        double x1 = point(0,0);
        double x2 = point(0,1);

        f(0,0) = 3*x1*x1 + 3*x2*x2 - x1*x1*x2;

        return f;
    }

    int solve(MatSDS &startPoint) override
    {
        MatSDS x = startPoint;
        for (int i = 0; i < _maxIter; ++i)
        {
            _xStep.push_back({(float)x[0],(float)x[1]});
            MatSDS dk = calGradient(x);

            std::cout<<std::left<<"Iter(s): "<<std::setw(4)<<i<<", Loss: "<<std::setw(12)<<dk.L2()<<" Result: "<<function(x)[0]<<std::endl;

            if(dk.L2() < _eps)
            {
                startPoint = x;
                return i+1;
            }
            x = x + _learningRate*dk;
        }

        return -1;
    }
};



int main()
{
    NewtonProblem1 function(0.1, 100, 0.001);
    MatSDS startPoint(1,2,{1.5,1.5});
    int res = function.solve(startPoint);
    if(res < 0)
    {
        std::cout<<"求解失败"<<std::endl;
    }
    else
    {
        std::cout<<"求解成功! 迭代次数: "<<res<<std::endl;
        std::cout<<"最小值点："<<res<<std::endl;
        startPoint.print();

        std::cout<<"此时方程的值为："<<std::endl;
        function.function(startPoint).print();

#ifdef WIN32
        Gui::setFont("c:/windows/fonts/MSYH.TTC",16);
#endif
        std::cout<<"按\"esc\"退出!"<<std::endl;
        Gui::plotLine(u8"最速梯度下降法迭代X中间值","x",function.getXStep());
        Gui::wait();
    }


}

```

结果:迭代次数13次，求得最小值点

![迭代次数13次，求得最小值点](https://img-blog.csdnimg.cn/20210702132449358.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70#pic_center)


![SGD迭代过程中对X进行可视化](https://img-blog.csdnimg.cn/20210702132443409.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70#pic_center)


#### 4. 牛顿法

梯度下降法初始点选取问题, 会导致迭代次数过多, 可使用牛顿法可以处理.

![牛顿法和SGD可视化比较](https://img-blog.csdnimg.cn/20210702132406434.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70#pic_center)

目标函数$argmin f({x})$在${x}_k$处进行二阶泰勒展开:

$$f({x}_k+{d}_k)\approx f({x}_k)+J({x}_k)^T{d}_k+ \frac{1}{2}{d}_k^TH({x}_k){d}_k$$

目标函数变为:

$$argmin(f({x}_k)+J({x}_k)^T\Delta x+\frac{1}{2}{d}_k^TH({x}_k){d}_k)$$

关于${d}_k$求导,并让其为0,可以得到步长:

$${d}_k= -H({x}_k)^{-1}J{x}_k$$

**与梯度下降法比较，牛顿法的好处：**

A点的Jacobian和B点的Jacobian值差不多, 但是A点的Hessian矩阵较大, 步长比较小, B点的Hessian矩阵较小,步长较大, 这个是比较合理的.如果是梯度下降法,则梯度相同, 步长也一样,很显然牛顿法要好得多. 弊端就是Hessian矩阵计算量非常大.

![牛顿法和梯度下降法的优缺点](https://img-blog.csdnimg.cn/20210702132420751.png#pic_center)

###### 步骤

**step1**.给定初始点${x_0} \in R^n$, $k=0$,学习率$\alpha$和最小误差$\xi$

**step2**.判断${x_k}$是否满足终止条件，是则终止

**step3**.确定$f({x})$在点${x_0}$的下降方向$d_k = -H({x}_k)^{-1}J{x}_k$

**step4**.令${x}_{k+1}={x}_k+\alpha {d}_k,k=k+1$ 返回**step2**

##### 举例

**$y = 3x_1^2+3x_2^2-x_1^2+x_2,初始点(1.5,1.5)和(0,-3),\xi=10^{-3}$**

```c++
#include <Msnhnet/math/MsnhMatrixS.h>
#include <iostream>
#include <Msnhnet/cv/MsnhCVGui.h>

using namespace Msnhnet;

class Newton
{
public:
    Newton(int maxIter, double eps):_maxIter(maxIter),_eps(eps){}

    void setMaxIter(int maxIter)
    {
        _maxIter = maxIter;
    }

    virtual int solve(MatSDS &startPoint) = 0;

    void setEps(double eps)
    {
        _eps = eps;
    }

    //正定性判定
    bool isPosMat(const MatSDS &H)
    {
         MatSDS eigen = H.eigen()[0];
         for (int i = 0; i < eigen.mWidth; ++i)
         {
            if(eigen[i]<=0)
            {
                return false;
            }
         }

         return true;
    }

    const std::vector<Vec2F32> &getXStep() const
    {
        return _xStep;
    }

protected:
    int _maxIter = 100;
    double _eps = 0.00001;
    std::vector<Vec2F32> _xStep;

protected:
    virtual MatSDS calGradient(const MatSDS& point) = 0;
    virtual MatSDS calHessian(const MatSDS& point) = 0;
    virtual bool calDk(const MatSDS& point, MatSDS &dk) = 0;
    virtual MatSDS function(const MatSDS& point) = 0;
};


class NewtonProblem1:public Newton
{
public:
    NewtonProblem1(int maxIter, double eps):Newton(maxIter, eps){}

    MatSDS calGradient(const MatSDS &point) override
    {
        MatSDS J(1,2);
        double x1 = point(0,0);
        double x2 = point(0,1);

        J(0,0) = 6*x1 - 2*x1*x2;
        J(0,1) = 6*x2 - x1*x1;

        return J;
    }

    MatSDS calHessian(const MatSDS &point) override
    {
        MatSDS H(2,2);
        double x1 = point(0,0);
        double x2 = point(0,1);

        H(0,0) = 6 - 2*x2;
        H(0,1) = -2*x1;
        H(1,0) = -2*x1;
        H(1,1) = 6;

        return H;
    }


    bool calDk(const MatSDS& point, MatSDS &dk) override
    {
        MatSDS J = calGradient(point);
        MatSDS H = calHessian(point);
        if(!isPosMat(H))
        {
            return false;
        }
        dk = -1*H.invert()*J;
        return true;
    }

    MatSDS function(const MatSDS &point) override
    {
        MatSDS f(1,1);
        double x1 = point(0,0);
        double x2 = point(0,1);

        f(0,0) = 3*x1*x1 + 3*x2*x2 - x1*x1*x2;

        return f;
    }

    int solve(MatSDS &startPoint) override
    {
        MatSDS x = startPoint;
        for (int i = 0; i < _maxIter; ++i)
        {

            _xStep.push_back({(float)x[0],(float)x[1]});

            MatSDS dk;

            bool ok = calDk(x, dk);

            if(!ok)
            {
                return -2;
            }

            x = x + dk;

            std::cout<<std::left<<"Iter(s): "<<std::setw(4)<<i<<", Loss: "<<std::setw(12)<<dk.L2()<<" Result: "<<function(x)[0]<<std::endl;

            if(dk.LInf() < _eps)
            {
                startPoint = x;
                return i+1;
            }
        }

        return -1;
    }
};



int main()
{
    NewtonProblem1 function(100, 0.01);
    MatSDS startPoint(1,2,{1.5,1.5});

    try
    {
        int res = function.solve(startPoint);
        if(res == -1)
        {
            std::cout<<"求解失败"<<std::endl;
        }
        else if(res == -2)
        {
            std::cout<<"Hessian 矩阵非正定, 求解失败"<<std::endl;
        }
        else
        {
            std::cout<<"求解成功! 迭代次数: "<<res<<std::endl;
            std::cout<<"最小值点："<<res<<std::endl;
            startPoint.print();

            std::cout<<"此时方程的值为："<<std::endl;
            function.function(startPoint).print();

#ifdef WIN32
        Gui::setFont("c:/windows/fonts/MSYH.TTC",16);
#endif
        std::cout<<"按\"esc\"退出!"<<std::endl;
        Gui::plotLine(u8"牛顿法迭代X中间值","x",function.getXStep());
        Gui::wait();
        }

    }
    catch(Exception ex)
    {
        std::cout<<ex.what();
    }

}

```

结果:对于初始点 **(1.5,1.5)** 迭代次数6次，求得最小值点,迭代次数比梯度下降法少了一半

![牛顿法求解最小值，只需要迭代六次](https://img-blog.csdnimg.cn/20210702132435996.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70#pic_center)

![牛顿法迭代过程中，X的可视化结果，可以看到这里X迈的步子是很大的](https://img-blog.csdnimg.cn/20210702132427759.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70#pic_center)

结果:二对于初始点 **(0,3)**, 由于在求解过程中会出现hessian矩阵非正定的情况，故需要对newton法进行改进.（这个请看下期文章）

![求解失败](https://img-blog.csdnimg.cn/20210702132413932.png#pic_center)

#### 5. 源码

https://github.com/msnh2012/numerical-optimizaiton(`https://github.com/msnh2012/numerical-optimizaiton`)

#### 6. 依赖包

https://github.com/msnh2012/Msnhnet(`https://github.com/msnh2012/Msnhnet`)


#### 7. 参考文献

1. Numerical Optimization. Jorge Nocedal Stephen J. Wrigh
2. Methods for non-linear least squares problems. K. Madsen, H.B. Nielsen, O. Tingleff.
3. Practical Optimization_ Algorithms and Engineering Applications. Andreas Antoniou Wu-Sheng Lu
4. 最优化理论与算法. 陈宝林
5. 数值最优化方法. 高立

网盘资料下载：链接：https://pan.baidu.com/s/1hpFwtwbez4mgT3ccJp33kQ 提取码：b6gq 


#### 8. 最后

- 欢迎关注我们维护的一个深度学习框架Msnhnet:
- https://github.com/msnh2012/Msnhnet
  Msnhnet除了是一个深度网络推理库之外，还是一个小型矩阵库，包含了矩阵常规操作，LU分解，Cholesky分解，SVD分解。


-----------------------------------------------------------------------------------------------

欢迎关注GiantPandaCV, 在这里你将看到独家的深度学习分享，坚持原创，每天分享我们学习到的新鲜知识。( • ̀ω•́ )✧

有对文章相关的问题，或者想要加入交流群，欢迎添加BBuf微信：

![二维码](https://img-blog.csdnimg.cn/20200110234905879.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)