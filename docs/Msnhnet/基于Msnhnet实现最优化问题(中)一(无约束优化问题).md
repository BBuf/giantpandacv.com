> 接上文：[基于Msnhnet实现最优化问题（上）SGD&&牛顿法](https://mp.weixin.qq.com/s/Ty2GiGifD2AReceDQQykCQ)
#### 1. 阻尼牛顿法

牛顿法最突出的优点是收敛速度快，具有局部二阶收敛性，但是，基本牛顿法初始点需要足够“靠近”极小点，否则，有可能导致算法不收敛。

这样就引入了阻尼牛顿法，阻尼牛顿法最核心的一点在于可以修改每次迭代的步长，通过沿着牛顿法确定的方向**一维搜索**最优的步长，最终选择使得函数值最小的步长。

**补充：一维搜索非精确搜索方法。**

1.Armijo条件(控制步长太大)

$$
f({x}_k +\alpha_k{d}_k)≤f({x}_k )+\rho\alpha_kf’({x}_k )^T{d}_k,\rho \in (0,0.5)
$$

满足Armijo条件的点为$[0,\beta_1]$和$[\beta_2,\beta_3]$区间的点.

![Armijo条件](https://img-blog.csdnimg.cn/img_convert/1dc051b03a57260f4501be09788b57ca.png#pic_center)


2.Goldstein准则(控制步长太小)

$$
f({x}_k +\alpha_k{d}_k)≤f({x}_k )+\rho\alpha_kf’({x}_k )^T{d}_k\\
f({x}_k +\alpha_k{d}_k)≥f({x}_k )+(1−\rho)\alpha_kf’({x}_k )^T{d}_k\\
\rho \in (0,0.5)
$$

满足Goldstein准则的点为$[\beta_7,\beta_4]$和$[\beta_3,\beta_6]$区间的点.

![Goldstein准则](https://img-blog.csdnimg.cn/img_convert/610bfb230930b2ce4a4f6bd7e15f6e6f.png#pic_center)


3.Wolfe准则

$$
f({x}_k +\alpha_k{d}_k)≤f({x}_k )+\rho\alpha_kf’({x}_k )^T{d}_k\\
f’({x}_k +\alpha_k{d}_k)^T{d}_k≥\sigma f’({x}_k )^T{d}_k\\
1 > \sigma >\rho > 0
$$

满足Wolfe准则的点为$[\beta_7,\beta_4]$,$[\beta_8,\beta_9]$和$[\beta_{10},\beta_6]$区间的点.

![Wolfe准则](https://img-blog.csdnimg.cn/img_convert/108418995e6ab683fab32e44e5d62697.png#pic_center)


**补充：一维搜索非精确搜索方法一般步骤(以Armijo为例)。**

$While\quad f({x}_k +\alpha_k{d}_k)>f({x}_k )+\rho\alpha_kf’({x}_k )^T{d}_k$
$\quad \alpha_k=\tau^m_k,m_k=m_k+1, \tau \in(0,1)$
$End$

![阻尼牛顿法步骤](https://img-blog.csdnimg.cn/img_convert/ad251558573bfdc42d9f4db3ed1e33c4.png)

##### 举例

$y = 3x_1^2+3x_2^2-x_1^2+x_2$,初始点$(1.5,1.5)$和$(0,3),\xi=10^{-3}$

```c++
#include <Msnhnet/math/MsnhMatrixS.h>
#include <Msnhnet/cv/MsnhCVGui.h>
#include <iostream>

using namespace Msnhnet;

class DampedNewton
{
public:
    DampedNewton(int maxIter, double eps, double rho, double tau):_maxIter(maxIter),_eps(eps),_rho(rho),_tau(tau){}


    void setMaxIter(int maxIter)
    {
        _maxIter = maxIter;
    }

    virtual int solve(MatSDS &startPoint) = 0;

    void setEps(double eps)
    {
        _eps = eps;
    }

    void setRho(double rho)
    {
        _rho = rho;
    }

    void setTau(double tau)
    {
        _tau = tau;
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
    double _rho = 0.2;
    double _tau = 0.9;
    std::vector<Vec2F32> _xStep;
protected:
    virtual MatSDS calGradient(const MatSDS& point) = 0;
    virtual MatSDS calHessian(const MatSDS& point) = 0;
    virtual bool calDk(const MatSDS& point, MatSDS &dk) = 0;
    virtual MatSDS function(const MatSDS& point) = 0;
};


class DampedNewtonProblem1:public DampedNewton
{
public:
    DampedNewtonProblem1(int maxIter, double eps, double rho, double tau):DampedNewton(maxIter, eps, rho, tau){}

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

            double alpha = 1;

            //Armijo准则
            for (int i = 0; i < 100; ++i)
            {
                MatSDS left  = function(x + alpha*dk);
                MatSDS right = function(x) + this->_rho*alpha*calGradient(x).transpose()*dk;

                if(left(0,0) <= right(0,0))
                {
                    break;
                }

                alpha = alpha * _tau;
            }

            std::cout<<std::left<<"Iter(s): "<<std::setw(4)<<i<<", Loss: "<<std::setw(12)<<dk.L2()<<" Result: "<<function(x)[0]<<std::endl;

            x = x + alpha*dk;

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
    DampedNewtonProblem1 function(100, 0.001, 0.4, 0.8);
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
        Gui::plotLine(u8"阻尼牛顿法迭代X中间值","x",function.getXStep());
        Gui::wait();

        }

    }
    catch(Exception ex)
    {
        std::cout<<ex.what();
    }
}

```

结果:对于初始点 **(1.5,1.5)** ,迭代4次即可完成

![阻尼牛顿法,4次完成迭代](https://img-blog.csdnimg.cn/img_convert/befe974451b7003d6fedc5c5d9f6b659.png#pic_center)


![阻尼牛顿法X中间点可视化](https://img-blog.csdnimg.cn/img_convert/9e0f6c921762ce79b6284d981d3f7353.png#pic_center)


结果:对于初始点 **(0,3)** ,同样是Hessian矩阵不正定.

![阻尼牛顿法求解失败](https://img-blog.csdnimg.cn/img_convert/5b86e6c8b7b33eaf07c5c06097fa7726.png#pic_center)



#### 2. 牛顿Levenberg-Marquardt法

LM(Levenberg-Marquardt)法是处理Hessian矩阵$H$奇异、不正定等情形的一个最简单有效的方法，求解${d}_k$公式变为：

$$
{d}_k=−(H({x}_k)+{\color{red}{v_kI}})^{−1}J({x}_k)
$$

式中：

$v_k>0,I$为单位阵,如果$(H({x}_k)+v_kI)$还不正定,可取$v_k=2v_k$

**步骤**

$step1.$给定初始点${x}_0\in R^n,v_k>1,k=0,\rho\in(0,0.5), \tau\in(0,1)$,以及最小误差$\xi;$

$step2.$判断${x}_k$是否满足终止条件,是则终止;

$step3.$求解$H_{new}({x}_k)=H({x}_k)+v_k∗I;$

$step4.$判定$H_{new}({x}_k)$正定性,如果非正定,令$H_{new}({x}_k)=H({x}_k)+2∗v_k∗I;$

$step5.$确定$f(x)$在${x}_k$点的下降方向${d}_k=−H_{new}({x}_k)^{−1}J({x}_k);$

$step6.$计算$\alpha _k=\tau^m,m=m+1;$

$step7.$如果$f({x}_k+\alpha _k{d}_k)>f({x}_k )+\rho\alpha _kJ({x}_k)^T{d}_k$,返回$step4$,否则继续;

$step8. $令${x}_k+1={x}_k+\alpha _k{d}_k,k=k+1$;返回$step2.$

##### 举例

$y = 3x_1^2+3x_2^2-x_1^2+x_2,$初始点(0,3),$\xi=10^{-3}$

```c++
#include <Msnhnet/math/MsnhMatrixS.h>
#include <Msnhnet/cv/MsnhCVGui.h>
#include <iostream>

using namespace Msnhnet;

class NewtonLM
{
public:
    NewtonLM(int maxIter, double eps, double vk, double rho, double tau):_maxIter(maxIter),_eps(eps),_vk(vk),_rho(rho),_tau(tau){}


    void setMaxIter(int maxIter)
    {
        _maxIter = maxIter;
    }

    virtual int solve(MatSDS &startPoint) = 0;

    void setEps(double eps)
    {
        _eps = eps;
    }

    void setRho(double rho)
    {
        _rho = rho;
    }

    void setTau(double tau)
    {
        _tau = tau;
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
    double _vk  = 3;
    double _rho = 0.2;
    double _tau = 0.9;
    std::vector<Vec2F32> _xStep;
protected:
    virtual MatSDS calGradient(const MatSDS& point) = 0;
    virtual MatSDS calHessian(const MatSDS& point) = 0;
    virtual MatSDS calDk(const MatSDS& point) = 0;
    virtual MatSDS function(const MatSDS& point) = 0;
};


class NewtonLMProblem1:public NewtonLM
{
public:
    NewtonLMProblem1(int maxIter, double eps, double vk, double rho, double tau):NewtonLM(maxIter, eps, vk,rho,tau){}

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


    MatSDS calDk(const MatSDS& point) override
    {
        MatSDS J = calGradient(point);
        MatSDS H = calHessian(point);

        MatSDS I = MatSDS::eye(H.mWidth);


        MatSDS Hp  = H + _vk*I;

        if(!isPosMat(Hp))
        {
            H = H + 2*_vk*I;
        }
        else
        {
            H = Hp;
        }

        return -1*H.invert()*J;
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
            //这里就不用检查正定了
            MatSDS dk = calDk(x);

            double alpha = 1;

            //Armijo准则
            for (int i = 0; i < 100; ++i)
            {
                MatSDS left  = function(x + alpha*dk);
                MatSDS right = function(x) + this->_rho*alpha*calGradient(x).transpose()*dk;

                if(left(0,0) <= right(0,0))
                {
                    break;
                }

                alpha = alpha * _tau;
            }

            std::cout<<std::left<<"Iter(s): "<<std::setw(4)<<i<<", Loss: "<<std::setw(12)<<dk.L2()<<" Result: "<<function(x)[0]<<std::endl;

            x = x + alpha*dk;

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
    NewtonLMProblem1 function(1000, 0.001,3, 0.4, 0.8);
    MatSDS startPoint(1,2,{0,3});

    try
    {
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
        Gui::plotLine(u8"牛顿LM法迭代X中间值","x",function.getXStep());
        Gui::wait();
        }

    }
    catch(Exception ex)
    {
        std::cout<<ex.what();
    }
}

```

结果: 对于初始点 **(0,3)** ,迭代8次即可完成,解决了Newton法Hessian矩阵不正定的问题.

![牛顿LM法8次迭代求解成功](https://img-blog.csdnimg.cn/img_convert/20db3449c899c1b826582465167e8e74.png#pic_center)

![牛顿LM法X中间点可视化](https://img-blog.csdnimg.cn/img_convert/1be50cfdca4fa7c094985fa427d19643.png#pic_center)


#### 3.拟牛顿法

牛顿法虽然收敛速度快,但是计算过程中需要计算目标函数的Hassian矩阵,有时候Hassian矩阵不能保持正定从而导致牛顿法失效.从而提出拟牛顿法.

**思路:**

通过用不含二阶导数的矩阵$U$代替牛顿法中的$H^{−1}$,然后沿着$−UJ$的方向做一维搜索.不同的构建$U$的方法有不同的拟牛顿法.

**特点:**

1.不用求Hessian矩阵;

2.不用求逆;

**拟牛顿条件**

$令{y}_k=J({x}_k+1)−J({x}_k), {s}_k={x}_{k+1}−{x}_k,有:$

$$
{y}_k=H({x}_{k+1}){s}_k\quad or \quad {s}_k=H({x}_{k+1})^{−1}{y}_k
$$

**- DFP法**
不含二阶导数的矩阵$U$(这里写成$D$区分$BFGS$)代替$H^{−1}$,拟牛顿条件写成:

$$
{s}_k=D_{k+1}{y}_k
$$

叠加方式求$D_{k+1}$,一般取$D_0=I$:

$$
{d}_k+1={d}_k+\Delta {d}_k,k=0,1,2...
$$

$\Delta {d}_k$确定(推导过程省略):

$$
\Delta {d}_k=\frac{{s}_k{s}_k^T}{{s}_k^T{y}_k}−\frac{{d}_k{y}_k{y}_k^T{d}_k}{{y}_k^T{d}_k{y}_k}
$$

**步骤:**

$step1.$给定初始点${x}_0 \in R^n,k=0,\rho \in(0,0.5), \tau\in(0,1),$以及最小误差$\xi;$

$step2.$判断${x}_k$是否满足终止条件,是则终止;

$step3.$确定$f(x)$在${x}_k$点的下降方向${d}_k=−{d}_kJ({x}_k );$

$step4.$计算$\alpha_k=\tau^m,m=m+1;$

$step5.$如果$f({x}_k +\alpha_k{d}_k)>f({x}_k )+\rho \alpha_kJ({x}_k )^T{d}_k$,Armijo准则,返回$step4$,否则继续;

$step6.$令${s}_k=\alpha_k{d}_k;{x}_k +1={x}_k +{s}_k;$

$step7.$计算${y}_k=J({x}_k +1)−J({x}_k );$

$step8.$计算$\Delta {d}_k=\frac{{s}_k{s}_k^T}{{s}_k^T{y}_k}−\frac{{d}_k{y}_k{y}_k^T{d}_k}{{y}_k^T{d}_k{y}_k};$

$step9. k=k+1;$返回$step2.$

**- BFGS法**

不含二阶导数的矩阵$U$(这里写成$B$区分$DFP$)代替$H$,拟牛顿条件写成:

$${y}_k=B_{k+1}{s}_k$$

叠加方式求$B_{k+1}$,一般取$B_0=I$:

$$B_{k+1}=B_k+\Delta B_k,k=0,1,2...$$

$\Delta B_k$确定(推导过程省略):

$$
\Delta B_k=\frac{{y}_k{y}_k^T}{{y}_k^T{s}_k}−\frac{B_k{s}_k{s}_k^TB_k}{{s}_k^TB{s}_k}
$$

利用$Sheman−Morrison$公式:

设$A\in R^n$为非奇异方正,$u,v\in R^n$,若$1+v^TA^{−1}u≠0$,则有:

$$
(A+uv^T)^{−1}=A^{−1}−\frac{A^{−1}uv^TA^{−1}}{1+v^TA^{−1}u}
$$

得到$B_{k+1}^{−1}$和$B_k^{−1}$的关系:

$$
B_{k+1}^{−1}=(I−\frac{{s}_k{y}_k^T}{{y}_k^T{s}_k})B_k^{−1}(I−\frac{{y}_k{s}_k^T}{{y}_k^T{s}_k})+\frac{{s}_k{s}_k^T}{{y}_k^T{s}_k}
$$

令${d}_k=B_k^{−1}$:

$$
D_{k+1}=(I−\frac{{s}_k{y}_k^T}{{y}_k^T{s}_k}){d}_k(\frac{I−{y}_k{s}_k^T}{{y}_k^T{s}_k})+\frac{{s}_k{s}_k^T}{{y}_k^T{s}_k}
$$

**步骤:**

$step1.$给定初始点${x}_0\in R^n,k=0,\rho\in (0,0.5), \tau\in (0,1),$以及最小误差$\xi;$

$step2.$判断${x}_k$是否满足终止条件,是则终止;

$step3.$确定$f(x)$在${x}_k$点的下降方向${d}_k=−{d}_kJ({x}_k);$

$step4.$计算$\alpha_k=\tau^m,m=m+1;$

$step5.$如果$f({x}_k+\alpha_k{d}_k)>f({x}_k)+\rho\alpha_kJ({x}_k)^T{d}_k,$Armijo准则,返回step4,否则继续;

$step6.$令${s}_k=\alpha_k{d}_k;x_{k+1}={x}_k+{s}_k;$

$step7.$计算${y}_k = J(x_{k+1})−J({x}_k);$

$step8.$计算$D_{k+1}=(I−\frac{{s}_k{y}_k^T}{{y}_k^T{s}_k}){d}_k(\frac{I−{y}_k{s}_k^T}{{y}_k^T{s}_k})+\frac{{s}_k{s}_k^T}{{y}_k^T{s}_k}$

$step9. k=k+1;$返回$step2.$

##### 举例

$y = 3x_1^2+3x_2^2-x_1^2+x_2,$初始点(4,3),$\xi=10^{-3}$

```c++
#include <Msnhnet/math/MsnhMatrixS.h>
#include <Msnhnet/cv/MsnhCVGui.h>
#include <iostream>

using namespace Msnhnet;

class NewtonLM
{
public:
    NewtonLM(int maxIter, double eps, double vk, double rho, double tau):_maxIter(maxIter),_eps(eps),_vk(vk),_rho(rho),_tau(tau){}


    void setMaxIter(int maxIter)
    {
        _maxIter = maxIter;
    }

    virtual int solve(MatSDS &startPoint) = 0;

    void setEps(double eps)
    {
        _eps = eps;
    }

    void setRho(double rho)
    {
        _rho = rho;
    }

    void setTau(double tau)
    {
        _tau = tau;
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
    double _vk  = 3;
    double _rho = 0.2;
    double _tau = 0.9;
    std::vector<Vec2F32> _xStep;
protected:
    virtual MatSDS calGradient(const MatSDS& point) = 0;
    virtual MatSDS calHessian(const MatSDS& point) = 0;
    virtual MatSDS calDk(const MatSDS& point) = 0;
    virtual MatSDS function(const MatSDS& point) = 0;
};


class NewtonLMProblem1:public NewtonLM
{
public:
    NewtonLMProblem1(int maxIter, double eps, double vk, double rho, double tau):NewtonLM(maxIter, eps, vk,rho,tau){}

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


    MatSDS calDk(const MatSDS& point) override
    {
        MatSDS J = calGradient(point);
        MatSDS H = calHessian(point);

        MatSDS I = MatSDS::eye(H.mWidth);


        MatSDS Hp  = H + _vk*I;

        if(!isPosMat(Hp))
        {
            H = H + 2*_vk*I;
        }
        else
        {
            H = Hp;
        }

        return -1*H.invert()*J;
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
            //这里就不用检查正定了
            MatSDS dk = calDk(x);

            double alpha = 1;

            //Armijo准则
            for (int i = 0; i < 100; ++i)
            {
                MatSDS left  = function(x + alpha*dk);
                MatSDS right = function(x) + this->_rho*alpha*calGradient(x).transpose()*dk;

                if(left(0,0) <= right(0,0))
                {
                    break;
                }

                alpha = alpha * _tau;
            }

            std::cout<<std::left<<"Iter(s): "<<std::setw(4)<<i<<", Loss: "<<std::setw(12)<<dk.L2()<<" Result: "<<function(x)[0]<<std::endl;

            x = x + alpha*dk;

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
    NewtonLMProblem1 function(1000, 0.001,3, 0.4, 0.8);
    MatSDS startPoint(1,2,{0,3});

    try
    {
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
        Gui::plotLine(u8"牛顿LM法迭代X中间值","x",function.getXStep());
        Gui::wait();
        }

    }
    catch(Exception ex)
    {
        std::cout<<ex.what();
    }
}


```

结果:对于初始点 **(4,3)** ,迭代9次即可完成,此点Newton法,DFP法都无解,BFGS方法有解,一般来说BFGS效果比较好.

![拟牛顿法9次完成求解](https://img-blog.csdnimg.cn/img_convert/39d3089c872545899fd73f43b914d3ae.png#pic_center)


![拟牛顿法X中间值可视化](https://img-blog.csdnimg.cn/img_convert/febf21fad0c39f2d6c8e28a6cacd279a.png#pic_center)


#### 4. 源码

https://github.com/msnh2012/numerical-optimizaiton(`https://github.com/msnh2012/numerical-optimizaiton`)

#### 5. 依赖包

https://github.com/msnh2012/Msnhnet(`https://github.com/msnh2012/Msnhnet`)


#### 6. 参考文献

1. Numerical Optimization. Jorge Nocedal Stephen J. Wrigh
2. Methods for non-linear least squares problems. K. Madsen, H.B. Nielsen, O. Tingleff.
3. Practical Optimization_ Algorithms and Engineering Applications. Andreas Antoniou Wu-Sheng Lu
4. 最优化理论与算法. 陈宝林
5. 数值最优化方法. 高立

网盘资料下载：链接：https://pan.baidu.com/s/1hpFwtwbez4mgT3ccJp33kQ 提取码：b6gq 


#### 7. 最后

- 欢迎关注我和BBff及公众号的小伙伴们一块维护的一个深度学习框架Msnhnet: https://github.com/msnh2012/Msnhnet
  Msnhnet除了是一个深度网络推理库之外，还是一个小型矩阵库，包含了矩阵常规操作，LU分解，Cholesky分解，SVD分解。

-----------------------------------------------------------------------------------------------
欢迎关注GiantPandaCV, 在这里你将看到独家的深度学习分享，坚持原创，每天分享我们学习到的新鲜知识。( • ̀ω•́ )✧

有对文章相关的问题，或者想要加入交流群，欢迎添加BBuf微信：

![二维码](https://img-blog.csdnimg.cn/20200110234905879.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)