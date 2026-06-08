# Deep Learning

**Deep Learning**: Involves work with neural networks. **Neural networks** are the core principle behind Deep Learning. 

- **AI** involves **ML**, and **ML** involves **DL** (Deep Learning).
- Neural networks are highly computationally expensive.
- So, we had ML algorithms that were faster than NN (Neural Networks), and although the algorithms existed, they couldn't be implemented or used due to computational limitations.

**Q: Why use neural networks? Why shift to a new algorithm when existing methods of machine learning already exist?**

![image.png](Deep%20Learning%206623443c8e164419a4f4d3dfb1b4b66a/image.png)

### Logistic Regression Algorithm

- **Consider the logistic regression algorithm**, where Q (weights) has to be learned. The hypothesis function is as follows: -
    
    $$
    \color{black} S ( x ) = \frac { 1 } { 1 + e ^ { - Q x } }
    $$
    
    $$
    \color{black}   If  \space S ( x ) > 0.5 \implies Q. x > 0\rightarrow Class-1
    $$
    
    $$
    \color{black}  If  \space S ( x ) \leq 0.5 \implies Q .x < 0\rightarrow Class-0
    $$
    
    Thus, the **hypothesis function** states that: - 
    
    - If S(x)>0.5, then output **class 1**.
    - If S(x)≤0.5, then output **class 0**.
    
    Here even if Q.x is a non-linear function, the decision boundary will still be linear,  due to our limits on θ, due to the limits on Q.x >0 or Q.x <0
    

---

We would have to add features with different mathematical functions, for **logistic regression to create a non-linear decision boundary**.

Say a dataset has features X1, X2, X3 and X4. The new features for a non-linear boundary could be represented as:

 $X_1, X_2, X_3, X_2^2, X_3^3, X_4^4, sin(x4), log(X^3) , etc.$

**Problem**: The features we had to add were all fit and fixed, which might or might not increase accuracy.

Even with **SVM (Support Vector Machine)**, we used non-linear kernels. Still, the kernel parameters had to be selected by us, such as what kind of kernel to use, and the hyperparameters, etc.

### Neural Networks

- **Neural networks** are used to solve the above problem of non-linear decision boundary.

### Q: What is the neural network doing?

**Ans:** Consider logistic regression (a two feature dataset)

If  $y=m_1z_1+m_2z_2+m_3z_3+m_3z_0$

***### Missing illustration for neural network***

y = m_1 z_1 + m_2 z_2 + m_3 z_3 + m_3 z_0

y=m1z1+m2z2+m3z3+m3z0

This represents a **featured dataset** that is then classified into **class 0** or **class 1**.

---

### Neural Networks for the Same 2-Feature Dataset

For the equation:

$y= w_1 x_1 + w_2 x_2 + w_3 x_3$

***(Missing illustration for Neural Networks)***

![image.png](Deep%20Learning%206623443c8e164419a4f4d3dfb1b4b66a/image%201.png)

### Important Points:

- In **neural networks**, we can use any function that we want.
- The output f3​ is **not** linearly dependent on x0​,x1​,x2​.
    
    f3f_3
    
    x0,x1,x2x_0, x_1, x_2
    
- The **decision boundary** is non-linear in nature. It depends on the weights (e.g., w00​,w01​,w10​,w11​) and the functions (e.g., f1​,f2​,f3​) used.
    
    w00,w01,w10,w11w_{00}, w_{01}, w_{10}, w_{11}
    
    f1,f2,f3f_1, f_2, f_3
    

Understanding Linear Decision Boundary Using Logistic Regression / Simple Neural Network