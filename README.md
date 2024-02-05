**CHIST - Oveview**

Given a user provided "function" over a physical "domain"
    $Y(X) \quad \text{where} \quad X \in \Omega_X$
this package uses the fit method to return the "CDF" & "PDF"
    $F_Y(y), f_Y(y)$
over their corresponding probability space $\Omega_Y$. The method uses a finite element discretisation consisting of n elements.

**Example**

As an example we construct the PDF & CDF of the function $Y(x_1) = x^{3/2}$ on the interval $x_1 \in [0,1]$. To do so we first import

`from PDF_Projector import FEptp`

where FEptp which stands for Finite Element physical to probability is the object we will use. To instantiate this object we can call

`ptp = FEptp()`

or we can specify the function spaces for the CDF and PDF

`ptp = FEptp(func_space_CDF = {"family":"DG","degree":1},func_space_PDF= {"family":"CG","degree":1})`

Having specified the function spaces for our problem, we must then specify the phyiscal domain $\Omega_X$, the domain of the CDF & PDF $\Omega_Y$ and the number of elements we will use. This is done by calling the domain method

`x1,y = ptp.domain(Omega_X = {'x1':(0,1)}, Omega_Y = {'Y':(0,1)}, N_elements=100)`

which returns the co-ordinates of the domain $x_1,y$. Armed with the co-ordinates we can then specify our function $Y(x_1)$ and generate the CDF & PDF by calling the fit method
    
`ptp.fit(function_Y = x1**(3/2), quadrature_degree=1000)`

Once fitted we can then plot these functions by calling `ptp.plot()` or evaluate them on a user specified grid y by calling

`F_Y,f_Y,y_i  = ptp.evaluate(y = [0., 0.1, 0.2])`

which returns the CDF F_Y & PDF f_y on the grid y_i.

In two dimensions the method works exactly the same way and if we are happy with the function spaces chosen we can simply re-run

`x1,x2,y = ptp.domain(Omega_X = {'x1':(0,1),'x2':(0,1)}, Omega_Y = {'Y':(0,2)}, N_elements=50)`

to generate the domain, and then call

`ptp.fit(function_Y = x1 + x2, quadrature_degree=200)`

to generate the finite element approximation of $Y(X) = x_1 + x_2$.
