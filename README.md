# Neural System Level Synthesis

PyTorch implementation of Neural System Level Synthesis using Recurrent Equilibrium Networks, 
as presented in "Neural System Level Synthesis: 
Learning over All Stabilizing Policies for Nonlinear Systems".

## Implementation details

The [implementation details](docs/implementation_details.pdf) can be found in the ```docs``` folder.

## Installation

```bash
git clone https://github.com/DecodEPFL/neurSLS.git

cd neurSLS

python setup.py install
```

## Basic usage

Two environments of robots in the <i>xy</i>-plane are proposed to train the neurSLS controllers.
Firstly, we propose the problem _mountains_, where two agents need to pass through a narrow corridor 
while avoiding collisions.
Secondly, in the problem _swapping_, 12 robots need to switching positions while avoiding collisions among them.

To train the controllers, run the following script:
```bash
./run.py --sys_model [SYS_MODEL]
```
where available values for `SYS_MODEL` are `corridor` and `robots`. 

## Examples: 

### Mountains problem (2 robots)

The following gifs show trajectories of the 2 robots before and after the training of a neurSLS controller, 
where the agents that need to coordinate in order to pass through a narrow passage, 
starting from a random initial position marked with &#9675;, sampled from a Normal distribution centered in 
[&#177;2 , -2] with standard deviation of 0.5.

<p align="center">
<img src="./figures/corridorOL.gif" alt="robot_trajectories_before_training" width="400"/>
<img src="./figures/corridor.gif" alt="robot_trajectories_after_training_a_neurSLS_controller" width="400"/>
</p> 

### Swapping problem (12 robots)

The following gifs show the trajectories of the 12 robots before and after the training of a neurSLS controller, 
where the agents swap their initial fixed positions, while avoiding all collisions.

<p align="center">
<img src="./figures/robotsOL.gif" alt="robot_trajectories_before_training" width="400"/>
<img src="./figures/robots.gif" alt="robot_trajectories_after_training_a_neurSLS_controller" width="400"/>
</p> 

### Early stopping of the training
We verify that neurSLS controllers ensure closed-loop stability by design even during exploration. 
Results are reported in the following gifs, where we train the neurSLS controller 
for 0\%, 25\%, 50\% and 75\% of the total number of iterations.

**Mountains problem**: <br>
Note that, for this example, we fix the initial condition to be [&#177;2 , -2] and 
we train for 150 epochs.
<p align="center">
<img src="./figures/corridor0.gif" alt="mountains_0_training" width="200"/>
<img src="./figures/corridor25.gif" alt="mountains_25_training" width="200"/>
<img src="./figures/corridor50.gif" alt="mountains_50_training" width="200"/>
<img src="./figures/corridor75.gif" alt="mountains_75_training" width="200"/>
</p> 

**Swapping problem**:
<p align="center">
<img src="./figures/robots0.gif" alt="robot_trajectories_0_training" width="200"/>
<img src="./figures/robots25.gif" alt="robot_trajectories_25_training" width="200"/>
<img src="./figures/robots50.gif" alt="robot_trajectories_50_training" width="200"/>
<img src="./figures/robots75.gif" alt="robot_trajectories_75_training" width="200"/>
</p>

In both cases, the training is performed for _t_ &in; [0,5].  
Partially trained distributed controllers exhibit suboptimal behavior, but never 
compromise closed-loop stability.

<!-- 
## Implementation details

We consider a fleet of mobile robots that need to asymptotically achieve a pre-specified formation 
described by (p&#x0304;<sub>x</sub> , p&#x0304;<sub>y</sub>) for each agent _i_. The requirements are:

R1) The control policy asymptotically steers the agents to the target position (closed-loop stability)

R2) A cost function is minimized over a finite horizon of 5 seconds



Each robot _i_ is modeled as a point-mass vehicle with position 
_p<sub>t</sub>_ &in;&reals;<sup>2</sup> and velocity 
_q<sub>t</sub>_ &in;&reals;<sup>2</sup> subject to nonlinear drag forces 
(e.g., air or water resistance). 
The discrete-time model for each vehicle of mass _m_ is 

p<sub>t</sub> = p<sub>t-1</sub> + T<sub>s</sub> q<sub>t-1</sub>

q<sub>t</sub> = q<sub>t-1</sub> - T<sub>s</sub> 1/m C(q<sub>t-1</sub>) q<sub>t-1</sub> + u<sub>t-1</sub>

where _u<sub>t</sub>_ denotes the force control input, _T<sub>s</sub>_ > 0 is the sampling time
and C(·) is a positive <i>drag function</i>.
We consider a  base controller 

u<sub>t</sub> = K'(p&#x0304;-p<sub>t</sub>) 

with _K'_ = diag(_k_,_k_) and _k_ > 0,

which is _strongly stabilizing_. 


Given a set of _N_ vehicles, we denote the overall state as X = (x<sup>1</sup>, ..., x<sup>N</sup>) 
and control input U = (u<sup>1</sup>, ..., u<sup>N</sup>) as the stacked state and input vectors 
of each agent, respectively. Then, the cost function to minimize is given by

L = &sum; (L<sub>x</sub>(t) + a<sub>u</sub> L<sub>u</sub>(t) + 
a<sub>ca</sub> L<sub>ca</sub>(t) + a<sub>obst</sub> L<sub>obst</sub>(t) 

where the sum is done over t = 0, 1, ..., T, 
and a<sub>u</sub>, a<sub>ca</sub>, a<sub>obst</sub> > 0 are hyperparameters to be set.


The first two addends represent the cost of the state and input signals respectively and are given by: 

L<sub>x</sub>(t) = X<sup>&top;</sup>(t) Q X(t), &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp;
L<sub>u</sub>(t) = U<sup>&top;</sup>(t) U(t).

where _Q_ is a predefined weighted matrix.


The third addend strongly penalizes the event that any two robots _i_ and _j_ find themselves 
at a distance _d_<sub>ij</sub>(t) < _D_, where _D_ &in;&reals;<sup>+</sup> is a pre-defined safety distance.
It is given by

L<sub>ca</sub>(t) = &sum;<sub>i,j, i&ne;j</sub> (d<sub>ij</sub>(t) + &epsilon;)<sup>-2</sup> 
if d<sub>ij</sub>(t) &le; D, and L<sub>ca</sub>(t) = 0 otherwise,

where &epsilon; > 0 is a small constant such that L<sub>ca</sub>(t) < &infin;.


The last addend penalizes the _xy_ position of each robot according to a predefined map with obstacles.
For the corridor case, we modelled each obstacle as a Gaussian centered at (&pm;1.5, 0) and (&pm;2.5, 0)
with covariance diag(0.2, 0.2).


### Mountains problem (2 robots)

The system consists of 2 robots of radius 0.5<i>m</i> and mass m = 1<i>kg</i>. 
The drag function is given by 

C(q)q = b<sub>1</sub> q + b<sub>2</sub> |q| q, 

with b<sub>1</sub> = 1<i>N·s/m</i> and b<sub>2</sub> = 0.1<i>N·s/m</i>. 
For the based controller _K'_, _k<sub>1</sub>_, _k<sub>2</sub>_ = 1<i>N/m</i>.

The REN is a deep neural network with depth _r_ = 32 layers (_v_ &in; &reals;<sup>r</sup> ).
Its internal state &xi; is of dimension _q_ = 32.

We train Neur-SLS control policies to optimize the performance over a horizon of _5_ seconds 
with sampling time T<sub>s</sub>=0.05 seconds, resulting in _T_ = 100 time-steps.
We use gradient descent with Adam for 500 epochs in order to minimize the loss function with 
hyperparameters Q = diag(1,1,1,1), a<sub>u</sub> = 0.1, a<sub>ca</sub> = 100 and a<sub>obst</sub> = 5000.
The initial positions of the robots are sampled from a Normal distribution centered at (&pm;2, -2), with
covariance diag(&sigma;<sup>2</sup>, &sigma;<sup>2</sup>).
We set &sigma; = 0.2 for the first 300 epochs and then increased it to &sigma; = 0.5.
At each epoch we simulate five trajectories over which we calculate the corresponding loss.
The learning rate is set to 0.001.


### Swapping problem (12 robots)

The system consists of 12 robots of radius 0.25<i>m</i> and mass m = 1<i>kg</i>. 
The drag function is given by C(q)q = bq, with b = 1<i>N·s/m</i>. 
For the based controller _K'_, we set _k<sub>1</sub>_, _k<sub>2</sub>_ = 1<i>N/m</i>.

The REN is a deep neural network with depth _r_ = 24 layers (_v_ &in; &reals;<sup>r</sup> ).
Its internal state &xi; is of dimension _q_ = 96.

We train Neur-SLS control policies to optimize the performance over a horizon of _5_ seconds 
with sampling time T<sub>s</sub>=0.05 seconds, resulting in _T_ = 100 time-steps.
We use gradient descent with Adam for 1500 epochs in order to minimize the loss function with 
hyperparameters Q = diag(1,1,1,1), a<sub>u</sub> = 0.1 and a<sub>ca</sub> = 1000. 
Since there are no fixed obstacles in the environment, we set a<sub>obst</sub> = 0.
The learning rate is set to 0.002 for the entire training.
 -->

## License
This work is licensed under a
[Creative Commons Attribution 4.0 International License][cc-by].

[![CC BY 4.0][cc-by-image]][cc-by] 

[cc-by]: http://creativecommons.org/licenses/by/4.0/
[cc-by-image]: https://i.creativecommons.org/l/by/4.0/88x31.png
[cc-by-shield]: https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg


## References
[[1]](https://arxiv.org/pdf/2203.11812.pdf) Luca Furieri, Clara Galimberti, Giancarlo Ferrari-Trecate.
"Neural System Level Synthesis: Learning over All Stabilizing Policies for Nonlinear Systems,"
arXiv:2203.11812, 2022.
