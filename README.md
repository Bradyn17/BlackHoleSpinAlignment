Black Hole Spin Alignment Classifier 

An E(3)-equivariant neural network for studying simplified black hole spin alignment in binary systems, built using the e3nn library

-- 

Project Overview: 

This project explores whether equivariant neural networks can rapidly classify black hole spin alignment, a key observable for distinguishing formation channels in gravitational wave astronomy. Using the e3nn library, I built a rotation-invariant classifier that achieves near-perfect accuracy on synthetic binary black hole systems.

 **Results:**
- Training Accuracy: 99.8%
- Validation Accuracy: 100%
- Maximum Rotation Invariance Error: 3.81 * 10^-6
- Mean Rotation Invariance Error: 3.38 * 10^-7

Motivation: Why Spin Alignment Matters

Primary black hole spin alignment reveals its history:

Black hole spin alignment can provide information about the history of black hole systems. Different formations may produce different statistical distributions of spin orientations. Isolated formation may favor aligned configurations, while dynamical formations may favor more randomly oriented spins. The effective spin paramater (x_eff) is a parameter involved in the spin-orbit relationship. 
Why Equivariance?: 

Standard neural networks fail for astrophysical classification because our observations are viewing-angle dependent, but the physics which describe the object is not. Standard MLP models learn coordinate-dependent patterns. If the system is rotated 90°, the model will see both sets of inputs as completely different.

E(3)-equivariant networks respect the symmetries of a three-dimensional space across translations and rotations. It recognizes physical relationships regardless of observer position, binary system orientation, or coordinate system choice. 


The Process:

Input: 
Spin vectors (v1 & v2) representing simplified black hole spin-alignment geometry
- -> Tensor product (v1 ⊗ v2): Produces rotationally invariant scalar features containing dot-product-like geometric information
- -> Batch Normalization
- -> MLP Classifier (10 -> 16 -> 2)
Output: 
Parallel or Perpendicular

Key Learning:
- I studied and implemented an E(3)-equivariant ML model, understanding concepts such as irreducible representations.
- The importance of equivariance, and fully visualizing its effects. 
- Further bridging together mathematics and physics with concepts such as tensor product and how it relates to x_eff (effective spin) commonly used in gravitational wave astronomy.
- This project is a pivotal point in my learning. I handled incredibly complex topics and documentation while completely understanding every part of the code. 

Future Improvements:
 - Proof of concept: needs validation on real black hole data.
 - Incorporate the quantified advantage over standard neural network models.
 - Simplified the physics: Should expand more to accurately match parameters found in black hole detections
 - Try to predict aligniment from different angles than the binary case I used

**Contact**
- Bradyn Livingston
- bradynlivingston0053@gmail.com
- Indiana University - Indianapolis


Developed as an independent exploration of geometric deep learning for gravitational wave astronomy.



