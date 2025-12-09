Black Hole Spin Alignment Classifier 

An E(3)-equivariant neural network for classifying black hole spin alignment in binary systems, built using the e3nn library

-- 

Project Overview: 

This project explores whether equivariant neural networks can rapidly classify black hole spin alignment, a key observable for distinguishing formation channels in gravitational wave astronomy. Using the e3nn library, I built a rotation-invariant classifier that achieves near-perfect accuracy on synthetic binary black hole systems.

 **Results:**
- Training Accuracy: 99.8%
- Validation Accuracy: 100%
- Equivariance Test: 100%

Motivation: Why Spin Alignment Matters

Primary black hole spin alignment reveals its history:
    - Aligned (χ_eff > +0.3): Isolated binary evolution
    - Misaligned (χ_eff ≈ 0): Dynamical capture (clusters, AGN)
    - Anti-aligned (χ_eff < 0): Hypothesized hierarchical mergers 

 Current LIGO observations show a mix of these alignments, suggesting different ways a black hole may form across time. 

Why Equivariance?: 

Standard neural networks fail for astrophysical classification because our observations are viewing-angle dependent, but the physics which describe the object is not. Standard MLP models learn coordinate-dependent patterns. If the system is rotated 90°, the model will see both sets of inputs as completely different.

E(3)-equivariant networks respect the symmetries of a three-dimensional space across translations and rotations. It recognizes physical relationships regardless of observer position, binary system orientation, or coordinate system choice. When observers such as LIGO Hanford and LIGO Livingston operate under different reference frames (3000km apart), this makes the ML model learn significantly easier. 


The Process:

Input: 
Spin vectors (v1 & v2) from binary black hole systems 
- -> Tensor product (v1 ⊗ v2): Computes geometric features (dot product, cross product, etc)
- -> Batch Normalization
- -> MLP Classifier (10 -> 16 -> 2)
Output: 
Aligned or Misaligned

Key Learning:
- I studied and implemented an E(3)-equivariant ML model, understanding concepts such as irreducible representations.
- The importance of equivariance, and fully visualizing its effects. 
- Further bridging together mathematics and physics with concepts such as tensor product and how it relates to x_eff (effective spin) commonly used in LIGO parameters.
- This project is a pivotal point in my learning. I handled incredibly complex topics and documentation while completely understanding every part of the code. 

Future Improvements:
 - Proof of concept: needs validation on real LIGO data.
 - Incorporate the quantified advantage over standard neural network models.
 - Simplified the physics: Should expand more to accurately match parameters found in black hole detections

**Contact**
- Bradyn Livingston
- bradynlivingston0053@gmail.com
- Indiana University - Indianapolis


Developed as an independent exploration of geometric deep learning for gravitational wave astronomy.



