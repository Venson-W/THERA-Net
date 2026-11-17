// Numerical checks for the THERA-Net adapter. LGPL-3.0-or-later.
#include "thera_law.hh"
#include <iostream>

int main() {
  const double eta=2670.*3464./2., pressure=120e6;
  for(double a : {.006, .008, .016, .018})
    for(double theta : {.001, 1., 1e12, 1e24})
      for(double v : {-10., -.001, -1e-12, 1e-12, .001, 10.}) {
        const double log_beta=(.6+.012*std::log(1e-6*theta/.02))/a-std::log(2e-6);
        const double expected_tau=pressure*a*std::asinh(std::exp(log_beta)*v);
        const double driving=eta*v+expected_tau;
        const double actual=THERALaw::solve(driving,eta,pressure*a,log_beta,1e-12);
        const double residual=eta*actual+pressure*a*std::asinh(std::exp(log_beta)*actual)-driving;
        if(!std::isfinite(actual) || std::abs(residual)>2e-7+2e-13*std::abs(driving)
            || actual*v<0. || std::abs(actual-v)>2e-10*std::abs(v)+1e-20) {
          std::cerr<<"Impedance check failed: "<<a<<" "<<theta<<" "<<v<<" "<<actual<<"\n";
          return 1;
        }
      }
  if(THERALaw::solve(0.,eta,1e6,100.,1.)!=0.) return 1;
  std::cout<<"96 signed impedance roots and zero driving passed\n";
}
