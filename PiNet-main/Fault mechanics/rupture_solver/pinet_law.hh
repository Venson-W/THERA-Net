// PiNet field adapter for UGUCA. LGPL-3.0-or-later, consistent with UGUCA.
// Derived from RateAndStateLaw by D. S. Kammer, G. Albertini and C.-Y. Ke.
#pragma once
#include "rate_and_state_law.hh"
#include "interface.hh"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <limits>

class PiNetLaw : public uguca::RateAndStateLaw {
 public:
  uguca::NodalFieldComponent fref;
  bool advance_state=false; // Initialization equilibrates velocity at the supplied theta0.
  explicit PiNetLaw(uguca::BaseMesh &mesh, bool pc=true)
    : RateAndStateLaw(mesh,.008,.012,.02,1e-6,.6,1.,EvolutionLaw::AgingLaw,pc),
      fref(mesh) { fref.setAllValuesTo(.6); }
  uguca::NodalFieldComponent &getDc() { return Dc; }

  // Same exact frozen-speed aging step, evaluated without cancellation at small z.
  void computeTheta(uguca::NodalFieldComponent &target,
                    uguca::NodalFieldComponent &speed) override {
    const double dt=interface->getTimeStep();
    for(int i=0;i<mesh.getNbLocalNodes();++i) {
      if(!advance_state) { target(i)=theta(i); continue; }
      const double z=std::abs(speed(i))*dt/Dc(i);
      const double phi=z<1e-6 ? 1.-z/2.+z*z/6.-z*z*z/24. : -std::expm1(-z)/z;
      target(i)=theta(i)*std::exp(-z)+dt*phi;
      if(!(target(i)>0.) || !std::isfinite(target(i)))
        throw std::runtime_error("Non-positive/non-finite aging state");
    }
  }

  // A monotone scalar impedance/friction solve, with safeguarded Newton steps
  // in log(abs(V)). This changes the numerical root finder, not the RSF law.
  static double solve(double trial, double eta, double ap, double logZ, double previous) {
    if(trial==0.) return 0.;
    const double S=std::abs(trial);
    double lo=-745., hi=std::log(S/eta);
    double q=std::clamp(std::log(std::max(std::abs(previous),1e-20)),lo,hi);
    for(int it=0;it<100;++it) {
      const double u=logZ+q;
      const double y=u>30. ? u+std::log(2.) : std::asinh(std::exp(u));
      const double dy=u>30. ? 1. : std::exp(u)/std::hypot(1.,std::exp(u));
      const double ev=std::exp(q), residual=ap*y+eta*ev-S;
      if(std::abs(residual)<1e-7+1e-13*S)
        return std::copysign(ev,trial);
      if(residual>0.) hi=q; else lo=q;
      double qn=q-residual/(ap*dy+eta*ev);
      if(!std::isfinite(qn) || qn<=lo || qn>=hi) qn=.5*(lo+hi);
      if(std::abs(qn-q)<1e-13) return std::copysign(std::exp(qn),trial);
      q=qn;
    }
    throw std::runtime_error("PiNet RSF monotone root did not converge");
  }

  void computeCohesiveForces(uguca::NodalField &cohesion, bool predicting=false) override {
    interface->closingNormalGapForce(cohesion.component(1),predicting);
    interface->maintainShearGapForce(cohesion);
    const auto &top=interface->getTop();
    const auto &bot=interface->getBot();
    const double ft=top.getMaterial().getCs()/top.getMaterial().getShearModulus();
    const double fb=bot.getMaterial().getCs()/bot.getMaterial().getShearModulus();
    const double eta=1./(ft+fb);
    uguca::NodalField gap(mesh);
    interface->computeGapVelocity(gap,false);
    gap.computeNorm(V,1);
    auto &state=predicting ? theta_pc : theta;
    computeTheta(state,V);
    for(int i=0;i<mesh.getNbLocalNodes();++i) {
      const double ap=a(i)*std::abs(cohesion.component(1)(i));
      const double logZ=(fref(i)+b(i)*std::log(V0*state(i)/Dc(i)))/a(i)-std::log(2.*V0);
      const double S=interface->getLoad().component(0)(i)
          +eta*(ft*top.getInternal().storage(0)[i]+fb*bot.getInternal().storage(0)[i]);
      const double v=solve(S,eta,ap,logZ,V(i));
      const double u=logZ+std::log(std::max(std::abs(v),std::numeric_limits<double>::min()));
      const double y=u>30. ? u+std::log(2.) : std::asinh(std::exp(u));
      cohesion.component(0)(i)=v==0. ? 0. : std::copysign(ap*y,v);
    }
  }
};
